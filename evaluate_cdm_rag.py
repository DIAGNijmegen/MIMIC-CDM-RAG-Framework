"""Evaluate saved CDM trajectories using configs/evaluate.yaml and Hydra overrides."""

import glob
import importlib
import pickle
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file

SUPPORTED_PATHOLOGIES = (
    "appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"
)


def load_evaluator(pathology):
    """Create a fresh evaluator without importing the inference entry point."""
    if pathology not in SUPPORTED_PATHOLOGIES:
        raise ValueError(f"Unsupported pathology: {pathology}")
    module = importlib.import_module(f"evaluators.{pathology}_evaluator")
    return getattr(module, f"{pathology.capitalize()}Evaluator")()


def calculate_average(evals, field, pathology):
    if not evals:
        raise ValueError(f"No patients evaluated for {pathology}; cannot average {field}.")
    return sum(evaluation["scores"][field] for evaluation in evals.values()) / len(evals), len(evals)


def calculate_percentages(evals, field):
    base_field = field.removesuffix(" Percentage")
    for evaluation in evals.values():
        evaluation["scores"][field] = evaluation["scores"][base_field] / evaluation["max_scores"][base_field]
    return evals


def count_unnecessary(evals, field):
    for evaluation in evals.values():
        evaluation["scores"][field] = len(evaluation["answers"][field])
    return evals


def reference_for(patient):
    return tuple(patient[field] for field in (
        "Discharge Diagnosis", "ICD Diagnosis", "Procedures ICD9",
        "Procedures ICD10", "Procedures Discharge",
    ))


def collect_retrievals(target, retrieved_chunks):
    """Count retrieved chunks, preserving the legacy PDF and MedCPT schemas."""
    for chunks in (retrieved_chunks or {}).values():
        for chunk in chunks:
            if chunk.get("format") == "medcpt":
                tags = chunk.get("tags", "").lower()
                pathologies = [p for p in SUPPORTED_PATHOLOGIES if p in tags] or ["other"]
                for pathology in pathologies:
                    document = target.setdefault(pathology, {}).setdefault(
                        chunk.get("document_reference", "N/A"), {}
                    )
                    entry = document.setdefault(chunk.get("chunk_id", "0"), {
                        "id": chunk.get("chunk_id", "0"), "count": 0,
                        "content": chunk.get("content", ""),
                    })
                    entry["count"] += 1
            else:
                page = target.setdefault(chunk["document_reference"], {}).setdefault(chunk["page_number"], {})
                entry = page.setdefault(chunk["order_in_document"], {
                    "id": chunk["chunk_id"], "count": 0, "content": chunk.get("content", ""),
                })
                entry["count"] += 1


def find_results(args, model, pathology, experiment):
    log_dir = Path(to_absolute_path(args.logs_dir))
    if experiment == args.vanilla_experiment:
        run_suffix = "*[0-9]"
    elif experiment == "":
        run_suffix = "*"
    else:
        run_suffix = f"*_{experiment}"
    prefix = f"{pathology}_{args.agent}_{model}_"
    pattern = str(log_dir / f"{prefix}{run_suffix}" / f"{prefix}*_results.pkl")
    matches = glob.glob(pattern)
    if "_SEED=" not in model:
        # The wildcard also matches seeded runs; exclude them for the base seed.
        matches = [path for path in matches if "_SEED=" not in Path(path).name]
    matches = sorted(matches, key=lambda path: (len(path), path))
    if not matches:
        raise FileNotFoundError(f"No results for {model}/{pathology}/{experiment!r}. Searched: {pattern}")
    if len(matches) > 1:
        if args.multiple_matches == "error":
            raise ValueError(f"Multiple result files match {pattern}: {matches}")
        tqdm.write(f"Found {len(matches)} files; using shortest path: {matches[0]}")
    return matches[0]


def save_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)

def evaluate(args: DictConfig):
    """Evaluate configured models and experiments, saving compatible pickle artifacts."""
    if not args.models or not args.pathologies or not args.experiments:
        raise ValueError("models, pathologies, and experiments must be nonempty lists.")
    if type(args.base_seed) is not int or any(type(seed) is not int for seed in args.seeds):
        raise ValueError("base_seed and every entry in seeds must be integers.")
    if args.multiple_matches not in ("shortest", "error"):
        raise ValueError("multiple_matches must be 'shortest' or 'error'.")
    for pathology in args.pathologies:
        if pathology not in SUPPORTED_PATHOLOGIES:
            raise ValueError(f"Unsupported pathology: {pathology}")
    for experiment in args.experiments:
        if experiment and (Path(experiment).name != experiment or experiment in (".", "..")):
            raise ValueError("Experiment names must be single directory names or an empty string.")
    models = list(dict.fromkeys(
        model if seed == args.base_seed else f"{model}_SEED={seed}"
        for model in args.models for seed in (args.seeds or [args.base_seed])
    ))
    base_mimic = to_absolute_path(args.base_mimic)
    with open(to_absolute_path(args.id_difficulty_path), "rb") as handle:
        id_difficulty = pickle.load(handle)
    patient_data = {
        pathology: load_hadm_from_file(f"{pathology}_hadm_info_{args.difficulty}", base_mimic=base_mimic)
        for pathology in args.pathologies
    }
    # Resolve every input before writing outputs so an invalid selection fails early.
    result_paths = {
        (experiment, model, pathology): find_results(args, model, pathology, experiment)
        for experiment in args.experiments for model in models for pathology in args.pathologies
    }
    for experiment in args.experiments:
        output_dir = Path(to_absolute_path(args.output_dir)) / experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(args, output_dir / "evaluation_config.yaml", resolve=True)
        combined = {name: {} for name in ("evals", "results", "scores", "retrievals")}
        for model in models:
            all_evals, all_results, all_retrievals = {}, {}, {}
            for pathology in args.pathologies:
                result_path = result_paths[experiment, model, pathology]
                tqdm.write(f"Loading {result_path}")
                results = {key: value for record in read_from_pickle_file(result_path) for key, value in record.items()}
                all_evals[pathology], all_results[pathology], all_retrievals[pathology] = {}, {}, {}
                for patient_id in tqdm(id_difficulty[pathology][args.difficulty], desc=f"Processing {pathology}"):
                    if patient_id not in results:
                        continue
                    result = results[patient_id]
                    evaluation = load_evaluator(pathology)._evaluate_agent_trajectory(
                        prediction=result["output"], input=result["input"],
                        reference=reference_for(patient_data[pathology][patient_id]),
                        agent_trajectory=result["intermediate_steps"],
                        retrieved_chunks=result.get("retrieval"),
                    )
                    all_evals[pathology][patient_id] = evaluation
                    all_results[pathology][patient_id] = result
                    collect_retrievals(all_retrievals[pathology], result.get("retrieval"))
            averages = {}
            for field in args.score_fields:
                averages[field] = {}
                for pathology in args.pathologies:
                    if field in ("Unnecessary Laboratory Tests", "Unnecessary Imaging"):
                        count_unnecessary(all_evals[pathology], field)
                    elif field.endswith(" Percentage"):
                        calculate_percentages(all_evals[pathology], field)
                    averages[field][pathology], _ = calculate_average(all_evals[pathology], field, pathology)
            artifacts = dict(evals=all_evals, results=all_results, scores=averages, retrievals=all_retrievals)
            for name, value in artifacts.items():
                combined[name][model] = value
                save_pickle(output_dir / f"{model}_{name}.pkl", value)
        for name, value in combined.items():
            save_pickle(output_dir / f"{name}.pkl", value)
        tqdm.write(f"Saved evaluation to {output_dir}")


@hydra.main(config_path="./configs", config_name="evaluate", version_base=None)
def run(args: DictConfig):
    evaluate(args)


if __name__ == "__main__":
    run()

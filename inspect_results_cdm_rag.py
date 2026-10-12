"""Export the combined STACK, treatment and diagnostic analyses as CSV files.

Configure configs/inspect.yaml or pass Hydra key=value overrides when running
this script. Every export uses the same cohort and model selection. Figures,
LaTeX tables and model-name substitutions are intentionally absent. Clinical
imaging measurements are still exported as CSV data.

Only one model's results/evaluations/retrievals are held in memory at a time.
The clinician laboratory baseline uses a fresh evaluator for each patient.
"""

import copy
import gc
import importlib
import os
import pickle
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import hydra
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from agents.AgentAction import AgentAction
from dataset.utils import load_hadm_from_file

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


MIMIC_BASE = "/container/data"
EXPERIMENTS_BASE = "/container/data/logs/analysis"
OUTPUT_BASE = "/container/data/logs/analysis/output"
DIAG = "Gracious Diagnosis"
DIFFICULTY = "first_diag"
PATHOLOGIES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
DIAGNOSTIC_CLASSES = list(PATHOLOGIES)

# Defaults for direct use of the analysis helpers. CLI defaults live in
# configs/inspect.yaml; main(config) applies them for the duration of that run.
EXPERIMENTS = ["BIN"]
MODELS = [
    "gemma-2-9b-it-exl2-4.0bpw",
    "gemma-2-9b-it-exl2-4.0bpw_stella_en_400M_v5_pdf",
]
BASE_SEED = 2023
SEEDS = [BASE_SEED]
SKIP_MISSING_MODEL_FILES = True
VALIDATION_TOLERANCE = 1e-12


def load_evaluator(pathology):
    """Create a fresh clinical evaluator without importing the inference runner."""
    if pathology not in PATHOLOGIES:
        raise ValueError(f"Unsupported pathology: {pathology}")
    module = importlib.import_module(f"evaluators.{pathology}_evaluator")
    return getattr(module, f"{pathology.capitalize()}Evaluator")()


def _expand_seeded_models_as_independent(models, seeds):
    """Select only the requested seeds; BASE_SEED uses the unsuffixed label."""
    expanded = []
    for model in models:
        if "_SEED=" in model:
            expanded.append(model)
            continue
        for seed in seeds:
            expanded.append(model if seed == BASE_SEED else f"{model}_SEED={seed}")
    return expanded


def split_seeded_model_name(model_name):
    """
    Returns:
        base_model, seed

    The unsuffixed model is explicitly treated as BASE_SEED.
    """
    marker = "_SEED="
    if marker not in model_name:
        return (model_name, BASE_SEED)
    (base_model, seed_text) = model_name.rsplit(marker, 1)
    return (base_model, int(seed_text))


def _empty_pathology_dict(pathos=PATHOLOGIES):
    return {patho: {} for patho in pathos}


def _pickle_path(experiment, model, suffix, f1=False):
    return os.path.join(
        EXPERIMENTS_BASE, experiment, f"{model}_{suffix}{('_f1' if f1 else '')}.pkl"
    )


def _select_by_difficulty(
    raw,
    id_difficulty,
    difficulty,
    pathos=PATHOLOGIES,
    reference_raw=None,
    experiment=None,
    model=None,
):
    """Subset a model payload to the ids used by the requested difficulty split."""
    selected = {}
    for patho in pathos:
        selected[patho] = {}
        raw_patho = raw.get(patho, {}) if isinstance(raw, dict) else {}
        ref_patho = (
            reference_raw.get(patho, raw_patho)
            if isinstance(reference_raw, dict)
            else raw_patho
        )
        wanted_ids = id_difficulty.get(patho, {}).get(
            difficulty, list(raw_patho.keys())
        )
        for _id in wanted_ids:
            if _id not in ref_patho:
                if experiment is not None and model is not None:
                    print(
                        f"For experiment {experiment} and model {model}, {_id} not in {patho}"
                    )
                continue
            if _id in raw_patho:
                selected[patho][_id] = raw_patho[_id]
    return selected


def get_diagnosis_score(patient_eval):
    """
    Return the raw Gracious Diagnosis score.

    This script requires it to be binary (0 or 1), because the claim
    Sensitivity == per-class Diagnostic Accuracy only holds under binary
    correctness.
    """
    if not isinstance(patient_eval, dict):
        return np.nan
    scores = patient_eval.get("scores", {})
    if not isinstance(scores, dict):
        return np.nan
    return scores.get(DIAG, np.nan)


def normalize_binary_diag_score(value):
    """
    Convert valid 0/1/bool values to integer 0/1.

    Anything else raises an error rather than silently changing the metric.
    """
    if pd.isna(value):
        raise ValueError(f"Missing {DIAG!r} score.")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        numeric = float(value)
    except Exception:
        raise ValueError(f"Non-numeric {DIAG!r} score: {value!r}")
    if numeric == 0.0:
        return 0
    if numeric == 1.0:
        return 1
    raise ValueError(
        f"{DIAG!r} is not binary: {value!r}. The sensitivity/accuracy equivalence must not be assumed."
    )


def normalize_text(value):
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    text = text.replace("▁", " ")
    text = text.replace("\r", "\n")
    text = re.sub("[ \\t]+", " ", text)
    text = re.sub("\\n+", "\n", text)
    return text.strip()


def compact_text(value, max_chars=1500):
    text = normalize_text(value)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


NEGATION_WORDS = [
    "no ",
    "not ",
    "without ",
    "negative for ",
    "unlikely ",
    "unlikely to be ",
    "ruled out ",
    "rule out ",
    "absence of ",
    "no evidence of ",
]


def target_mentions(text):
    """
    Detect positive mentions of the four target diagnoses in an ISOLATED
    final-diagnosis field.

    This function is deliberately NOT used on the whole reasoning/output.
    """
    text = normalize_text(text).lower()
    found = []
    for pathology in DIAGNOSTIC_CLASSES:
        for match in re.finditer(
            f"\\b{re.escape(pathology)}\\b", text, flags=re.IGNORECASE
        ):
            start = max(0, match.start() - 40)
            preceding = text[start : match.start()]
            negated = any((negation in preceding for negation in NEGATION_WORDS))
            if not negated:
                found.append(pathology)
                break
    return sorted(set(found))


def classify_isolated_final_diagnosis(text):
    """
    Convert an already-isolated final diagnosis into:

        appendicitis
        cholecystitis
        diverticulitis
        pancreatitis
        other
        None        -> genuinely ambiguous

    """
    if text is None:
        return None
    text = normalize_text(text)
    if not text:
        return None
    labels = target_mentions(text)
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 0:
        return "other"
    return None


DIAGNOSIS_KEYS = {
    "final diagnosis",
    "final_diagnosis",
    "finaldiagnosis",
    "diagnosis",
    "predicted diagnosis",
    "predicted_diagnosis",
    "prediction",
    "disease",
    "pathology",
}


WRAPPER_KEYS = {"action_input", "input", "arguments", "kwargs"}


def extract_diagnosis_from_structured_value(value):
    """
    Extract diagnosis content ONLY from diagnosis-specific keys/wrappers.

    Critically, this does NOT recursively search every value in the structure.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return extract_diagnosis_from_structured_value(value[0])
        return None
    if not isinstance(value, dict):
        return None
    for key, nested_value in value.items():
        key_normalized = str(key).strip().lower()
        if key_normalized in DIAGNOSIS_KEYS:
            extracted = extract_diagnosis_from_structured_value(nested_value)
            if extracted:
                return extracted
    for key, nested_value in value.items():
        key_normalized = str(key).strip().lower()
        if key_normalized in WRAPPER_KEYS:
            extracted = extract_diagnosis_from_structured_value(nested_value)
            if extracted:
                return extracted
    return None


def extract_final_diagnosis_section(output_value):
    """
    Extract ONLY the text after the LAST occurrence of 'Final Diagnosis:'.

    Stops when another obvious section starts.

    This fixes the original bug where the whole model reasoning was searched
    for words like "appendicitis".
    """
    text = normalize_text(output_value)
    if not text:
        return None
    final_markers = list(
        re.finditer(
            r"\bfinal\s*diagnosis\s*:",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not final_markers:
        return None
    start = final_markers[-1].end()
    section = text[start:]
    stop_match = re.search(
        r"(?:\n|(?<=\.)\s+)"
        r"(?:treatment|management|plan|recommendations?|disposition|follow[- ]?up|next steps?)"
        r"\s*:",
        section,
        flags=re.IGNORECASE,
    )
    if stop_match:
        section = section[: stop_match.start()]
    section = section.strip(" \n\t:-")
    if not section:
        return None
    return section


def extract_hard_diagnosis_from_result(result_record):
    """
    Extract a single hard diagnosis from the actual final-diagnosis content.

    Returns
    -------
    predicted_label
        One of four target classes, "other", or None.

    source
        Where the final diagnosis came from.

    final_diagnosis_text
        The isolated final diagnosis text.

    raw_payload
        Compact raw source for audit.
    """
    if not isinstance(result_record, dict):
        return (None, "missing result record", "", "")
    final_diagnosis_actions = []
    diagnosis_actions = []
    intermediate_steps = result_record.get("intermediate_steps", []) or []
    for step in intermediate_steps:
        try:
            action = step[0]
            tool = str(getattr(action, "tool", "") or "").strip()
            if tool not in {"Final Diagnosis", "Diagnosis"}:
                continue
            tool_input = getattr(action, "tool_input", None)
            if tool == "Final Diagnosis":
                final_diagnosis_actions.append(tool_input)
            else:
                diagnosis_actions.append(tool_input)
        except Exception:
            continue
    if final_diagnosis_actions:
        value = final_diagnosis_actions[-1]
        diagnosis_text = extract_diagnosis_from_structured_value(value)
        raw = compact_text(value)
        if diagnosis_text:
            predicted = classify_isolated_final_diagnosis(diagnosis_text)
            return (
                predicted,
                "Final Diagnosis tool"
                if predicted is not None
                else "Final Diagnosis tool (ambiguous)",
                diagnosis_text,
                raw,
            )
    if diagnosis_actions:
        value = diagnosis_actions[-1]
        diagnosis_text = extract_diagnosis_from_structured_value(value)
        raw = compact_text(value)
        if diagnosis_text:
            predicted = classify_isolated_final_diagnosis(diagnosis_text)
            return (
                predicted,
                "Diagnosis tool"
                if predicted is not None
                else "Diagnosis tool (ambiguous)",
                diagnosis_text,
                raw,
            )
    structured_keys = [
        "final_diagnosis",
        "Final Diagnosis",
        "predicted_diagnosis",
        "diagnosis",
        "Diagnosis",
        "prediction",
    ]
    for key in structured_keys:
        if key not in result_record:
            continue
        value = result_record.get(key)
        diagnosis_text = extract_diagnosis_from_structured_value(value)
        if not diagnosis_text and isinstance(value, str):
            diagnosis_text = normalize_text(value)
        if not diagnosis_text:
            continue
        predicted = classify_isolated_final_diagnosis(diagnosis_text)
        return (
            predicted,
            f"result[{key!r}]"
            if predicted is not None
            else f"result[{key!r}] (ambiguous)",
            diagnosis_text,
            compact_text(value),
        )
    if "output" in result_record:
        output_value = result_record.get("output")
        diagnosis_text = extract_final_diagnosis_section(output_value)
        raw = compact_text(output_value)
        if diagnosis_text is None:
            return (None, "result['output'] (no Final Diagnosis field)", "", raw)
        predicted = classify_isolated_final_diagnosis(diagnosis_text)
        return (
            predicted,
            "result['output'] Final Diagnosis"
            if predicted is not None
            else "result['output'] Final Diagnosis (ambiguous)",
            diagnosis_text,
            raw,
        )
    return (None, "no final diagnosis found", "", "")


def build_prediction_dataframe(model_name, selected_results, selected_evals):
    """
    Build one row per selected patient.

    The evaluator score is used as a consistency constraint, not as a
    substitute for an incorrect text-extraction result.

    Rules
    -----
    Evaluator says CORRECT:
        final canonical hard label must be true_label.
        If text extraction failed or produced a synonym/non-target result,
        canonicalize to true_label.

    Evaluator says INCORRECT:
        an extracted different target class or "other" is accepted.

        If extraction says true_label anyway, this is a conflict and is
        deliberately marked UNRESOLVED rather than incorrectly creating a TP.
    """
    (base_model, seed) = split_seeded_model_name(model_name)
    rows = []
    for true_label in PATHOLOGIES:
        eval_records = selected_evals.get(true_label, {})
        result_records = selected_results.get(true_label, {})
        for patient_id, patient_eval in eval_records.items():
            result_record = result_records.get(patient_id, {})
            (
                raw_extracted_label,
                extraction_source,
                final_diagnosis_text,
                raw_payload,
            ) = extract_hard_diagnosis_from_result(result_record)
            raw_diag_score = get_diagnosis_score(patient_eval)
            evaluator_correct = normalize_binary_diag_score(raw_diag_score)
            predicted_label = raw_extracted_label
            extraction_conflict = False
            canonicalization_reason = ""
            if evaluator_correct == 1:
                if predicted_label != true_label:
                    canonicalization_reason = "Evaluator marks diagnosis correct; canonicalized hard label to ground-truth class"
                    predicted_label = true_label
            elif predicted_label == true_label:
                extraction_conflict = True
                canonicalization_reason = "CONFLICT: evaluator marks diagnosis incorrect but extracted hard label equals true label"
                predicted_label = None
            prediction_resolved = predicted_label is not None
            if prediction_resolved:
                prediction_matches_true = predicted_label == true_label
                evaluator_extraction_agreement = bool(evaluator_correct) == bool(
                    prediction_matches_true
                )
            else:
                prediction_matches_true = np.nan
                evaluator_extraction_agreement = np.nan
            rows.append(
                {
                    "Model": model_name,
                    "Raw Model": model_name,
                    "Base Model": base_model,
                    "Seed": seed,
                    "Patient ID": patient_id,
                    "TrueLabel": true_label,
                    "PredictedLabel": predicted_label
                    if prediction_resolved
                    else "UNRESOLVED",
                    "Raw Extracted Label": raw_extracted_label
                    if raw_extracted_label is not None
                    else "UNRESOLVED",
                    "Final Diagnosis Text": final_diagnosis_text,
                    "Prediction Resolved": bool(prediction_resolved),
                    "Prediction Source": extraction_source,
                    "Raw Prediction Payload": raw_payload,
                    "Evaluator Diagnosis Score": raw_diag_score,
                    "Evaluator Correct": bool(evaluator_correct),
                    "Prediction Matches True Label": prediction_matches_true,
                    "Evaluator/Extraction Agreement": evaluator_extraction_agreement,
                    "Extraction Conflict": bool(extraction_conflict),
                    "Canonicalization Reason": canonicalization_reason,
                }
            )
    return pd.DataFrame(rows)


def compute_diag_accuracy_rows(model_name, selected_evals):
    """
    Compute exactly the same per-class Diagnostic Accuracy concept used by
    your original DiagAcc CSV:

        mean(scores["Gracious Diagnosis"])

    Also emits Mean as the unweighted mean over the four pathologies.
    """
    (base_model, seed) = split_seeded_model_name(model_name)
    rows = []
    pathology_values = []
    for pathology in PATHOLOGIES:
        scores = []
        for patient_eval in selected_evals.get(pathology, {}).values():
            raw_score = get_diagnosis_score(patient_eval)
            binary_score = normalize_binary_diag_score(raw_score)
            scores.append(binary_score)
        accuracy = float(np.mean(scores)) if scores else np.nan
        pathology_values.append(accuracy)
        rows.append(
            {
                "Model": model_name,
                "Raw Model": model_name,
                "Base Model": base_model,
                "Seed": seed,
                "Pathology": pathology.capitalize(),
                "N": len(scores),
                "Diagnostic Accuracy": accuracy * 100.0
                if pd.notna(accuracy)
                else np.nan,
            }
        )
    mean_accuracy = (
        float(np.mean(pathology_values))
        if all((pd.notna(x) for x in pathology_values))
        else np.nan
    )
    rows.append(
        {
            "Model": model_name,
            "Raw Model": model_name,
            "Base Model": base_model,
            "Seed": seed,
            "Pathology": "Mean",
            "N": int(
                sum(
                    (
                        len(selected_evals.get(pathology, {}))
                        for pathology in PATHOLOGIES
                    )
                )
            ),
            "Diagnostic Accuracy": mean_accuracy * 100.0
            if pd.notna(mean_accuracy)
            else np.nan,
        }
    )
    return pd.DataFrame(rows)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_diagnostic_metrics(prediction_df, diag_accuracy_df):
    """
    Compute:

        Sensitivity
        Specificity
        F1
        PPV
        NPV

    from the canonical hard predictions.

    Publication safety rule:
        If ANY patient remains unresolved, metrics that depend on knowing the
        predicted negative/positive class cannot be considered exact.

    We therefore do NOT calculate a misleading subset result.
    """
    metric_columns = [
        "Model",
        "Raw Model",
        "Base Model",
        "Seed",
        "Pathology",
        "N Patients",
        "N Total Cohort",
        "N Resolved Predictions",
        "N Unresolved Predictions",
        "Prediction Coverage",
        "Complete Prediction Data",
        "TP",
        "FP",
        "TN",
        "FN",
        "Sensitivity",
        "Specificity",
        "F1",
        "PPV",
        "NPV",
        "Diagnostic Accuracy",
        "Sensitivity Minus Diagnostic Accuracy",
    ]
    if prediction_df.empty:
        return (pd.DataFrame(columns=metric_columns), pd.DataFrame())
    model = prediction_df["Model"].iloc[0]
    raw_model = prediction_df["Raw Model"].iloc[0]
    base_model = prediction_df["Base Model"].iloc[0]
    seed = prediction_df["Seed"].iloc[0]
    n_total = len(prediction_df)
    resolved_mask = prediction_df["Prediction Resolved"].astype(bool)
    n_resolved = int(resolved_mask.sum())
    n_unresolved = int(n_total - n_resolved)
    complete = n_unresolved == 0
    coverage = n_resolved / n_total if n_total > 0 else np.nan
    resolved_rows = prediction_df.loc[resolved_mask]
    disagreement_rows = resolved_rows[
        resolved_rows["Evaluator/Extraction Agreement"].eq(False)
    ]
    if not disagreement_rows.empty:
        print(
            "\n[ERROR] Resolved hard predictions disagree with evaluator correctness:"
        )
        print(
            disagreement_rows[
                [
                    "Raw Model",
                    "Seed",
                    "Patient ID",
                    "TrueLabel",
                    "PredictedLabel",
                    "Final Diagnosis Text",
                    "Evaluator Correct",
                    "Prediction Matches True Label",
                    "Prediction Source",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )
        raise RuntimeError(
            f"{raw_model}: resolved diagnosis extraction still disagrees with the evaluator."
        )
    accuracy_lookup = {}
    for _, row in diag_accuracy_df.iterrows():
        pathology = str(row["Pathology"]).lower()
        if pathology == "mean":
            continue
        accuracy_lookup[pathology] = float(row["Diagnostic Accuracy"]) / 100.0
    metric_rows = []
    for target in PATHOLOGIES:
        target_mask = prediction_df["TrueLabel"].eq(target)
        n_target = int(target_mask.sum())
        diagnostic_accuracy = accuracy_lookup[target]
        if complete:
            y_true = prediction_df["TrueLabel"].eq(target)
            y_pred = prediction_df["PredictedLabel"].eq(target)
            tp = int((y_true & y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            tn = int((~y_true & ~y_pred).sum())
            fn = int((y_true & ~y_pred).sum())
            sensitivity = safe_ratio(tp, tp + fn)
            specificity = safe_ratio(tn, tn + fp)
            ppv = safe_ratio(tp, tp + fp)
            npv = safe_ratio(tn, tn + fn)
            f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
            difference = sensitivity - diagnostic_accuracy
            if abs(difference) > VALIDATION_TOLERANCE:
                debug = prediction_df[prediction_df["TrueLabel"].eq(target)][
                    [
                        "Patient ID",
                        "TrueLabel",
                        "PredictedLabel",
                        "Evaluator Diagnosis Score",
                        "Evaluator Correct",
                        "Prediction Matches True Label",
                        "Final Diagnosis Text",
                        "Prediction Source",
                    ]
                ]
                print("\n============================================================")
                print("SENSITIVITY / DIAGNOSTIC ACCURACY VALIDATION FAILED")
                print("============================================================")
                print(f"Model:      {raw_model}")
                print(f"Seed:       {seed}")
                print(f"Pathology:  {target}")
                print(f"Sensitivity:         {sensitivity:.12f}")
                print(f"Diagnostic Accuracy: {diagnostic_accuracy:.12f}")
                print(f"Difference:          {difference:.12f}")
                print()
                print(debug.head(50).to_string(index=False))
                raise RuntimeError(
                    "Sensitivity != per-class Diagnostic Accuracy. Hard diagnosis extraction is still inconsistent."
                )
        else:
            tp = np.nan
            fp = np.nan
            tn = np.nan
            fn = np.nan
            sensitivity = diagnostic_accuracy
            specificity = np.nan
            ppv = np.nan
            npv = np.nan
            f1 = np.nan
            difference = 0.0
        metric_rows.append(
            {
                "Model": model,
                "Raw Model": raw_model,
                "Base Model": base_model,
                "Seed": seed,
                "Pathology": target,
                "N Patients": n_target,
                "N Total Cohort": n_total,
                "N Resolved Predictions": n_resolved,
                "N Unresolved Predictions": n_unresolved,
                "Prediction Coverage": coverage,
                "Complete Prediction Data": bool(complete),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Sensitivity": sensitivity * 100.0 if pd.notna(sensitivity) else np.nan,
                "Specificity": specificity * 100.0 if pd.notna(specificity) else np.nan,
                "F1": f1 * 100.0 if pd.notna(f1) else np.nan,
                "PPV": ppv * 100.0 if pd.notna(ppv) else np.nan,
                "NPV": npv * 100.0 if pd.notna(npv) else np.nan,
                "Diagnostic Accuracy": diagnostic_accuracy * 100.0,
                "Sensitivity Minus Diagnostic Accuracy": difference * 100.0,
            }
        )
    metrics_df = pd.DataFrame(metric_rows, columns=metric_columns)
    macro = {
        "Model": model,
        "Raw Model": raw_model,
        "Base Model": base_model,
        "Seed": seed,
        "N Patients": n_total,
        "N Resolved Predictions": n_resolved,
        "N Unresolved Predictions": n_unresolved,
        "Prediction Coverage": coverage,
        "Complete Prediction Data": bool(complete),
    }
    for metric in ["Sensitivity", "Specificity", "F1", "PPV", "NPV"]:
        if metrics_df[metric].notna().all():
            macro[metric] = float(metrics_df[metric].mean())
        else:
            macro[metric] = np.nan
    macro["Diagnostic Accuracy"] = float(metrics_df["Diagnostic Accuracy"].mean())
    macro_df = pd.DataFrame([macro])
    return (metrics_df, macro_df)


def append_dataframe(df, filepath):
    if df is None or df.empty:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, mode="a", header=not os.path.exists(filepath), index=False)


def export_diagnostic_csvs(
    model, selected_results, selected_evals, output_dir, dt_string
):
    """Export evaluator accuracy and audited hard predictions for the same cohort."""
    accuracy = compute_diag_accuracy_rows(model, selected_evals)
    append_dataframe(
        accuracy, os.path.join(output_dir, f"DiagAccCDM_Fig3_{dt_string}.csv")
    )
    predictions = build_prediction_dataframe(model, selected_results, selected_evals)
    if predictions.empty:
        return
    append_dataframe(
        predictions, os.path.join(output_dir, f"DiagnosisPredictions_{dt_string}.csv")
    )
    audit = predictions[
        ~predictions["Prediction Resolved"]
        | predictions["Extraction Conflict"]
        | predictions["Canonicalization Reason"].astype(str).str.len().gt(0)
    ]
    # Keep a header-only audit file when no predictions require review.
    audit_path = os.path.join(output_dir, f"DiagnosisExtractionAudit_{dt_string}.csv")
    if not os.path.exists(audit_path):
        predictions.iloc[:0].to_csv(audit_path, index=False)
    append_dataframe(audit, audit_path)
    metrics, macro = compute_diagnostic_metrics(predictions, accuracy)
    append_dataframe(
        metrics,
        os.path.join(output_dir, f"DiagnosisMetrics_perPathology_{dt_string}.csv"),
    )
    append_dataframe(
        macro,
        os.path.join(output_dir, f"DiagnosisMetrics_macro_perModel_{dt_string}.csv"),
    )
    unresolved = int((~predictions["Prediction Resolved"]).sum())
    conflicts = int(predictions["Extraction Conflict"].sum())
    print(
        f"[diagnostic] {model}: {len(predictions)} patients, {unresolved} unresolved, {conflicts} conflicts"
    )
    if unresolved:
        print(
            "  Inspect DiagnosisExtractionAudit; metrics requiring complete predictions are NaN."
        )


def export_experiment(experiment, models, output_dir, dt_string, id_difficulty):
    """Load each model once and write every CSV for a single experiment."""
    pathologies = PATHOLOGIES
    os.makedirs(output_dir, exist_ok=True)

    def _append_rows_to_csv(rows, columns, filepath):
        df = pd.DataFrame(rows, columns=columns)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, mode="a", header=not os.path.exists(filepath), index=False)

    def _model_pickle_exists(experiment_name, model_name, suffix, f1=False):
        return os.path.exists(_pickle_path(experiment_name, model_name, suffix, f1=f1))

    def _load_raw_model_payload(experiment_name, model_name, suffix, f1=False):
        path = _pickle_path(experiment_name, model_name, suffix, f1=f1)
        if not os.path.exists(path):
            if suffix != "retrievals" and not SKIP_MISSING_MODEL_FILES:
                raise FileNotFoundError(path)
            return _empty_pathology_dict(pathologies)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_selected_model_payloads(experiment_name, model_name):
        raw_evals = _load_raw_model_payload(experiment_name, model_name, "evals")
        raw_results = _load_raw_model_payload(experiment_name, model_name, "results")
        raw_retrievals = _load_raw_model_payload(
            experiment_name, model_name, "retrievals"
        )
        selected_evals = (
            _select_by_difficulty(
                raw_evals,
                id_difficulty,
                DIFFICULTY,
                pathologies,
                experiment=experiment_name,
                model=model_name,
            )
            if raw_evals
            else _empty_pathology_dict(pathologies)
        )
        selected_results = (
            _select_by_difficulty(
                raw_results,
                id_difficulty,
                DIFFICULTY,
                pathologies,
                reference_raw=raw_evals,
                experiment=experiment_name,
                model=model_name,
            )
            if raw_results
            else _empty_pathology_dict(pathologies)
        )
        return (selected_results, selected_evals, raw_retrievals)

    def _avg_score(model_evals_one, field, patho):
        vals = []
        for ev in model_evals_one.get(patho, {}).values():
            scores = ev.get("scores", {}) if isinstance(ev, dict) else {}
            if field in scores:
                vals.append(scores[field])
        return float(np.mean(vals)) if vals else np.nan

    def _score_rows_for_model(model_name, model_evals_one, field):
        rows = []
        vals = []
        for patho in pathologies:
            value = _avg_score(model_evals_one, field, patho)
            vals.append(value)
            rows.append([model_name, patho.capitalize(), value])
        mean_value = float(np.nanmean(vals)) if any(pd.notna(vals)) else np.nan
        rows.append([model_name, "Mean", mean_value])
        return rows

    def _treatment_summary_row_for_model(model_name, model_evals_one):
        """Compute micro-averaged treatment metrics across all pathologies.

        Each required patient-treatment pair is one treatment opportunity.

        Conditional treatment recall:
            requested required treatments among correctly diagnosed opportunities
            / required-treatment opportunities among correctly diagnosed patients.

        End-to-end treatment success:
            requested required treatments with a correct diagnosis
            / all required-treatment opportunities.
        """
        required_opportunities = 0
        correctly_diagnosed_opportunities = 0
        successful_treatment_requests = 0
        for pathology in pathologies:
            for _patient_id, patient_eval in model_evals_one.get(pathology, {}).items():
                if not isinstance(patient_eval, dict):
                    continue
                answers = patient_eval.get("answers")
                if not isinstance(answers, dict):
                    continue
                treatment_required = answers.get("Treatment Required", {})
                treatment_requested = answers.get("Treatment Requested", {})
                if not isinstance(treatment_required, dict):
                    continue
                if not isinstance(treatment_requested, dict):
                    treatment_requested = {}
                scores = patient_eval.get("scores", {})
                correctly_diagnosed = (
                    bool(scores.get(DIAG, False)) if isinstance(scores, dict) else False
                )
                for treatment, required in treatment_required.items():
                    if not bool(required):
                        continue
                    required_opportunities += 1
                    if correctly_diagnosed:
                        correctly_diagnosed_opportunities += 1
                        if bool(treatment_requested.get(treatment, False)):
                            successful_treatment_requests += 1
        conditional_recall = (
            100.0 * successful_treatment_requests / correctly_diagnosed_opportunities
            if correctly_diagnosed_opportunities > 0
            else np.nan
        )
        end_to_end_success = (
            100.0 * successful_treatment_requests / required_opportunities
            if required_opportunities > 0
            else np.nan
        )
        return [
            model_name,
            required_opportunities,
            correctly_diagnosed_opportunities,
            successful_treatment_requests,
            conditional_recall,
            end_to_end_success,
        ]

    def _treatment_detail_rows_for_model(model_name, model_evals_one):
        """Return detailed treatment metrics by pathology and treatment.

        Each row corresponds to one Model x Pathology x Treatment combination.
        The conditional metric matches the main treatment plot: among required
        treatment opportunities for correctly diagnosed patients, what fraction
        of those treatments were requested?

        The end-to-end metric uses every required treatment opportunity as its
        denominator, but its numerator still requires BOTH a correct diagnosis and
        a requested required treatment.

        The all-diagnoses treatment-request metric is different: it ignores the
        diagnosis score entirely and measures requested required treatments among
        ALL patients for whom that treatment was required.
        """
        treatment_stats = {}
        for pathology in pathologies:
            for _patient_id, patient_eval in model_evals_one.get(pathology, {}).items():
                if not isinstance(patient_eval, dict):
                    continue
                answers = patient_eval.get("answers")
                if not isinstance(answers, dict):
                    continue
                treatment_required = answers.get("Treatment Required", {})
                treatment_requested = answers.get("Treatment Requested", {})
                if not isinstance(treatment_required, dict):
                    continue
                if not isinstance(treatment_requested, dict):
                    treatment_requested = {}
                scores = patient_eval.get("scores", {})
                correctly_diagnosed = (
                    bool(scores.get(DIAG, False)) if isinstance(scores, dict) else False
                )
                for treatment, required in treatment_required.items():
                    if not bool(required):
                        continue
                    key = (pathology, treatment)
                    stats = treatment_stats.setdefault(
                        key,
                        {
                            "required_opportunities": 0,
                            "correctly_diagnosed_opportunities": 0,
                            "successful_treatment_requests": 0,
                            "requested_required_all_diagnoses": 0,
                        },
                    )
                    stats["required_opportunities"] += 1
                    requested = bool(treatment_requested.get(treatment, False))
                    if requested:
                        stats["requested_required_all_diagnoses"] += 1
                    if correctly_diagnosed:
                        stats["correctly_diagnosed_opportunities"] += 1
                        if requested:
                            stats["successful_treatment_requests"] += 1
        rows = []
        for (pathology, treatment), stats in sorted(
            treatment_stats.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
        ):
            required_opportunities = stats["required_opportunities"]
            correctly_diagnosed_opportunities = stats[
                "correctly_diagnosed_opportunities"
            ]
            successful_treatment_requests = stats["successful_treatment_requests"]
            requested_required_all_diagnoses = stats["requested_required_all_diagnoses"]
            conditional_recall = (
                100.0
                * successful_treatment_requests
                / correctly_diagnosed_opportunities
                if correctly_diagnosed_opportunities > 0
                else np.nan
            )
            end_to_end_success = (
                100.0 * successful_treatment_requests / required_opportunities
                if required_opportunities > 0
                else np.nan
            )
            all_diagnoses_request_rate = (
                100.0 * requested_required_all_diagnoses / required_opportunities
                if required_opportunities > 0
                else np.nan
            )
            rows.append(
                [
                    model_name,
                    pathology,
                    treatment,
                    successful_treatment_requests,
                    correctly_diagnosed_opportunities,
                    conditional_recall,
                    required_opportunities,
                    end_to_end_success,
                    requested_required_all_diagnoses,
                    all_diagnoses_request_rate,
                ]
            )
        return rows

    def _tool_hallucination_summary_row_for_model(
        model_name, model_results_one, model_evals_one
    ):
        """Aggregate evaluator-counted invalid tools per 100 tool calls.

        Numerator:
            Sum of patient-level scores["Invalid Tools"] across all selected
            pathologies and patients.

        Denominator:
            Total number of recorded intermediate_steps across the corresponding
            selected model results. Each intermediate step is treated as one tool
            call, including invalid/hallucinated calls.
        """
        total_hallucinations = 0.0
        total_tool_calls = 0
        for pathology in pathologies:
            for patient_eval in model_evals_one.get(pathology, {}).values():
                if not isinstance(patient_eval, dict):
                    continue
                scores = patient_eval.get("scores", {})
                if not isinstance(scores, dict):
                    continue
                try:
                    invalid_tools = float(scores.get("Invalid Tools", 0) or 0)
                except (TypeError, ValueError):
                    invalid_tools = 0.0
                if np.isfinite(invalid_tools):
                    total_hallucinations += invalid_tools
            for result in model_results_one.get(pathology, {}).values():
                if not isinstance(result, dict):
                    continue
                steps = result.get("intermediate_steps", [])
                if steps is None:
                    continue
                try:
                    total_tool_calls += len(steps)
                except TypeError:
                    continue
        hallucinations_per_100_calls = (
            100.0 * total_hallucinations / total_tool_calls
            if total_tool_calls > 0
            else np.nan
        )
        raw_hallucination_count = (
            int(total_hallucinations)
            if float(total_hallucinations).is_integer()
            else total_hallucinations
        )
        return [
            model_name,
            raw_hallucination_count,
            total_tool_calls,
            hallucinations_per_100_calls,
        ]

    modality_map = {
        "CT": "CT",
        "Ultrasound": "Ultrasound",
        "Radiograph": "Radiograph",
        "MRI": "MRI",
        "CTU": "CT",
        "ERCP": "Radiograph",
        "Upper GI Series": "Radiograph",
        "MRE": "MRI",
        "MRCP": "MRI",
        "MRA": "MRI",
        "EUS": "Ultrasound",
        "Carotid ultrasound": "Ultrasound",
        "None": "None",
        "PTC": "Other",
        "HIDA": "Other",
        "Drainage": "Other",
    }

    def _imaging_category(modality):
        m = modality_map.get(modality, modality)
        m = str(m).strip()
        if m in {"CT", "Ultrasound", "MRI", "Radiograph"}:
            return m
        if m == "None":
            return "None"
        return "Other"

    def _extract_agent_imaging_orders(model_results_one):
        out = {p: {} for p in pathologies}
        for patho in pathologies:
            for patient_id, rec in model_results_one.get(patho, {}).items():
                mods = []
                for step in rec.get("intermediate_steps", []):
                    try:
                        action = step[0]
                        if getattr(action, "tool", None) != "Imaging":
                            continue
                        tool_input = getattr(action, "tool_input", {}) or {}
                        ai = tool_input.get("action_input", {}) or {}
                        if (
                            ai.get("region") == "Abdomen"
                            and ai.get("modality") is not None
                        ):
                            mods.append(_imaging_category(ai.get("modality")))
                    except Exception:
                        continue
                out[patho][patient_id] = mods
        return out

    def _extract_agent_lab_orders(model_evals_one):
        out = {p: {} for p in pathologies}
        for patho in pathologies:
            for patient_id, ev in model_evals_one.get(patho, {}).items():
                try:
                    out[patho][patient_id] = ev["answers"]["Correct Laboratory Tests"]
                except Exception:
                    out[patho][patient_id] = {}
        return out

    def _extract_agent_physical_exam_orders(model_results_one):
        out = {p: {} for p in pathologies}
        for patho in pathologies:
            for patient_id, rec in model_results_one.get(patho, {}).items():
                did_pe = False
                for step in rec.get("intermediate_steps", []):
                    try:
                        if getattr(step[0], "tool", None) == "Physical Examination":
                            did_pe = True
                            break
                    except Exception:
                        continue
                out[patho][patient_id] = ["Physical Examination"] if did_pe else []
        return out

    def _is_nonempty_local(v):
        if v is None:
            return False
        try:
            return len(v) > 0
        except TypeError:
            return bool(v)

    def _to_count_vector_local(
        items,
        vocab,
        include_other=False,
        other_token="Other",
        dict_value_mode="presence",
    ):
        counts = Counter()
        idx = {c: i for (i, c) in enumerate(vocab)}
        if isinstance(items, dict):
            for k, v in items.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    inc = float(v)
                elif dict_value_mode == "presence":
                    inc = 1.0 if _is_nonempty_local(v) else 0.0
                elif dict_value_mode == "len":
                    inc = float(len(v)) if hasattr(v, "__len__") else 0.0
                else:
                    inc = 1.0 if _is_nonempty_local(v) else 0.0
                if k in idx:
                    counts[k] += inc
                elif include_other and other_token in idx:
                    counts[other_token] += inc
        else:
            for item in items or []:
                if item in idx:
                    counts[item] += 1.0
                elif include_other and other_token in idx:
                    counts[other_token] += 1.0
        vec = np.zeros(len(vocab), dtype=float)
        for k, v in counts.items():
            if k in idx:
                vec[idx[k]] += float(v)
        return vec

    def _bc_similarity(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        denom = float(np.sum(x + y))
        if denom == 0.0:
            return 1.0
        return float(1.0 - np.sum(np.abs(x - y)) / denom)

    def _volume_error(agent_vec, clinician_vec):
        y_total = float(np.sum(clinician_vec))
        if y_total == 0.0:
            return 0.0 if float(np.sum(agent_vec)) == 0.0 else np.nan
        return float(abs(np.sum(agent_vec) - y_total) / y_total)

    LAB_ALIGN_VOCAB_BY_PATHO = {
        "appendicitis": ["Inflammation"],
        "diverticulitis": ["Inflammation"],
        "cholecystitis": ["Inflammation", "Liver", "Gallbladder"],
        "pancreatitis": ["Inflammation", "Pancreas", "Severity"],
    }

    IMAGING_ALIGN_VOCAB = ["CT", "Ultrasound", "MRI", "Radiograph", "Other"]

    PHYS_ALIGN_VOCAB = ["Physical Examination"]

    def _vectorize_imaging(_patho, items):
        return _to_count_vector_local(items, IMAGING_ALIGN_VOCAB, include_other=False)

    def _vectorize_lab(patho, items):
        return _to_count_vector_local(
            items,
            LAB_ALIGN_VOCAB_BY_PATHO[patho],
            include_other=False,
            dict_value_mode="presence",
        )

    def _vectorize_phys(_patho, items):
        return _to_count_vector_local(items, PHYS_ALIGN_VOCAB, include_other=False)

    def _compute_alignment_one_model(clinician_orders, agent_orders, vectorize_fn):
        res = {}
        vol = {}
        for patho in pathologies:
            c_dict = clinician_orders.get(patho, {})
            a_dict = agent_orders.get(patho, {})
            shared_ids = sorted(set(c_dict.keys()) & set(a_dict.keys()))
            if not shared_ids:
                continue
            sims = []
            vols = []
            for patient_id in shared_ids:
                c_vec = np.asarray(vectorize_fn(patho, c_dict[patient_id]), dtype=float)
                a_vec = np.asarray(vectorize_fn(patho, a_dict[patient_id]), dtype=float)
                sims.append(_bc_similarity(a_vec, c_vec))
                vols.append(_volume_error(a_vec, c_vec))
            res[patho] = float(np.mean(sims)) if sims else np.nan
            vol[patho] = float(np.nanmean(vols)) if vols else np.nan
        vals = [res[p] for p in pathologies if p in res and (not np.isnan(res[p]))]
        vol_vals = [vol[p] for p in pathologies if p in vol and (not np.isnan(vol[p]))]
        res["mean"] = float(np.mean(vals)) if vals else np.nan
        vol["mean"] = float(np.mean(vol_vals)) if vol_vals else np.nan
        return (res, vol)

    def _alignment_long_rows(model_name, results):
        rows = []
        for patho in pathologies:
            if patho in results:
                rows.append(
                    [model_name, patho.capitalize(), float(results[patho]) * 100.0]
                )
        if "mean" in results and (not np.isnan(results["mean"])):
            rows.append([model_name, "Mean", float(results["mean"]) * 100.0])
        return rows

    def _save_alignment_pivot(long_rows, filepath):
        if not long_rows:
            return
        df = pd.DataFrame(long_rows, columns=["Model", "Pathology", "Alignment"])
        pivot = df.pivot(index="Model", columns="Pathology", values="Alignment")
        order = [p.capitalize() for p in pathologies] + ["Mean"]
        pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
        pivot.to_csv(filepath)

    hadm_info_by_patho = {
        pathology: load_hadm_from_file(
            f"{pathology}_hadm_info_{DIFFICULTY}", base_mimic=MIMIC_BASE
        )
        for pathology in pathologies
    }

    clinician_imaging_orders = {p: {} for p in pathologies}

    for patho, hadm_info in hadm_info_by_patho.items():
        for patient_id, info in hadm_info.items():
            mods = []
            for rad in info.get("Radiology", []) or []:
                if rad.get("Region") == "Abdomen" and rad.get("Modality") is not None:
                    mods.append(_imaging_category(rad.get("Modality")))
            clinician_imaging_orders[patho][patient_id] = mods

    PHYSICAL_EXAM_KEYS = (
        "Physical Examination",
        "Physical Exam",
        "Physical examination",
        "Physical exam",
        "PE",
    )

    clinician_phys_orders = {p: {} for p in pathologies}

    for patho, hadm_info in hadm_info_by_patho.items():
        for patient_id, info in hadm_info.items():
            did_pe = False
            for key in PHYSICAL_EXAM_KEYS:
                if key in info and _is_nonempty_local(info[key]):
                    did_pe = True
                    break
            clinician_phys_orders[patho][patient_id] = (
                ["Physical Examination"] if did_pe else []
            )

    clinician_lab_orders = {p: {} for p in pathologies}

    mimic_doctor_lab_audit_rows = []

    required_lab_tests = {}

    evaluators = {p: load_evaluator(p) for p in pathologies}

    for patho in pathologies:
        required_lab_tests[patho] = list(
            getattr(evaluators[patho], "required_lab_tests", [])
        )

    for patho, hadm_info in hadm_info_by_patho.items():
        for patient_id, info in hadm_info.items():
            evaluator = load_evaluator(patho)
            raw_laboratory_tests = list((info.get("Laboratory Tests", {}) or {}).keys())
            action = AgentAction(
                tool="Laboratory Tests",
                tool_input={"action_input": raw_laboratory_tests},
                log="",
                custom_parsings=0,
            )
            evaluator.score_laboratory_tests(action)
            answers = copy.deepcopy(getattr(evaluator, "answers", {}) or {})
            correct_laboratory_tests = copy.deepcopy(
                answers.get("Correct Laboratory Tests", {}) or {}
            )
            clinician_lab_orders[patho][patient_id] = correct_laboratory_tests
            for required_lab_test in required_lab_tests[patho]:
                matched_tests = copy.deepcopy(
                    correct_laboratory_tests.get(required_lab_test, [])
                )
                requested = int(_is_nonempty_local(matched_tests))
                label = (
                    "Severity"
                    if required_lab_test == "Seriousness"
                    else required_lab_test
                )
                mimic_doctor_lab_audit_rows.append(
                    [
                        patient_id,
                        patho.capitalize(),
                        label,
                        requested,
                        repr(matched_tests),
                        repr(raw_laboratory_tests),
                    ]
                )

    IMAGING_MODALITY_ORDER = ["CT", "Ultrasound", "MRI", "Radiograph", "Other", "None"]

    def _model_identity_for_csv(model_name):
        (base_model, seed) = split_seeded_model_name(model_name)
        return (model_name, base_model, seed)

    def _imaging_percentage_rows_for_model(model_name, model_results_one):
        """Count first abdominal imaging request, else None."""
        (display_model, base_model, seed) = _model_identity_for_csv(model_name)
        rows = []
        for patho in pathologies:
            counts = Counter()
            patient_records = model_results_one.get(patho, {})
            denominator = len(patient_records)
            for _patient_id, rec in patient_records.items():
                selected_modality = "None"
                if isinstance(rec, dict):
                    for step in rec.get("intermediate_steps", []) or []:
                        try:
                            action = step[0]
                            if getattr(action, "tool", None) != "Imaging":
                                continue
                            tool_input = getattr(action, "tool_input", {}) or {}
                            action_input = tool_input.get("action_input", {}) or {}
                            if action_input.get("region") == "Abdomen":
                                selected_modality = _imaging_category(
                                    action_input.get("modality")
                                )
                                break
                        except Exception:
                            continue
                counts[selected_modality] += 1
            for modality in IMAGING_MODALITY_ORDER:
                count = int(counts.get(modality, 0))
                percentage = 100.0 * count / denominator if denominator > 0 else np.nan
                rows.append(
                    [
                        display_model,
                        base_model,
                        seed,
                        patho.capitalize(),
                        modality,
                        count,
                        denominator,
                        percentage,
                    ]
                )
        return rows

    def _all_imaging_request_rows_for_model(model_name, model_results_one):
        """Export every abdominal imaging request, preserving patient and request order."""
        (display_model, base_model, seed) = _model_identity_for_csv(model_name)
        rows = []
        for patho in pathologies:
            patient_records = model_results_one.get(patho, {})
            for patient_id, rec in patient_records.items():
                if not isinstance(rec, dict):
                    continue
                abdominal_request_number = 0
                for step_index, step in enumerate(
                    rec.get("intermediate_steps", []) or [], start=1
                ):
                    try:
                        action = step[0]
                        if getattr(action, "tool", None) != "Imaging":
                            continue
                        tool_input = getattr(action, "tool_input", {}) or {}
                        action_input = tool_input.get("action_input", {}) or {}
                        if action_input.get("region") != "Abdomen":
                            continue
                        abdominal_request_number += 1
                        raw_modality = action_input.get("modality")
                        rows.append(
                            [
                                display_model,
                                base_model,
                                seed,
                                patho.capitalize(),
                                patient_id,
                                abdominal_request_number,
                                step_index,
                                _imaging_category(raw_modality),
                                raw_modality,
                                action_input.get("region"),
                            ]
                        )
                    except Exception:
                        continue
        return rows

    def _all_imaging_request_rows_for_mimic_doctors():
        """Export every abdominal radiology request recorded for the MIMIC-doctor baseline."""
        rows = []
        for patho, hadm_info in hadm_info_by_patho.items():
            for patient_id, info in hadm_info.items():
                abdominal_request_number = 0
                for rad_index, rad in enumerate(
                    info.get("Radiology", []) or [], start=1
                ):
                    if rad.get("Region") != "Abdomen":
                        continue
                    abdominal_request_number += 1
                    raw_modality = rad.get("Modality")
                    rows.append(
                        [
                            "MIMIC Doctors",
                            "MIMIC Doctors",
                            None,
                            patho.capitalize(),
                            patient_id,
                            abdominal_request_number,
                            rad_index,
                            _imaging_category(raw_modality),
                            raw_modality,
                            rad.get("Region"),
                        ]
                    )
        return rows

    def _imaging_percentage_rows_for_mimic_doctors():
        """Mirror the existing MIMIC-doctor imaging plot logic exactly.

        Note: the original plot only adds a doctor row when an abdominal image
        exists; patients without abdominal imaging are not added as ``None``.
        This helper intentionally preserves that denominator.
        """
        rows = []
        for patho, hadm_info in hadm_info_by_patho.items():
            counts = Counter()
            denominator = 0
            for _patient_id, info in hadm_info.items():
                for rad in info.get("Radiology", []) or []:
                    if rad.get("Region") == "Abdomen":
                        counts[_imaging_category(rad.get("Modality"))] += 1
                        denominator += 1
                        break
            for modality in IMAGING_MODALITY_ORDER:
                count = int(counts.get(modality, 0))
                percentage = 100.0 * count / denominator if denominator > 0 else np.nan
                rows.append(
                    [
                        "MIMIC Doctors",
                        "MIMIC Doctors",
                        None,
                        patho.capitalize(),
                        modality,
                        count,
                        denominator,
                        percentage,
                    ]
                )
        return rows

    def _laboratory_percentage_rows_for_model(model_name, model_evals_one):
        """Mirror LaboratoryTestPercentages: fraction requesting each required category."""
        (display_model, base_model, seed) = _model_identity_for_csv(model_name)
        rows = []
        for patho in pathologies:
            patient_evals = model_evals_one.get(patho, {})
            denominator = len(patient_evals)
            for required_lab_test in required_lab_tests[patho]:
                requested_count = 0
                for patient_eval in patient_evals.values():
                    try:
                        correct_tests = patient_eval["answers"][
                            "Correct Laboratory Tests"
                        ]
                        if _is_nonempty_local(correct_tests.get(required_lab_test, [])):
                            requested_count += 1
                    except Exception:
                        continue
                percentage = (
                    100.0 * requested_count / denominator if denominator > 0 else np.nan
                )
                label = (
                    "Severity"
                    if required_lab_test == "Seriousness"
                    else required_lab_test
                )
                rows.append(
                    [
                        display_model,
                        base_model,
                        seed,
                        patho.capitalize(),
                        label,
                        requested_count,
                        denominator,
                        percentage,
                    ]
                )
        return rows

    def _laboratory_percentage_rows_for_mimic_doctors():
        rows = []
        for patho in pathologies:
            patient_orders = clinician_lab_orders.get(patho, {})
            denominator = len(patient_orders)
            for required_lab_test in required_lab_tests[patho]:
                requested_count = sum(
                    (
                        1
                        for orders in patient_orders.values()
                        if isinstance(orders, dict)
                        and _is_nonempty_local(orders.get(required_lab_test, []))
                    )
                )
                percentage = (
                    100.0 * requested_count / denominator if denominator > 0 else np.nan
                )
                label = (
                    "Severity"
                    if required_lab_test == "Seriousness"
                    else required_lab_test
                )
                rows.append(
                    [
                        "MIMIC Doctors",
                        "MIMIC Doctors",
                        None,
                        patho.capitalize(),
                        label,
                        requested_count,
                        denominator,
                        percentage,
                    ]
                )
        return rows

    def _physical_exam_percentage_rows_for_model(model_name, model_evals_one):
        """Export early/late PE by pathology plus the unweighted pathology mean across pathologies."""
        (display_model, base_model, seed) = _model_identity_for_csv(model_name)
        rows = []
        fields_and_labels = [
            ("Physical Examination", "Physical Examination"),
            ("Late Physical Examination", "(Late) Physical Examination"),
        ]
        for field, category in fields_and_labels:
            pathology_percentages = []
            for patho in pathologies:
                values = []
                for patient_eval in model_evals_one.get(patho, {}).values():
                    if not isinstance(patient_eval, dict):
                        continue
                    scores = patient_eval.get("scores", {})
                    if not isinstance(scores, dict) or field not in scores:
                        continue
                    try:
                        value = float(scores[field])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
                percentage = 100.0 * float(np.mean(values)) if values else np.nan
                pathology_percentages.append(percentage)
                rows.append(
                    [
                        display_model,
                        base_model,
                        seed,
                        patho.capitalize(),
                        category,
                        len(values),
                        percentage,
                    ]
                )
            valid = [x for x in pathology_percentages if pd.notna(x)]
            mean_percentage = float(np.mean(valid)) if valid else np.nan
            rows.append(
                [
                    display_model,
                    base_model,
                    seed,
                    "Mean",
                    category,
                    sum(
                        (
                            1
                            for patho in pathologies
                            for patient_eval in model_evals_one.get(patho, {}).values()
                            if isinstance(patient_eval, dict)
                            and isinstance(patient_eval.get("scores", {}), dict)
                            and (field in patient_eval.get("scores", {}))
                        )
                    ),
                    mean_percentage,
                ]
            )
        return rows

    hallucination_csv = os.path.join(output_dir, f"HallucinationData_{dt_string}.csv")

    tool_hallucination_summary_csv = os.path.join(
        output_dir, f"ToolHallucinationCounts_{dt_string}.csv"
    )

    treatment_summary_csv = os.path.join(
        output_dir, f"TreatmentSummary_{dt_string}.csv"
    )

    treatment_details_csv = os.path.join(
        output_dir, f"TreatmentDetails_{dt_string}.csv"
    )

    imaging_percentages_csv = os.path.join(
        output_dir, f"ImagingPercentages_{dt_string}.csv"
    )

    imaging_all_requests_csv = os.path.join(
        output_dir, f"ImagingAllRequests_{dt_string}.csv"
    )

    laboratory_percentages_csv = os.path.join(
        output_dir, f"LaboratoryTestPercentages_{dt_string}.csv"
    )

    mimic_doctor_lab_evaluations_csv = os.path.join(
        output_dir, f"MIMICDoctorLaboratoryEvaluations_{dt_string}.csv"
    )

    mimic_doctor_lab_percentages_csv = os.path.join(
        output_dir, f"MIMICDoctorLaboratoryTestPercentages_{dt_string}.csv"
    )

    physical_exam_percentages_csv = os.path.join(
        output_dir, f"PhysicalExaminationPercentages_{dt_string}.csv"
    )

    treatment_requested_csv = os.path.join(
        output_dir, f"TreatmentRequested_{dt_string}.csv"
    )

    treatment_requested_correct_diag_csv = os.path.join(
        output_dir, f"TreatmentRequested_CorrectDiagnosis_{dt_string}.csv"
    )

    treatment_requested_all_diag_csv = os.path.join(
        output_dir, f"TreatmentRequested_AllDiagnoses_{dt_string}.csv"
    )

    retrieval_doc_csv = os.path.join(
        output_dir, f"RetrievalDocumentCounts_{dt_string}.csv"
    )

    retrieval_patho_csv = os.path.join(
        output_dir, f"RetrievalMetrics_perPathology_{dt_string}.csv"
    )

    retrieval_summary_csv = os.path.join(
        output_dir, f"RetrievalMetrics_withMean_{dt_string}.csv"
    )

    retrieval_wide_csv = os.path.join(
        output_dir, f"RetrievalMetrics_withMean_wide_{dt_string}.csv"
    )

    brier_csv = os.path.join(output_dir, f"BrierScores_{dt_string}.csv")

    alignment_rows = {
        "ImagingAlignment_PatientMeanBC": [],
        "LabAlignment_PatientMeanBC": [],
        "PhysicalExamAlignment_PatientMeanBC": [],
        "ImagingVolumeError_PatientMean": [],
        "LabVolumeError_PatientMean": [],
        "PhysicalExamVolumeError_PatientMean": [],
        "OverallAlignment_PatientMean": [],
    }

    retr_summary_parts = []

    brier_scores = {}

    RETR_LABELS = [
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
        "other",
    ]

    def _retrieved_label_from_document(document):
        doc = str(document).lower()
        for lab in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
            if lab in doc:
                return lab
        return "other"

    def _retrieval_metrics_for_one_model(model_name, model_retrievals):
        document_rows = []
        patho_rows = []
        for gt_pathology in pathologies:
            pathology_retrievals = (
                model_retrievals.get(gt_pathology, {})
                if isinstance(model_retrievals, dict)
                else {}
            )
            label_counts = {lab: 0 for lab in RETR_LABELS}
            total_chunks = 0
            for document, by_page in pathology_retrievals.items():
                retrieved_label = _retrieved_label_from_document(document)
                doc_chunk_count = 0
                for page in by_page:
                    for order in by_page[page]:
                        doc_chunk_count += by_page[page][order].get("count", 0)
                total_chunks += doc_chunk_count
                label_counts[retrieved_label] += doc_chunk_count
                document_rows.append(
                    {
                        "Model": model_name,
                        "Pathology": gt_pathology.capitalize(),
                        "DocumentReference": document,
                        "RetrievedLabel": retrieved_label.capitalize(),
                        "ChunkCount": int(doc_chunk_count),
                    }
                )
            if total_chunks > 0:
                p_vec = np.array(
                    [label_counts[lab] / total_chunks for lab in RETR_LABELS],
                    dtype=float,
                )
                y_vec = np.array(
                    [1.0 if lab == gt_pathology else 0.0 for lab in RETR_LABELS],
                    dtype=float,
                )
                precision = label_counts[gt_pathology] / total_chunks
                brier = float(np.sum((p_vec - y_vec) ** 2))
            else:
                p_vec = np.array([0.0] * len(RETR_LABELS), dtype=float)
                precision = 0.0
                brier = np.nan
            row = {
                "Model": model_name,
                "Pathology": gt_pathology.capitalize(),
                "TotalChunks": int(total_chunks),
                "Precision@6_equiv": float(precision),
                "Brier": brier,
                "nDCG@6": np.nan,
            }
            for i, lab in enumerate(RETR_LABELS):
                row[f"Count_{lab.capitalize()}"] = int(label_counts[lab])
                row[f"Prop_{lab.capitalize()}"] = float(p_vec[i])
            patho_rows.append(row)
        doc_df = pd.DataFrame(document_rows)
        patho_df = pd.DataFrame(patho_rows)
        mean_rows = []
        if not patho_df.empty:
            sub = patho_df.copy()
            mean_row = {
                "Model": model_name,
                "Pathology": "Mean",
                "TotalChunks": int(sub["TotalChunks"].sum()),
                "Precision@6_equiv": float(sub["Precision@6_equiv"].mean()),
                "Brier": float(sub["Brier"].mean(skipna=True))
                if sub["Brier"].notna().any()
                else np.nan,
                "nDCG@6": np.nan,
            }
            for lab in RETR_LABELS:
                mean_row[f"Count_{lab.capitalize()}"] = int(
                    sub[f"Count_{lab.capitalize()}"].sum()
                )
                mean_row[f"Prop_{lab.capitalize()}"] = (
                    mean_row[f"Count_{lab.capitalize()}"] / mean_row["TotalChunks"]
                    if mean_row["TotalChunks"] > 0
                    else 0.0
                )
            mean_rows.append(mean_row)
        summary_df = pd.concat([patho_df, pd.DataFrame(mean_rows)], ignore_index=True)
        return (doc_df, patho_df, summary_df)

    def _document_pathology(document_name):
        name = str(document_name).lower()
        for lab in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
            if lab in name:
                return lab
        if "general" in name:
            return "general"
        return "z_other"

    def _compute_brier_score_oldstyle(df_heatmap):
        scores = {}
        for i, actual_patho in enumerate(df_heatmap.index):
            brier_score = 0.0
            for j, _predicted_patho in enumerate(df_heatmap.columns):
                prob = df_heatmap.iloc[i, j]
                actual = 1.0 if i == j else 0.0
                weight = 1.0 if i != j else len(df_heatmap.columns) - 1
                brier_score += (1 - (prob - actual) ** 2) * weight
            brier_score /= (
                (len(df_heatmap.columns) - 1) * 2 if len(df_heatmap.columns) > 1 else 1
            )
            scores[actual_patho] = brier_score
        scores["mean"] = float(np.mean(list(scores.values()))) if scores else np.nan
        return scores

    def _write_retrieval_tables_and_brier(model_name, model_retrievals):
        df_heatmap_dict = {}
        model_is_rag = False
        for pathology in pathologies:
            df_heatmap_dict[pathology] = {}
            pathology_retrievals = (
                model_retrievals.get(pathology, {})
                if isinstance(model_retrievals, dict)
                else {}
            )
            total_chunks = 0
            for document in pathology_retrievals:
                for page in pathology_retrievals[document]:
                    for order in pathology_retrievals[document][page]:
                        total_chunks += pathology_retrievals[document][page][order].get(
                            "count", 0
                        )
            if total_chunks > 0:
                model_is_rag = True
            if total_chunks <= 0:
                continue
            for document, document_retrievals in sorted(
                pathology_retrievals.items(), key=lambda kv: str(kv[0])
            ):
                document_name = (
                    document.split("/")[-1]
                    if isinstance(document, str)
                    else str(document)
                )
                if len(document_name) > 20:
                    document_name = document_name[:25] + "..."
                doc_counts = 0
                for page in document_retrievals:
                    for order in document_retrievals[page]:
                        doc_counts += document_retrievals[page][order].get("count", 0)
                df_heatmap_dict[pathology][document_name] = (
                    doc_counts / total_chunks if total_chunks > 0 else 0.0
                )
        if not model_is_rag:
            return None
        df_heatmap_count = pd.DataFrame(df_heatmap_dict).fillna(0.0)
        df_heatmap_count["document_pathology"] = (
            df_heatmap_count.index.to_series().apply(_document_pathology)
        )
        df_heatmap_count = df_heatmap_count.sort_values(by="document_pathology")
        df_heatmap_count_sum = (
            df_heatmap_count.copy()
            .groupby("document_pathology")
            .sum()
            .rename(index={"z_other": "other"})
        )
        df_heatmap_count = df_heatmap_count.drop(columns=["document_pathology"]).T
        df_heatmap_count.to_csv(
            os.path.join(output_dir, f"Retrievals_{model_name}_heatmap_{dt_string}.csv")
        )
        df_heatmap_count_sum = df_heatmap_count_sum.T
        df_heatmap_count_sum.index = [
            p.capitalize() for p in df_heatmap_count_sum.index
        ]
        df_heatmap_count_sum.columns = [f"{p}%" for p in df_heatmap_count_sum.columns]
        df_heatmap_count_sum.to_csv(
            os.path.join(
                output_dir, f"Retrievals_{model_name}_heatmap_sum_{dt_string}.csv"
            )
        )
        return _compute_brier_score_oldstyle(df_heatmap_count_sum)

    for model in tqdm(models, desc=f"CSV export {experiment}", unit="model"):
        missing = [
            _pickle_path(experiment, model, suffix)
            for suffix in ("evals", "results")
            if not _model_pickle_exists(experiment, model, suffix)
        ]
        if missing:
            if not SKIP_MISSING_MODEL_FILES:
                raise FileNotFoundError(missing[0])
            print(f"[CSV export] Skipping {model}; missing inputs: {missing}")
            continue
        print(f"[CSV export] Loading {model}")
        (model_results_one, model_evals_one, model_retrievals_one) = (
            _load_selected_model_payloads(experiment, model)
        )
        export_diagnostic_csvs(
            model, model_results_one, model_evals_one, output_dir, dt_string
        )
        hallucination_rows = _score_rows_for_model(
            model, model_evals_one, "Invalid Tools"
        )
        _append_rows_to_csv(
            hallucination_rows,
            ["Model", "Pathology", "Tool Hallucination"],
            hallucination_csv,
        )
        tool_hallucination_summary_row = _tool_hallucination_summary_row_for_model(
            model, model_results_one, model_evals_one
        )
        _append_rows_to_csv(
            [tool_hallucination_summary_row],
            [
                "Model",
                "Total Tool Hallucinations",
                "Total Tool Calls",
                "Tool Hallucinations per 100 Tool Calls",
            ],
            tool_hallucination_summary_csv,
        )
        treatment_summary_row = _treatment_summary_row_for_model(model, model_evals_one)
        _append_rows_to_csv(
            [treatment_summary_row],
            [
                "Model",
                "Required Treatment Opportunities",
                "Correctly Diagnosed Required Opportunities",
                "Successful Required Treatment Requests",
                "Conditional Treatment Recall (%)",
                "End-to-End Treatment Success (%)",
            ],
            treatment_summary_csv,
        )
        treatment_detail_rows = _treatment_detail_rows_for_model(model, model_evals_one)
        _append_rows_to_csv(
            treatment_detail_rows,
            [
                "Model",
                "Pathology",
                "Treatment",
                "Successful Required Treatment Requests",
                "Correctly Diagnosed Required Opportunities",
                "Conditional Treatment Recall (%)",
                "Required Treatment Opportunities",
                "End-to-End Treatment Success (%)",
                "Requested Required Treatments - All Diagnoses",
                "All-Diagnoses Treatment Request (%)",
            ],
            treatment_details_csv,
        )
        (display_model, base_model, seed) = _model_identity_for_csv(model)
        treatment_correct_diag_rows = []
        treatment_all_diag_rows = []
        for detail_row in treatment_detail_rows:
            pathology_display = str(detail_row[1]).capitalize()
            treatment = detail_row[2]
            successful_correct_diag = detail_row[3]
            correctly_diagnosed_count = detail_row[4]
            conditional_percentage = detail_row[5]
            treatment_cohort_count = detail_row[6]
            requested_all_diagnoses = detail_row[8]
            all_diagnoses_percentage = detail_row[9]
            treatment_correct_diag_rows.append(
                [
                    display_model,
                    base_model,
                    seed,
                    pathology_display,
                    treatment,
                    treatment_cohort_count,
                    correctly_diagnosed_count,
                    successful_correct_diag,
                    conditional_percentage,
                ]
            )
            treatment_all_diag_rows.append(
                [
                    display_model,
                    base_model,
                    seed,
                    pathology_display,
                    treatment,
                    treatment_cohort_count,
                    requested_all_diagnoses,
                    all_diagnoses_percentage,
                ]
            )
        correct_diag_headers = [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Treatment",
            "Treatment Cohort Count",
            "Correctly Diagnosed Count",
            "Requested Count",
            "Request Percentage",
        ]
        all_diag_headers = [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Treatment",
            "Treatment Cohort Count",
            "Requested Count",
            "Request Percentage",
        ]
        _append_rows_to_csv(
            treatment_correct_diag_rows,
            correct_diag_headers,
            treatment_requested_correct_diag_csv,
        )
        treatment_legacy_rows = [
            [row[0], row[1], row[2], row[3], row[4], row[6], row[7], row[8]]
            for row in treatment_correct_diag_rows
        ]
        _append_rows_to_csv(
            treatment_legacy_rows,
            [
                "Model",
                "Base Model",
                "Seed",
                "Pathology",
                "Treatment",
                "Counts",
                "Requested Count",
                "Request Correct",
            ],
            treatment_requested_csv,
        )
        _append_rows_to_csv(
            treatment_all_diag_rows, all_diag_headers, treatment_requested_all_diag_csv
        )
        _append_rows_to_csv(
            _imaging_percentage_rows_for_model(model, model_results_one),
            [
                "Model",
                "Base Model",
                "Seed",
                "Pathology",
                "Modality",
                "Counts",
                "Total Patients",
                "Percentage",
            ],
            imaging_percentages_csv,
        )
        _append_rows_to_csv(
            _all_imaging_request_rows_for_model(model, model_results_one),
            [
                "Model",
                "Base Model",
                "Seed",
                "Pathology",
                "Patient ID",
                "Abdominal Request Number",
                "Step Index",
                "Modality",
                "Raw Modality",
                "Region",
            ],
            imaging_all_requests_csv,
        )
        _append_rows_to_csv(
            _laboratory_percentage_rows_for_model(model, model_evals_one),
            [
                "Model",
                "Base Model",
                "Seed",
                "Pathology",
                "Required Laboratory Test",
                "Requested Count",
                "Total Patients",
                "Percentage",
            ],
            laboratory_percentages_csv,
        )
        _append_rows_to_csv(
            _physical_exam_percentage_rows_for_model(model, model_evals_one),
            ["Model", "Base Model", "Seed", "Pathology", "Category", "N", "Percentage"],
            physical_exam_percentages_csv,
        )
        agent_imaging = _extract_agent_imaging_orders(model_results_one)
        agent_lab = _extract_agent_lab_orders(model_evals_one)
        agent_phys = _extract_agent_physical_exam_orders(model_results_one)
        (img_res, img_vol) = _compute_alignment_one_model(
            clinician_imaging_orders, agent_imaging, _vectorize_imaging
        )
        (lab_res, lab_vol) = _compute_alignment_one_model(
            clinician_lab_orders, agent_lab, _vectorize_lab
        )
        (phys_res, phys_vol) = _compute_alignment_one_model(
            clinician_phys_orders, agent_phys, _vectorize_phys
        )
        alignment_rows["ImagingAlignment_PatientMeanBC"].extend(
            _alignment_long_rows(model, img_res)
        )
        alignment_rows["LabAlignment_PatientMeanBC"].extend(
            _alignment_long_rows(model, lab_res)
        )
        alignment_rows["PhysicalExamAlignment_PatientMeanBC"].extend(
            _alignment_long_rows(model, phys_res)
        )
        alignment_rows["ImagingVolumeError_PatientMean"].extend(
            _alignment_long_rows(model, img_vol)
        )
        alignment_rows["LabVolumeError_PatientMean"].extend(
            _alignment_long_rows(model, lab_vol)
        )
        alignment_rows["PhysicalExamVolumeError_PatientMean"].extend(
            _alignment_long_rows(model, phys_vol)
        )
        overall = {}
        for patho in pathologies:
            scores = [x.get(patho, np.nan) for x in [img_res, lab_res, phys_res]]
            scores = [s for s in scores if not np.isnan(s)]
            overall[patho] = float(np.mean(scores)) if scores else np.nan
        patho_scores = [overall.get(p, np.nan) for p in pathologies]
        patho_scores = [s for s in patho_scores if not np.isnan(s)]
        overall["mean"] = float(np.mean(patho_scores)) if patho_scores else np.nan
        alignment_rows["OverallAlignment_PatientMean"].extend(
            _alignment_long_rows(model, overall)
        )
        (doc_df, patho_df, summary_df) = _retrieval_metrics_for_one_model(
            model, model_retrievals_one
        )
        if not doc_df.empty:
            doc_df.to_csv(
                retrieval_doc_csv,
                mode="a",
                header=not os.path.exists(retrieval_doc_csv),
                index=False,
            )
        if not patho_df.empty:
            patho_df.to_csv(
                retrieval_patho_csv,
                mode="a",
                header=not os.path.exists(retrieval_patho_csv),
                index=False,
            )
        if not summary_df.empty:
            summary_df.to_csv(
                retrieval_summary_csv,
                mode="a",
                header=not os.path.exists(retrieval_summary_csv),
                index=False,
            )
            retr_summary_parts.append(summary_df)
        brier = _write_retrieval_tables_and_brier(model, model_retrievals_one)
        if brier is not None:
            brier_scores[model] = brier
        del model_results_one, model_evals_one, model_retrievals_one
        del agent_imaging, agent_lab, agent_phys
        del img_res, img_vol, lab_res, lab_vol, phys_res, phys_vol
        del doc_df, patho_df, summary_df
        gc.collect()

    _append_rows_to_csv(
        _imaging_percentage_rows_for_mimic_doctors(),
        [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Modality",
            "Counts",
            "Total Patients",
            "Percentage",
        ],
        imaging_percentages_csv,
    )

    _append_rows_to_csv(
        _all_imaging_request_rows_for_mimic_doctors(),
        [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Patient ID",
            "Abdominal Request Number",
            "Step Index",
            "Modality",
            "Raw Modality",
            "Region",
        ],
        imaging_all_requests_csv,
    )

    mimic_doctor_lab_percentage_rows = _laboratory_percentage_rows_for_mimic_doctors()

    _append_rows_to_csv(
        mimic_doctor_lab_percentage_rows,
        [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Required Laboratory Test",
            "Requested Count",
            "Total Patients",
            "Percentage",
        ],
        laboratory_percentages_csv,
    )

    _append_rows_to_csv(
        [
            [row[3], row[4], row[5], row[6], row[7]]
            for row in mimic_doctor_lab_percentage_rows
        ],
        [
            "Pathology",
            "Required Laboratory Test",
            "Requested Count",
            "Total Patients",
            "Percentage",
        ],
        mimic_doctor_lab_percentages_csv,
    )

    _append_rows_to_csv(
        mimic_doctor_lab_audit_rows,
        [
            "Patient ID",
            "Pathology",
            "Required Laboratory Test",
            "Requested",
            "Matched Laboratory Tests",
            "Raw Laboratory Test Names",
        ],
        mimic_doctor_lab_evaluations_csv,
    )

    for name, rows in alignment_rows.items():
        _save_alignment_pivot(rows, os.path.join(output_dir, f"{name}_{dt_string}.csv"))

    if retr_summary_parts:
        retr_summary_df = pd.concat(retr_summary_parts, ignore_index=True)
        retr_summary_wide = retr_summary_df.pivot(
            index="Model",
            columns="Pathology",
            values=["Precision@6_equiv", "Brier", "nDCG@6", "TotalChunks"],
        )
        retr_summary_wide.to_csv(retrieval_wide_csv)

    if brier_scores:
        pd.DataFrame.from_dict(brier_scores, orient="index").to_csv(brier_csv)

    print(f"[CSV export] Finished: {output_dir}")


@contextmanager
def configured_analysis(args: DictConfig | None):
    """Apply one run's configuration to the shared analysis helpers, then restore it."""
    if args is None:
        yield os.path.join(MIMIC_BASE, "id_difficulty.pkl")
        return

    def string_list(name, allow_empty=False):
        values = args[name]
        if not isinstance(values, (list, tuple, ListConfig)):
            raise ValueError(f"{name} must be a list of strings.")
        values = list(values)
        if not values or any(
            not isinstance(value, str) or (not value and not allow_empty)
            for value in values
        ):
            raise ValueError(f"{name} must be a nonempty list of strings.")
        return list(dict.fromkeys(values))

    models = string_list("models")
    experiments = string_list("experiments", allow_empty=True)
    pathologies = string_list("pathologies")
    unknown = set(pathologies) - set(DIAGNOSTIC_CLASSES)
    if unknown:
        raise ValueError(f"Unsupported pathologies: {sorted(unknown)}")
    for experiment in experiments:
        if experiment and (
            os.path.basename(experiment) != experiment or experiment in (".", "..")
        ):
            raise ValueError(
                "Experiment names must be single directory names or empty."
            )
    if not isinstance(args.seeds, (list, tuple, ListConfig)):
        raise ValueError("seeds must be a list of integers.")
    seeds = list(args.seeds)
    if type(args.base_seed) is not int or any(type(seed) is not int for seed in seeds):
        raise ValueError("base_seed and every entry in seeds must be integers.")
    for model in models:
        if "_SEED=" in model:
            _, seed = model.rsplit("_SEED=", 1)
            try:
                int(seed)
            except ValueError:
                raise ValueError(f"Invalid seeded model ID: {model}")
    for name in ("difficulty", "diagnosis_score"):
        if not isinstance(args[name], str) or not args[name].strip():
            raise ValueError(f"{name} must be a nonempty string.")
    for name in ("skip_missing_model_files",):
        if type(args[name]) is not bool:
            raise ValueError(f"{name} must be true or false.")
    tolerance = args.validation_tolerance
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("validation_tolerance must be a finite nonnegative number.")

    def absolute_path(name):
        value = args[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path.")
        return to_absolute_path(os.path.expanduser(value))

    settings = {
        "MIMIC_BASE": absolute_path("base_mimic"),
        "EXPERIMENTS_BASE": absolute_path("analysis_dir"),
        "OUTPUT_BASE": absolute_path("output_dir"),
        "DIAG": args.diagnosis_score,
        "DIFFICULTY": args.difficulty,
        "PATHOLOGIES": pathologies,
        "EXPERIMENTS": experiments,
        "MODELS": models,
        "SEEDS": seeds,
        "BASE_SEED": args.base_seed,
        "SKIP_MISSING_MODEL_FILES": args.skip_missing_model_files,
        "VALIDATION_TOLERANCE": float(tolerance),
    }
    difficulty_path = absolute_path("id_difficulty_path")
    previous = {name: globals()[name] for name in settings}
    globals().update(settings)
    try:
        yield difficulty_path
    finally:
        globals().update(previous)


def export_configured_runs(difficulty_path):
    """Run all selected experiments with one shared model and cohort selection."""
    models = _expand_seeded_models_as_independent(MODELS, SEEDS)
    models = list(dict.fromkeys(models))
    experiments = list(dict.fromkeys(EXPERIMENTS))
    if not models or not experiments:
        raise ValueError(
            "Configure at least one model and experiment, and select at least one "
            "seed for unsuffixed model labels."
        )
    with open(difficulty_path, "rb") as handle:
        id_difficulty = pickle.load(handle)
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = os.path.join(OUTPUT_BASE, dt_string)
    for experiment in experiments:
        output_dir = (
            os.path.join(output_root, experiment or "default")
            if len(experiments) > 1
            else output_root
        )
        export_experiment(experiment, models, output_dir, dt_string, id_difficulty)


def main(args: DictConfig | None = None):
    """Export from a composed Hydra configuration or the direct helper defaults."""
    with configured_analysis(args) as difficulty_path:
        export_configured_runs(difficulty_path)


@hydra.main(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"),
    config_name="inspect",
    version_base=None,
)
def run(args: DictConfig):
    main(args)


if __name__ == "__main__":
    run()

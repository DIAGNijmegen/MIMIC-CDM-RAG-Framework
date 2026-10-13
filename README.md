# MIMIC-CDM RAG Framework

This framework extends [MIMIC-CDM](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework)
with retrieval-augmented generation (RAG) inside the clinical decision-making
loop. An LLM starts with a patient's history, requests physical examinations,
laboratory results, and imaging, then proposes a diagnosis and treatment plan.
Guideline retrieval supplies relevant clinical context as the case progresses.

## Dataset

The framework runs on the MIMIC-CDM dataset, derived from MIMIC-IV:

- [Dataset creation and conversion code](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset).
- [MIMIC-IV v2.2 access](https://physionet.org/content/mimiciv/2.2/).
- [Preprocessed MIMIC-IV-Ext-CDM dataset](https://physionet.org/content/mimic-iv-ext-cdm/).

Follow the dataset repository's instructions to generate or convert the patient
files, then point `base_mimic` to the resulting directory.

## Guidelines and models

Use the guidelines appropriate for your experiment, as PDFs or Markdown files.
The documents used in our experiments are listed in [docs/guidelines.md](docs/guidelines.md)
for comparison and reproducibility. Select your corpus through `rag_documents`.

Models are configurable through profiles in `configs/model/`. Our experiments
used **ExLlamaV2 with EXL2 models quantized to 4.0 bits per weight**. The project
retains other model-loading backends inherited from the original MIMIC-CDM
framework, so other models can be configured through those backends; they were
not evaluated in these experiments.

## Environment

A CUDA-capable GPU is required for the setup used here. Create a new virtual environment of your choosing with python=3.10, and then install the libraries from requirements.txt. Export your `CUDA_HOME` to your installed CUDA toolkit, rather than copying the example path. The pinned GPU packages target CUDA 12.8; a different setup may require compatible package builds.

```bash
export CUDA_HOME=/path/to/cuda/cuda_12.8.0
pip install --no-deps -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Alternatively, Docker was used for the experiments; if you prefer, you can also build the minimal supplied image from the repository root using:

```bash
docker build -t mimic-cdm-rag:py310 .
```

## Configuration and documentation

Most scripts use [Hydra YAML configs](configs/) or `key=value` command-line
arguments. CLI values override YAML. Set your paths and profiles before running:

| Configuration | Purpose |
| --- | --- |
| [configs/paths/example.yaml](configs/paths/example.yaml) | Patient data, local models, guidelines, laboratory mapping, and logging paths. |
| [configs/config.yaml](configs/config.yaml) | Inference defaults and selected `paths`, `model`, and `rag` profiles. |
| [configs/rag/](configs/rag/) | Corpus files, embedding, retrieval, reranking, and requery settings. |
| [configs/evaluate.yaml](configs/evaluate.yaml), [configs/inspect.yaml](configs/inspect.yaml) | Saved-run evaluation and metric export. |
| [configs/figures.yaml](configs/figures.yaml), [configs/figures/](configs/figures/) | Figure model names/IDs, seeds, inputs, and layout. |

For a reusable configuration, copy a driver YAML such as `configs/config.yaml`,
retain its defaults, and select it with `--config-name your_config`. Figure
profiles are selected with `figures=<name>`. Use `--cfg job --resolve` to inspect
settings. Each pipeline step below links to its argument guide.

## Pipeline

Run the commands from the repository root. The examples use the `Gemma2-9B` model and the `Qwen3-Embedding-0.6B` RAG profile; choose the profiles and labels that match
your experiment.

### 1. Convert guidelines to Markdown (optional)

PDFs can be used directly. To convert them with Docling and optionally clean the
Markdown by hand:

```bash
python docling/pdf_to_md_ocr.py /path/to/pdfs /path/to/markdown \
  --ocr --ocr-engine rapidocr --ocr-backend torch --add-title
```

See [PDF conversion options](docs/pdf_to_md_ocr.md). This script uses ordinary
CLI flags rather than Hydra.

### 2. Build corpus reports (optional)

Build a FAISS index and inspect corpus document/chunk word counts and composition reports:

```bash
python build_rag_bd.py rag=Qwen3-Embedding-0.6B
```

These reports were used to describe the experimental corpus. `run_rag.py`
builds its own index for each run and does not load this exported FAISS index.
See [corpus-building options](docs/build_rag_bd.md).

### 3. Run inference

**`run_rag.py` is the main entry point of the framework.** It runs the agent loop
and saves trajectories. This Bash example runs baseline and RAG conditions for
the four supported pathologies:

```bash
for pathology in appendicitis cholecystitis diverticulitis pancreatitis; do
  python run_rag.py model=Gemma2Instruct9B use_rag=false \
    pathology="$pathology" base_seed=2023 seed=2023
  python run_rag.py model=Gemma2Instruct9B rag=Qwen3-Embedding-0.6B use_rag=true \
    pathology="$pathology" base_seed=2023 seed=2023
done
```

Run either command separately for a single pathology. Change `seed` to repeat
runs with another seed; keep `base_seed` consistent across inference, evaluation,
and inspection. See [inference options](docs/run_rag.md).

### 4. Evaluate saved results

Score the saved trajectories. `models` contains exact saved labels, combining
`model_name` and `rag_name` for RAG runs:

```bash
python evaluate_cdm_rag.py \
  'models=[gemma-2-9b-it-exl2-4.0bpw,gemma-2-9b-it-exl2-4.0bpw_Qwen3-Embedding-0.6B_MedCPT_rerank]' \
  base_seed=2023 'seeds=[2023]'
```

`logs_dir` points to the inference run folders. See [evaluation options](docs/evaluation.md).

### 5. Inspect results and export metrics

Extract diagnostic, retrieval, treatment, and request metrics from the aggregate
evaluation files:

```bash
python inspect_results_cdm_rag.py \
  'models=[gemma-2-9b-it-exl2-4.0bpw,gemma-2-9b-it-exl2-4.0bpw_Qwen3-Embedding-0.6B_MedCPT_rerank]' \
  'experiments=[CDM_VANILLA]' base_seed=2023 'seeds=[2023]'
```

`analysis_dir` is the parent of the evaluation experiment folders. The script
exports CSVs into a timestamped output folder. See [inspection options](docs/inspection.md).

### 6. Generate figures and tables (optional)

Use the inspection CSV folder as input. Set the model/RAG IDs in
`configs/figures.yaml` to match your saved runs:

```bash
python make_figures/main_figure.py figures=main_figure \
  results_dir=/path/to/inspection_output output_dir=/path/to/figures \
  'selected_models=[gemma-2-9b-it-exl2-4.0bpw]' 'seeds=[2023]'
```

Other scripts produce ablation/retrieval tables, retrieval-composition figures,
order/treatment figures, and detailed per-model reports. See the
[figure guides](docs/figures/README.md).

## Running the scripts with Docker

The image provides the environment; mount the repository, patient data,
guidelines, and models. For example, in **PowerShell**:

```powershell
docker run --gpus all -it --rm `
  -v "C:\path\to\repository:/container/scripts" `
  -v "D:\path\to\mimic-cdm-dataset:/container/data" `
  -v "D:\path\to\guidelines:/container/guidelines" `
  -v "D:\path\to\models:/container/models" `
  -w /container/scripts `
  mimic-cdm-rag:py310 bash
```

These container paths match `configs/paths/example.yaml`. Inside the container,
run the pipeline commands above; for example:

```bash
python run_rag.py model=Gemma2Instruct9B rag=Qwen3-Embedding-0.6B \
  use_rag=true pathology=appendicitis
```

You can also replace the final `bash` in the Docker command with a Python command
to execute one script directly. Use container paths for all overrides. Inspection
outputs are normally under `/container/data/logs/analysis/output/<timestamp>`;
use that folder as figure `results_dir` and a mounted destination such as
`output_dir=/container/data/figures` to retain exports on the host.

## Acknowledgments

This framework builds upon [MIMIC-Clinical-Decision-Making-Framework](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework), originally developed by Hager, Paul and Jungmann, Friederike and Rueckert, Daniel and released under the MIT License.

Copyright for the original implementation remains with its respective copyright holders. The modifications and additional components developed as part of this research are Copyright (c) 2026, Diagnostic Image Analysis Group, Radboud University Medical Center, Nijmegen, the Netherlands and are released under the MIT License.

The original MIT license and copyright notices are preserved in this repository.

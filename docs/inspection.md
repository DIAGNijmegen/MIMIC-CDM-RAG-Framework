# Export inspection CSVs

Extract analysis values from the evaluation.
`inspect_results_cdm_rag.py` exports analysis CSVs using `configs/inspect.yaml`.
It keeps raw model IDs and produces no images.

```bash
python inspect_results_cdm_rag.py \
  base_mimic=/path/to/patient_data analysis_dir=/path/to/analysis \
  output_dir=/path/to/csv 'models=[my-model_my-rag-name]' \
  'experiments=[CDM_VANILLA]' 'seeds=[2023,123,4896]'
```

| Argument | Purpose |
| --- | --- |
| `base_mimic` | Patient and clinician data directory. |
| `analysis_dir` | Parent of aggregate experiment folders from evaluation. |
| `output_dir` | Destination; CSVs go into a timestamped subfolder. |
| `models` | Exact saved baseline/RAG model labels. |
| `experiments` | Experiment folder names; an empty-string entry selects `analysis_dir` itself. |
| `pathologies`, `difficulty` | Cohort selection. |
| `base_seed`, `seeds` | Unsuffixed run seed and variants to export. Include `base_seed` to select the original run. |
| `diagnosis_score` | Diagnosis correctness metric; defaults to `Gracious Diagnosis`. |
| `skip_missing_model_files` | Skip missing model files when `true`; otherwise fail. |
| `id_difficulty_path`, `validation_tolerance` | Admission selection file and diagnostic consistency tolerance. |

Copy `configs/inspect.yaml` to `configs/inspect_local.yaml`, keep its defaults,
and run `python inspect_results_cdm_rag.py --config-name inspect_local`.
CLI values override YAML; paths are inherited from `configs/paths/example.yaml`.
Use the same `base_seed` as inference and evaluation.

Exports cover diagnosis, hallucinations, treatment, imaging, laboratory requests,
physical examinations, alignment, and retrieval. Use the resulting folder with
the [figure scripts](figures/README.md).

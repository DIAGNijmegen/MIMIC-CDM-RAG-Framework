# Evaluate saved runs

`evaluate_cdm_rag.py` scores saved inference results using `configs/evaluate.yaml`.

```bash
python evaluate_cdm_rag.py \
  base_mimic=/path/to/patient_data logs_dir=/path/to/saved_runs \
  output_dir=/path/to/analysis 'models=[my-model_my-rag-name]' \
  'seeds=[2023,123,4896]'
```

| Argument | Purpose |
| --- | --- |
| `base_mimic` | Patient data, including `id_difficulty.pkl` and pathology admission files. |
| `logs_dir` | Folder containing inference run directories. |
| `output_dir` | Destination for evaluation pickles. |
| `models` | Exact saved model labels, including any RAG suffix. |
| `pathologies`, `difficulty` | Cohort selection; default difficulty is `first_diag`. |
| `experiments` | Saved run suffixes; `CDM_VANILLA` selects timestamp-only runs. |
| `base_seed`, `seeds` | Unsuffixed run seed and selected variants. Empty `seeds` selects the base run. |
| `multiple_matches` | `shortest` chooses a matching file; `error` requires a unique match. |
| `agent`, `score_fields` | Saved agent label and metrics to aggregate. |

Copy `configs/evaluate.yaml` to `configs/evaluate_local.yaml`, keep its defaults,
and run `python evaluate_cdm_rag.py --config-name evaluate_local`. CLI values
override YAML. Data paths are inherited from `configs/paths/example.yaml`.

Outputs include per-model and combined evaluation/result/score/retrieval pickles,
plus `evaluation_config.yaml`. Feed the experiment folders to
[inspection](inspection.md).

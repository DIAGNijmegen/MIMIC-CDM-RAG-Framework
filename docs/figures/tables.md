# Shared benchmark-table settings

These options apply to the [ablation](table_ablation.md) and
[retrieval-stack](table_retrieval_stack.md) scripts. Their profiles include
`configs/figures/_table.yaml`.

The scripts discover one CSV per pattern inside `results_dir`:

| Metric | Default pattern |
| --- | --- |
| Diagnosis | `DiagAccCDM_Fig3_*.csv` |
| Alignment | `OverallAlignment_PatientMean_*.csv` |
| Retrieval | `RetrievalMetrics_withMean_[0-9]*.csv` |
| Treatment | `TreatmentSummary_*.csv` |

The retrieval pattern excludes the wide-format export.

| Argument | Purpose |
| --- | --- |
| `metric_patterns.<source>` | Narrow a filename pattern when multiple files match. |
| `metric_dirs.<source>` | Read a metric from another folder; `null` uses `results_dir`. |
| `model_aliases` | Map legacy CSV IDs to configured IDs, without seed suffixes. |
| `duplicate_rows` | `mean` averages repeated records; `error` rejects them. |
| `missing_rows` | `keep` retains empty cells and records missing metrics; `error` rejects them. |
| `zero_as_missing` | Mask zero summary values; `false` preserves genuine zeros. |
| `figure_width`, `height_scale`, `min_height` | Table dimensions. |

Source keys are `diagnostic_accuracy`, `alignment`, `retrieval`, and `treatment`.
Missing baseline retrieval values are allowed. Use `render=false` for CSVs only;
see [shared arguments](README.md) for model selection and other export options.

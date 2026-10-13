# Figure scripts

Use inspection analysis values to produce figures.
Run from the repository root. Each script uses `configs/figures.yaml` and a
matching profile under `configs/figures/`.

| Script / profile | Guide |
| --- | --- |
| `main_figure` | [Main manuscript figure](main_figure.md) |
| `table_ablation` | [Ablation table](table_ablation.md) |
| `table_retrieval_stack` | [Retrieval-stack table](table_retrieval_stack.md) |
| `retrievals_figure` | [Retrieval composition](retrievals_figure.md) |
| `figures_orders_tests_treatments` | [Orders and treatments](figures_orders_tests_treatments.md) |
| `detailed_model_report` | [Per-model report](detailed_model_report.md) |
| `table_common` (helper) | [Shared table helpers](table_common.md) |

```bash
python make_figures/main_figure.py figures=main_figure \
  results_dir=/path/to/inspection_results output_dir=/path/to/figures
```

Shared arguments:

| Argument | Purpose |
| --- | --- |
| `results_dir`, `output_dir` | Input CSV folder and output folder. |
| `models` | Raw baseline key → `name`, `rag`, `size`, and `medical` metadata. |
| `selected_models` | Raw model keys to include; `null` uses all enabled models. |
| `base_seed`, `seeds` | Unsuffixed seed and saved variants to read. |
| `formats`, `dpi` | Export formats and raster resolution. |
| `render`, `statistics` | Images and calculation exports; `render=false` gives CSVs only. |
| `show` | Display figures interactively; defaults to `false`. |

Copy a profile, keep `# @package _global_`, and select it with `figures=my_profile`.
CLI values override profile and shared YAML settings. For external configs, use
`--config-path /path/to/configs --config-name figures` with the driver and profile
in that tree. `--cfg job --resolve` prints settings without reading inputs.
See [inspection](../inspection.md) to produce the CSVs, and [table settings](tables.md)
for options shared by the benchmark tables.

# Ablation table

`make_figures/table_ablation.py` compares baseline, corpus, and ablation variants,
with confidence intervals across seed-level model averages.

```bash
python make_figures/table_ablation.py figures=table_ablation \
  results_dir=/path/to/results output_dir=/path/to/figures
```

Profile: `configs/figures/table_ablation.yaml`.

| Argument | Purpose |
| --- | --- |
| `ablation_variants` | Rows with `name`, `label`, `group`, and baseline/RAG `source`. |
| Variant `replace`, `suffix` | Construct exact CSV IDs from each model's source ID. |
| Variant `alias_of` | Reuse another variant's summary; default `Augmented` aliases `w/o clean`. |
| `group_order`, `confidence` | Section order and confidence level. |
| `output_subdir`, `output_name` | Output folder and image stem. |

Defaults save images and calculation CSVs under `output_dir/ablation_table/`.
See [table settings](tables.md) for inputs and missing values, and
[shared arguments](README.md) for models, seeds, formats, and custom YAML.

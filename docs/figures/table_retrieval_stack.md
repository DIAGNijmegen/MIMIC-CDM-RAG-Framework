# Retrieval-stack table

`make_figures/table_retrieval_stack.py` compares lexical, semantic, and reranker
configurations, ranking metrics within each backbone.

```bash
python make_figures/table_retrieval_stack.py figures=table_retrieval_stack \
  results_dir=/path/to/results output_dir=/path/to/figures
```

Profile: `configs/figures/table_retrieval_stack.yaml`.

| Argument | Purpose |
| --- | --- |
| `stacks` | Raw CSV ID → backbone `model`, `lexical`, `semantic`, `reranker`, and `highlight`. |
| `selected_stacks` | Exact stack IDs to include; `null` includes stacks for selected backbones. |
| `seeds` | Defaults to `[base_seed]`; several seeds are averaged per stack. |
| `missing_rows` | Defaults to `keep`; use `error` to require complete data. |
| `output_subdir`, `output_name` | Output folder and image stem. |

Defaults save images and CSVs under `output_dir/stack_benchmark_table/`.
See [table settings](tables.md) and [shared arguments](README.md).

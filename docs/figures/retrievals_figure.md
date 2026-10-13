# Retrieval-composition figure

`make_figures/retrievals_figure.py` exports one figure with retrieval-class bars
and a document heatmap. Classes, documents, and height follow the selected data.

```bash
python make_figures/retrievals_figure.py figures=retrievals_figure \
  results_dir=/path/to/results output_dir=/path/to/figures
```

Profile: `configs/figures/retrievals_figure.yaml`. Inputs:
`RetrievalMetrics_perPathology_*.csv` and `RetrievalDocumentCounts_*.csv`.

| Argument | Purpose |
| --- | --- |
| `experiment_suffix` | Append to selected models' RAG IDs: empty, `_ambig`, `_4`, or `_nomd`. |
| `experiments` | Exact seed-independent IDs; overrides model/suffix selection. |
| `retrieval_dir`, `document_dir` | Optional separate input folders. |
| `retrieval_pattern`, `document_pattern` | CSV matching patterns. |
| `guideline_columns` | Class-to-column mapping; `null` infers active `Prop_*` classes. |
| `document_labels`, `document_name_rules` | Exact or filename-based display labels. |
| `pathology_colors`, `pathology_light_colors` | Class colors. |
| `figure_size`, `heatmap_title`, `heatmap_cmap` | Layout and heatmap appearance; null height is automatic. |
| `output_subdir`, `output_name` | Output folder and image stem. |

Defaults save images and CSVs under `output_dir/retrievals_figure/`.
See [shared arguments](README.md) for model selection, seeds, and formats.

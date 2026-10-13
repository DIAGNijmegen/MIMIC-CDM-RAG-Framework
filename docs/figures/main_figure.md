# Main manuscript figure

`make_figures/main_figure.py` plots overall/pathology diagnostic accuracy and
cohort/guideline composition, and exports statistical summaries.

```bash
python make_figures/main_figure.py figures=main_figure \
  results_dir=/path/to/results output_dir=/path/to/figures
```

Profile: `configs/figures/main_figure.yaml`. Input: `DiagAccCDM_Fig3_*.csv`.

| Argument | Purpose |
| --- | --- |
| `csv_pattern` | Match one diagnostic CSV; narrow it when several files exist. |
| `overall_dir`, `pathology_dir` | Optional separate input folders. |
| `output_name`, `figure_size` | Image filename stem and dimensions in inches. |
| `show_composition` | Include the cohort/guideline panel. |
| `cohort.counts`, `guidelines.counts` | Composition counts; update them for your dataset. |
| `pathologies`, `confidence` | Included pathologies and confidence level. |

PDF/SVG/PNG and summaries are saved under `output_dir`. See
[shared arguments](README.md) for model names, seeds, formats, and custom YAML.

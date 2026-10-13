# Detailed per-model report

`make_figures/detailed_model_report.py` exports a six-panel page per model:
diagnosis, retrieval, treatment, physical examinations, laboratory requests,
and imaging. Bars show seed means with sample SD error bars.

```bash
python make_figures/detailed_model_report.py figures=detailed_model_report \
  results_dir=/path/to/results output_dir=/path/to/figures
```

Profile: `configs/figures/detailed_model_report.yaml`. Required CSV prefixes:
`DiagAccCDM_Fig3`, `RetrievalMetrics_perPathology`, `RetrievalDocumentCounts`,
`TreatmentRequested_AllDiagnoses`, `PhysicalExaminationPercentages`,
`LaboratoryTestPercentages`, and `ImagingPercentages`.

| Argument | Purpose |
| --- | --- |
| `csv_patterns`, `input_dirs` | Per-source filename patterns and optional folders. |
| `selected_models`, `seeds`, `base_seed` | Models and saved runs to report. |
| `individual_pages` | Export each model in the selected formats. |
| `multipage_pdf` | Export a combined PDF when `pdf` is in `formats`. |
| `figure_size`, `dpi` | Page dimensions and raster resolution; defaults are `[7.05,9.65]` and 600 dpi. |
| `output_subdir`, `report_name` | Output folder and combined PDF stem. |
| `treatment_categories`, `laboratory_categories` | Pathology-to-category lists. |
| `physical_examination_categories`, `imaging_modalities`, `guideline_columns` | Included categories and retrieval columns. |
| `treatment_labels`, `physical_examination_labels` | Display labels. |
| `condition_colors`, `pathology_colors`, `pathology_light_colors` | Plot colors. |

Defaults save pages, a combined PDF, and calculation CSVs under
`output_dir/appendix_model_reports/`. Set `render=false` for CSVs only, or
`'formats=[pdf]' individual_pages=false statistics=false` for a combined PDF only.
Shared `cohort.counts` supplies subplot titles; update it for your dataset.
See [shared arguments](README.md) for model metadata and custom YAML.

# Order and treatment figures

`make_figures/figures_orders_tests_treatments.py` plots laboratory requests,
imaging, physical examinations, and treatment rates.

```bash
python make_figures/figures_orders_tests_treatments.py \
  figures=figures_orders_tests_treatments results_dir=/path/to/results \
  output_dir=/path/to/figures 'plots=[laboratory,physical_examination]'
```

Profile: `configs/figures/figures_orders_tests_treatments.yaml`.

| `plots` choice | Required CSV prefix |
| --- | --- |
| `laboratory` | `LaboratoryTestPercentages` |
| `imaging_patients` | `ImagingPercentages`, `ImagingAllRequests` |
| `imaging_stacked_percentages`, `imaging_stacked_counts` | `ImagingPercentages` |
| `physical_examination` | `PhysicalExaminationPercentages` |
| `treatment_all_diagnoses` | `TreatmentRequested_AllDiagnoses` |
| `treatment_correct_diagnosis` | `TreatmentRequested_CorrectDiagnosis` |

All plot types are selected by default. Files match `<prefix>_*.csv`.

| Argument | Purpose |
| --- | --- |
| `csv_patterns`, `input_dirs` | Per-source matching patterns and optional folders. |
| `laboratory_categories`, `treatment_categories` | Pathology-to-category lists. |
| `physical_examination_categories`, `imaging_modalities` | Included categories. |
| `category_labels`, `condition_colors`, `point_colors`, `modality_hatches` | Labels and appearance. |
| `include_doctors`, `doctor_model` | Clinician comparator for laboratory/imaging plots. |
| `show_values_table`, `show_violins` | Toggle tables and violin backgrounds. |
| `panel_columns`, `figure_width`, `panel_height`, `stacked_height` | Layout. |
| `output_subdir`, `output_prefix` | Output folder and filenames. |

`imaging_patients` counts unique patients per modality using shared `cohort.counts`;
update these counts for your dataset. Stacks use the source imaging summaries.
Treatment rates use the full treatment cohort or correctly diagnosed cohort,
according to the selected plot.

Defaults save outputs under `output_dir/orders_tests_treatments/`.
See [shared arguments](README.md) for models, seeds, formats, and CSV-only export.

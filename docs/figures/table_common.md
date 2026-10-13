# Shared table helpers

`make_figures/table_common.py` is imported by the table scripts; it has no CLI.
It loads and validates metric CSVs, parses model/seed identifiers, calculates
ranks, and exports images, CSVs, and resolved configurations.

Configure its behavior through `configs/figures/_table.yaml` and the selected
table profile. See [shared table settings](tables.md),
[ablation](table_ablation.md), and [retrieval-stack](table_retrieval_stack.md).

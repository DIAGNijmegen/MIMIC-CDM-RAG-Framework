"""Build the ablation benchmark table from inspection CSVs using configs/figures.yaml.

Select figures=table_ablation; inputs are discovered inside results_dir.
See docs/tables.md. Importing this module performs no file I/O or rendering.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

if __package__:
    from . import table_common as common
else:
    import table_common as common


def configuration_id(model, variant):
    source = variant.get("source", "rag")
    if source not in ("baseline", "rag"):
        raise ValueError("Variant source must be baseline or rag.")
    identifier = model.no_rag if source == "baseline" else model.rag
    for before, after in variant.get("replace", {}).items():
        if not before or before not in identifier:
            raise ValueError(f"Replacement {before!r} does not match {identifier!r}.")
        identifier = identifier.replace(before, after)
    return identifier + variant.get("suffix", "")


def prepare_data(args):
    """Average models within each seed, then calculate the inter-seed Student-t CI."""
    from scipy.stats import t

    pairs = common.validate_settings(args, "ablation_variants", "table_ablation")
    variants = list(args.ablation_variants)
    names = [variant.name for variant in variants]
    if not variants or len(set(names)) != len(names):
        raise ValueError("ablation_variants must have unique names.")
    definitions = []
    aliases = []
    for variant in variants:
        if not variant.name or not variant.label or not variant.group:
            raise ValueError("Every ablation variant needs name, label and group.")
        if variant.get("alias_of"):
            if variant.alias_of not in names or variant.alias_of == variant.name:
                raise ValueError(f"Invalid alias source: {variant.alias_of}")
            aliases.append(variant)
            continue
        for model in pairs:
            definitions.append(
                {
                    "ConfigID": configuration_id(model, variant),
                    "Name": model.no_rag,
                    "Model Name": model.label,
                    "Experiment": variant.name,
                }
            )
    definitions = pd.DataFrame(definitions)
    if definitions.empty:
        raise ValueError("Configure at least one non-alias ablation variant.")
    combined, paths, missing = common.load_metrics(
        args, definitions["ConfigID"].tolist(), {model.no_rag for model in pairs}
    )
    combined = combined.merge(
        definitions, on="ConfigID", how="left", validate="many_to_one"
    )
    metrics = [*common.METRICS, common.OPTIONAL_METRIC]
    model_seed = combined.groupby(["Experiment", "Seed", "Name"], as_index=False)[
        metrics
    ].mean()
    seed_level = model_seed.groupby(["Experiment", "Seed"], as_index=False)[
        metrics
    ].mean()
    summary = seed_level.groupby("Experiment")[metrics].agg(["count", "mean", "std"])
    for metric in metrics:
        n, mean, sd = (summary[(metric, key)] for key in ("count", "mean", "std"))
        margin = pd.Series(np.nan, index=summary.index, dtype=float)
        valid = n.gt(1) & sd.notna()
        margin.loc[valid] = (
            t.ppf((1 + args.confidence) / 2, n.loc[valid] - 1)
            * sd.loc[valid]
            / np.sqrt(n.loc[valid])
        )
        summary[(metric, "ci_low")] = mean - margin
        summary[(metric, "ci_high")] = mean + margin
    summary = summary.reindex(
        columns=pd.MultiIndex.from_product(
            [metrics, ["count", "mean", "std", "ci_low", "ci_high"]]
        )
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    for alias in aliases:
        row = summary.loc[summary["Experiment"].eq(alias.alias_of)].copy()
        if len(row) != 1:
            raise ValueError(
                f"Alias {alias.name!r} requires one row for {alias.alias_of!r}."
            )
        row["Experiment"] = alias.name
        summary = pd.concat([summary, row], ignore_index=True)
    if args.zero_as_missing:
        summary = summary.replace(0, np.nan)
    summary["Confidence Level"] = args.confidence
    descriptions = {variant.name: variant for variant in variants}
    table = pd.DataFrame(
        {
            "Group": summary["Experiment"].map(lambda name: descriptions[name].group),
            "Method": summary["Experiment"].map(lambda name: descriptions[name].label),
        }
    )
    for metric, label in zip(common.METRICS, common.PERFORMANCE_COLUMNS):
        multiplier = 100 if metric == "Precision@6_equiv" else 1
        table[label] = summary[f"{metric}_mean"] * multiplier
        table[common.CI_COLUMNS[label]] = (
            (summary[f"{metric}_ci_high"] - summary[f"{metric}_ci_low"])
            / 2
            * multiplier
        )
    _, ranks = common.ranked_metrics(table)
    table.insert(2, common.RANK_COLUMN, ranks)
    order = list(dict.fromkeys(args.group_order))
    order.extend(group for group in table["Group"].unique() if group not in order)
    table = pd.concat(
        [
            table.loc[table["Group"].eq(group)].sort_values(
                common.RANK_COLUMN, ascending=False, na_position="last", kind="stable"
            )
            for group in order
        ],
        ignore_index=True,
    )
    print(
        "CI method: models averaged within each seed, then Student-t CI across seed means."
    )
    return common.TableData(
        table,
        {
            "ablation_benchmark_table": table,
            "ablation_summary": summary,
            "model_seed_metrics": model_seed,
            "seed_level_metrics": seed_level,
            "combined_metrics": combined,
            "missing_metrics": missing,
        },
        paths,
        order,
        {pair.no_rag: pair.label for pair in pairs},
    )


def render_table(result, args):
    """Render the notebook table layout without saving or displaying it."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    data = result.table
    directions = common.DIRECTIONS
    value_formats = common.VALUE_FORMATS
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    group_order = result.section_order
    ci_columns = common.CI_COLUMNS
    ci_formats = {key: common.VALUE_FORMATS[key] for key in common.PERFORMANCE_COLUMNS}
    metric_columns = [
        "Rank by\nmedian rank",
        "Diagnostic\naccuracy %",
        "Test-ordering\nalignment %",
        "Treatment\nalignment %",
        "Retrieval\nP@6 %",
        "Retrieval spread\n(Brier score)",
    ]
    color_map = LinearSegmentedColormap.from_list("bad_to_good", ["#e9918b", "#d0e8cd"])
    normalized_scores = pd.DataFrame(
        index=data.index, columns=metric_columns, dtype=float
    )
    for column in metric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        valid_values = values.dropna()
        if valid_values.empty:
            score = pd.Series(np.nan, index=data.index, dtype=float)
        elif np.isclose(valid_values.min(), valid_values.max()):
            score = pd.Series(0.5, index=data.index, dtype=float)
            score.loc[values.isna()] = np.nan
        else:
            score = (values - valid_values.min()) / (
                valid_values.max() - valid_values.min()
            )
        if directions[column] == "lower":
            score = 1.0 - score
        normalized_scores[column] = score
    column_widths = [3.25, 2.05, 2.4, 2.55, 2.55, 2.2, 2.75]
    x_edges = np.concatenate(([0], np.cumsum(column_widths)))
    total_width = x_edges[-1]
    header_height = 1.22
    group_height = 0.58
    row_height = 0.7
    displayed_groups = [group for group in group_order if data["Group"].eq(group).any()]
    n_group_headers = sum(group != "Baseline" for group in displayed_groups)
    total_height = (
        header_height + n_group_headers * group_height + len(data) * row_height + 0.3
    )
    figure_height = max(args.min_height, total_height * args.height_scale)
    (fig, ax) = plt.subplots(figsize=(args.figure_width, figure_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")
    y = total_height - 0.08
    header_bottom = y - header_height
    if "Baseline" in displayed_groups:
        ax.text(
            x_edges[0] + 0.14,
            header_bottom + 0.74,
            "Baseline",
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#303030",
        )
    for column_index, column in enumerate(metric_columns, start=1):
        center_x = (x_edges[column_index] + x_edges[column_index + 1]) / 2
        ax.text(
            center_x,
            header_bottom + 0.74,
            column,
            ha="center",
            va="center",
            fontsize=11.3,
            fontweight="bold",
            color="#474747",
            linespacing=0.95,
        )
        direction_text = (
            "lower better" if directions[column] == "lower" else "higher better"
        )
        direction_arrow = "↓" if directions[column] == "lower" else "↑"
        ax.text(
            center_x - 0.06,
            header_bottom + 0.22,
            direction_text,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#777777",
        )
        ax.text(
            x_edges[column_index + 1] - 0.16,
            header_bottom + 0.22,
            direction_arrow,
            ha="center",
            va="center",
            fontsize=13.5,
            color="#b2b2b2",
            fontweight="bold",
        )
    ax.hlines(header_bottom, 0, total_width, color="#3f3f3f", linewidth=1.2)
    y = header_bottom
    for group in displayed_groups:
        group_data = data.loc[data["Group"].eq(group)]
        if group != "Baseline":
            y -= group_height
            ax.text(
                0.14,
                y + group_height * 0.54,
                group,
                ha="left",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color="#303030",
            )
            ax.hlines(y + 0.04, 0, total_width, color="#444444", linewidth=1.0)
        for row_index, row in group_data.iterrows():
            row_bottom = y - row_height
            is_framework = group == "Our Framework"
            ax.text(
                x_edges[0] + 0.18,
                row_bottom + row_height / 2,
                row["Method"],
                ha="left",
                va="center",
                fontsize=11,
                color="#505050",
                fontweight="bold" if is_framework else "normal",
            )
            for column_index, column in enumerate(metric_columns, start=1):
                value = row[column]
                if pd.isna(value):
                    face_color = (
                        "white" if column == "Rank by\nmedian rank" else "#f5f5f5"
                    )
                    value_text = "—"
                else:
                    if column == "Rank by\nmedian rank":
                        face_color = "white"
                    else:
                        score = normalized_scores.loc[row_index, column]
                        rgba = color_map(float(score))
                        white_mix = 0.18
                        face_color = tuple(
                            (1 - white_mix) * channel + white_mix
                            for channel in rgba[:3]
                        )
                    value_text = value_formats[column].format(value)
                ax.add_patch(
                    Rectangle(
                        (x_edges[column_index], row_bottom),
                        column_widths[column_index],
                        row_height,
                        facecolor=face_color,
                        edgecolor="none",
                    )
                )
                text_parts = [
                    TextArea(
                        value_text,
                        textprops={
                            "fontsize": 10.9,
                            "color": "#222222",
                            "fontweight": "bold"
                            if column == "Rank by\nmedian rank"
                            and pd.notna(value)
                            and np.isclose(value, 1)
                            else "normal",
                        },
                    )
                ]
                ci_column = ci_columns.get(column)
                if ci_column is not None and pd.notna(value):
                    ci_value = row.get(ci_column, np.nan)
                    if pd.notna(ci_value):
                        ci_text = "  ± " + ci_formats[column].format(ci_value)
                        text_parts.append(
                            TextArea(
                                ci_text,
                                textprops={
                                    "fontsize": 8.9,
                                    "color": "#777777",
                                    "fontweight": "normal",
                                },
                            )
                        )
                packed_text = HPacker(children=text_parts, align="center", pad=0, sep=0)
                packed_annotation = AnnotationBbox(
                    packed_text,
                    (x_edges[column_index] + 0.14, row_bottom + row_height / 2),
                    xycoords="data",
                    box_alignment=(0, 0.5),
                    frameon=False,
                    pad=0,
                )
                ax.add_artist(packed_annotation)
            y = row_bottom
    return (fig, ax)


def main(args: DictConfig):
    return common.export_table(args, prepare_data(args), render_table, "table_ablation")


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs"),
    config_name="figures",
    version_base=None,
)
def run(args: DictConfig):
    main(args)


if __name__ == "__main__":
    run()

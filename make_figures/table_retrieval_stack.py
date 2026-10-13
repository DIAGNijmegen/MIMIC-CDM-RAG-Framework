"""Build the retrieval-stack benchmark table from inspection CSVs using configs/figures.yaml.

Select figures=table_retrieval_stack; inputs are discovered inside results_dir.
See docs/figures/table_retrieval_stack.md. Importing this module performs no file I/O or rendering.
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


def prepare_data(args):
    """Compare explicitly configured stacks, ranking metrics within each backbone."""
    pairs = common.validate_settings(args, "stacks", "table_retrieval_stack")
    names = {pair.no_rag: pair.label for pair in pairs}
    if not isinstance(args.stacks, DictConfig) or not args.stacks:
        raise ValueError("stacks must map raw configuration IDs to stack metadata.")
    selected = args.selected_stacks
    if selected is not None:
        if not isinstance(selected, (list, tuple, common.ListConfig)) or not selected:
            raise ValueError("selected_stacks must be a nonempty list, or null.")
        unknown = set(selected) - set(args.stacks)
        if unknown:
            raise ValueError(f"Unknown stack IDs: {sorted(unknown)}")
        selected = set(selected)
    rows = []
    for index, (identifier, entry) in enumerate(args.stacks.items()):
        if selected is not None and identifier not in selected:
            continue
        for field in ("model", "lexical", "semantic", "reranker"):
            if (
                field not in entry
                or not isinstance(entry[field], str)
                or not entry[field]
            ):
                raise ValueError(f"Stack {identifier!r} needs {field!r}.")
        if entry.model not in names:
            if selected is not None:
                raise ValueError(
                    f"{identifier}: backbone {entry.model!r} is not selected in shared models."
                )
            continue
        if type(entry.get("highlight", False)) is not bool:
            raise ValueError("Stack highlight flags must be true or false.")
        rows.append(
            {
                "StackOrder": index,
                "StackID": identifier,
                "Backbone": entry.model,
                "Backbone Name": names[entry.model],
                "Lexical": entry.lexical,
                "Semantic": entry.semantic,
                "Reranker": entry.reranker,
                "Selected Stack": entry.get("highlight", False),
            }
        )
    definitions = pd.DataFrame(rows)
    if definitions.empty:
        raise ValueError("No configured stacks match the selected shared model keys.")
    seed_metrics, paths, missing = common.load_metrics(
        args, definitions["StackID"].tolist(), set(names)
    )
    numeric = seed_metrics.groupby("ConfigID", as_index=False)[
        [*common.METRICS, common.OPTIONAL_METRIC]
    ].mean()
    combined = definitions.merge(
        numeric,
        left_on="StackID",
        right_on="ConfigID",
        how="left",
        validate="one_to_one",
    ).drop(columns="ConfigID")
    if args.zero_as_missing:
        combined[common.METRICS] = combined[common.METRICS].replace(0, np.nan)
    table = pd.concat(
        [combined[definitions.columns], common.metric_plot_values(combined)], axis=1
    )
    table[common.RANK_COLUMN] = np.nan
    table["Median metric rank"] = np.nan
    order = [
        model.no_rag for model in pairs if table["Backbone"].eq(model.no_rag).any()
    ]
    for backbone in order:
        index = table.index[table["Backbone"].eq(backbone)]
        median, ranks = common.ranked_metrics(table.loc[index])
        table.loc[index, "Median metric rank"] = median
        table.loc[index, common.RANK_COLUMN] = ranks
    table = pd.concat(
        [
            table.loc[table["Backbone"].eq(backbone)].sort_values(
                [common.RANK_COLUMN, "Diagnostic\naccuracy %", "StackOrder"],
                ascending=[False, True, True],
                na_position="last",
                kind="stable",
            )
            for backbone in order
        ],
        ignore_index=True,
    )
    return common.TableData(
        table,
        {
            "retrieval_stack_benchmark_table": table,
            "combined_metrics": combined,
            "seed_metrics": seed_metrics,
            "missing_metrics": missing,
        },
        paths,
        order,
        names,
    )


def render_table(result, args):
    """Render the notebook table layout without saving or displaying it."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    data = result.table
    directions = common.DIRECTIONS
    value_formats = common.VALUE_FORMATS
    backbone_order = result.section_order
    BACKBONE_DISPLAY_NAMES = result.model_names
    descriptor_columns = ["Lexical", "Semantic", "Reranker"]
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
        np.nan, index=data.index, columns=metric_columns, dtype=float
    )
    for backbone in backbone_order:
        idx = data.index[data["Backbone"].eq(backbone)]
        if len(idx) == 0:
            continue
        for column in metric_columns:
            values = pd.to_numeric(data.loc[idx, column], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                score = pd.Series(np.nan, index=idx, dtype=float)
            elif np.isclose(valid.min(), valid.max()):
                score = pd.Series(0.5, index=idx, dtype=float)
                score.loc[values.isna()] = np.nan
            else:
                score = (values - valid.min()) / (valid.max() - valid.min())
            if directions[column] == "lower":
                score = 1.0 - score
            normalized_scores.loc[idx, column] = score
    column_widths = [1.55, 3.0, 3.15, 1.9, 2.25, 2.45, 2.4, 2.05, 2.65]
    all_visible_columns = descriptor_columns + metric_columns
    x_edges = np.concatenate(([0], np.cumsum(column_widths)))
    total_width = x_edges[-1]
    top_header_height = 0.62
    sub_header_height = 1.16
    backbone_height = 0.76
    row_height = 0.7
    displayed_backbones = [b for b in backbone_order if data["Backbone"].eq(b).any()]
    total_height = (
        top_header_height
        + sub_header_height
        + len(displayed_backbones) * backbone_height
        + len(data) * row_height
        + 0.24
    )
    figure_height = max(args.min_height, total_height * args.height_scale)
    (fig, ax) = plt.subplots(figsize=(args.figure_width, figure_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")
    y_top = total_height - 0.06
    super_bottom = y_top - top_header_height
    retrieval_left = x_edges[0]
    retrieval_right = x_edges[len(descriptor_columns)]
    results_left = retrieval_right
    results_right = total_width
    header_fill = "white"
    ax.add_patch(
        Rectangle(
            (retrieval_left, super_bottom),
            retrieval_right - retrieval_left,
            top_header_height,
            facecolor=header_fill,
            edgecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (results_left, super_bottom),
            results_right - results_left,
            top_header_height,
            facecolor=header_fill,
            edgecolor="none",
        )
    )
    ax.text(
        (retrieval_left + retrieval_right) / 2,
        super_bottom + top_header_height / 2,
        "Retrieval stack",
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        color="#303030",
    )
    ax.text(
        (results_left + results_right) / 2,
        super_bottom + top_header_height / 2,
        "Performance",
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        color="#303030",
    )
    ax.hlines(super_bottom, 0, total_width, color="#3f3f3f", linewidth=1.0)
    header_bottom = super_bottom - sub_header_height
    for col_idx, column in enumerate(all_visible_columns):
        left = x_edges[col_idx]
        right = x_edges[col_idx + 1]
        center_x = (left + right) / 2
        ax.add_patch(
            Rectangle(
                (left, header_bottom),
                right - left,
                sub_header_height,
                facecolor="white",
                edgecolor="none",
            )
        )
        if column in descriptor_columns:
            ax.text(
                center_x,
                header_bottom + 0.67,
                column,
                ha="center",
                va="center",
                fontsize=11.0,
                fontweight="bold",
                color="#404040",
            )
        else:
            ax.text(
                center_x,
                header_bottom + 0.73,
                column,
                ha="center",
                va="center",
                fontsize=10.7,
                fontweight="bold",
                color="#474747",
                linespacing=0.95,
            )
            direction_text = (
                "lower better" if directions[column] == "lower" else "higher better"
            )
            direction_arrow = "↓" if directions[column] == "lower" else "↑"
            ax.text(
                center_x - 0.05,
                header_bottom + 0.21,
                direction_text,
                ha="center",
                va="center",
                fontsize=8.7,
                color="#777777",
            )
            ax.text(
                right - 0.13,
                header_bottom + 0.21,
                direction_arrow,
                ha="center",
                va="center",
                fontsize=12.7,
                color="#b2b2b2",
                fontweight="bold",
            )
    ax.hlines(header_bottom, 0, total_width, color="#3f3f3f", linewidth=1.15)
    y = header_bottom
    for backbone in displayed_backbones:
        section = data.loc[data["Backbone"].eq(backbone)]
        y -= backbone_height
        ax.add_patch(
            Rectangle(
                (0, y),
                total_width,
                backbone_height,
                facecolor="white",
                edgecolor="none",
            )
        )
        ax.text(
            0.14,
            y + backbone_height * 0.45,
            BACKBONE_DISPLAY_NAMES.get(backbone, backbone),
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#303030",
        )
        ax.hlines(y + 0.02, 0, total_width, color="#444444", linewidth=1.0)
        for row_index, row in section.iterrows():
            row_bottom = y - row_height
            is_selected_stack = row["Selected Stack"]
            descriptor_fill = "#f2f2f2" if is_selected_stack else "white"
            for col_idx, column in enumerate(descriptor_columns):
                left = x_edges[col_idx]
                width = column_widths[col_idx]
                ax.add_patch(
                    Rectangle(
                        (left, row_bottom),
                        width,
                        row_height,
                        facecolor=descriptor_fill,
                        edgecolor="none",
                    )
                )
                text_pad = 0.18 if column == "Reranker" else 0.12
                ax.text(
                    left + text_pad,
                    row_bottom + row_height / 2,
                    str(row[column]),
                    ha="left",
                    va="center",
                    fontsize=10.2,
                    color="#303030" if is_selected_stack else "#4d4d4d",
                    fontweight="bold" if is_selected_stack else "normal",
                )
            for local_idx, column in enumerate(metric_columns):
                col_idx = len(descriptor_columns) + local_idx
                left = x_edges[col_idx]
                width = column_widths[col_idx]
                value = row[column]
                if pd.isna(value):
                    face_color = (
                        "#f2f2f2"
                        if is_selected_stack and column == "Rank by\nmedian rank"
                        else "white"
                    )
                    value_text = "—"
                else:
                    if column == "Rank by\nmedian rank":
                        face_color = "#f2f2f2" if is_selected_stack else "white"
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
                        (left, row_bottom),
                        width,
                        row_height,
                        facecolor=face_color,
                        edgecolor="none",
                    )
                )
                ax.text(
                    left + width / 2,
                    row_bottom + row_height / 2,
                    value_text,
                    ha="center",
                    va="center",
                    fontsize=10.6,
                    color="#222222",
                    fontweight="bold"
                    if column == "Rank by\nmedian rank"
                    and (
                        is_selected_stack or (pd.notna(value) and np.isclose(value, 1))
                    )
                    else "normal",
                )
            y = row_bottom
    ax.vlines(retrieval_right, y, y_top, color="#3f3f3f", linewidth=1.15, zorder=20)
    return (fig, ax)


def main(args: DictConfig):
    return common.export_table(
        args, prepare_data(args), render_table, "table_retrieval_stack"
    )


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs"),
    config_name="figures",
    version_base=None,
)
def run(args: DictConfig):
    main(args)


if __name__ == "__main__":
    run()

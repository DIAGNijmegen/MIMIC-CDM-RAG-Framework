"""Render one configurable retrieval-composition figure from inspection CSVs.

Select figures=retrievals_figure through configs/figures.yaml. See
docs/retrievals_figure.md. Importing this module performs no I/O or rendering.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

if __package__:
    from .main_figure import configured_model_pairs, find_accuracy_csv
    from .table_common import split_identifier
else:
    from main_figure import configured_model_pairs, find_accuracy_csv
    from table_common import split_identifier

RUN_KEYS = ["Experiment", "Seed", "Pathology"]


@dataclass
class RetrievalData:
    guideline_mean: pd.DataFrame
    guideline_sd: pd.DataFrame
    heatmap: pd.DataFrame
    documents: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    input_files: dict[str, Path]
    experiments: list[str]
    guideline_columns: dict[str, str]


def nonempty_strings(value, name):
    if (
        not isinstance(value, (ListConfig, list, tuple))
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty list of strings.")
    return list(dict.fromkeys(value))


def validate_config(args):
    if "retrieval_pattern" not in args:
        raise ValueError("Select the matching profile with figures=retrievals_figure.")
    for field in ("render", "statistics", "show"):
        if type(args[field]) is not bool:
            raise ValueError(f"{field} must be true or false.")
    if not args.render and not args.statistics:
        raise ValueError("Enable render or statistics.")
    if (
        not isinstance(args.seeds, (ListConfig, list, tuple))
        or not args.seeds
        or type(args.base_seed) is not int
        or any(type(seed) is not int for seed in args.seeds)
    ):
        raise ValueError("base_seed and seeds must be integers; seeds cannot be empty.")
    nonempty_strings(args.pathologies, "pathologies")
    if type(args.dpi) is not int or args.dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if args.render and (
        not isinstance(args.formats, (ListConfig, list, tuple))
        or not args.formats
        or any(fmt not in ("pdf", "svg", "png", "jpg") for fmt in args.formats)
    ):
        raise ValueError("formats must select pdf, svg, png and/or jpg.")
    size = args.figure_size
    if not isinstance(size, (ListConfig, list, tuple)) or len(size) != 2:
        raise ValueError(
            "figure_size must be [width, height], with optional null height."
        )
    for index, value in enumerate(size):
        if value is None and index == 1:
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Figure dimensions must be positive finite numbers.")
    for field in ("output_name", "output_subdir"):
        value = args[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or Path(value).name != value
            or value in (".", "..")
        ):
            raise ValueError(f"{field} must be a single filename/directory name.")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("output_dir must be a nonempty path.")
    if not isinstance(args.experiment_suffix, str):
        raise TypeError("experiment_suffix must be a string.")
    if args.experiments is not None:
        return nonempty_strings(args.experiments, "experiments")
    pairs = configured_model_pairs(args)
    experiments = [pair.rag + args.experiment_suffix for pair in pairs]
    if len(set(experiments)) != len(experiments):
        raise ValueError("Selected RAG experiment IDs must be unique.")
    return experiments


def read_selected(path, args, experiments, required):
    """Read exact experiment/seed/pathology matches, preserving arbitrary seeds."""
    frame = pd.read_csv(path)
    missing = {"Model", "Pathology", *required} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}.")
    identifiers = (
        frame["Raw Model"].fillna(frame["Model"])
        if "Raw Model" in frame
        else frame["Model"]
    )
    identities = [
        split_identifier(
            identifier,
            frame.at[index, "Seed"] if "Seed" in frame else np.nan,
            args.base_seed,
        )
        for index, identifier in identifiers.items()
    ]
    frame["Experiment"] = [experiment for experiment, _ in identities]
    frame["Seed"] = [seed for _, seed in identities]
    return frame.loc[
        frame["Experiment"].isin(experiments)
        & frame["Seed"].isin(args.seeds)
        & frame["Pathology"].isin(args.pathologies)
    ].copy()


def numeric_values(frame, columns, name, proportions=False):
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name}: values must be finite and nonnegative.")
    if proportions and (values > 1).any():
        raise ValueError(f"{name}: proportions must be between 0 and 1.")


def check_runs(frame, expected, name, unique=False):
    if unique and frame.duplicated(RUN_KEYS).any():
        raise ValueError(f"{name}: duplicate experiment/seed/pathology rows.")
    actual = pd.MultiIndex.from_frame(frame[RUN_KEYS].drop_duplicates())
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(f"{name}: missing required runs: {missing.tolist()}.")


def document_description(reference, retrieved_label, args):
    """Use exact overrides, then configured filename rules, then a concise filename."""
    override = args.document_labels.get(reference)
    rule = next(
        (
            item
            for item in args.document_name_rules
            if str(item.match).lower() in reference.lower()
        ),
        None,
    )
    label = override if override is not None else rule
    if isinstance(label, str):
        return retrieved_label, label
    if label is not None:
        if not isinstance(label, DictConfig) or not isinstance(label.get("name"), str):
            raise ValueError(f"Document label for {reference!r} needs a name.")
        specific = label.get("pathology", retrieved_label)
        if not isinstance(specific, str) or not specific:
            raise ValueError(f"Document pathology for {reference!r} must be text.")
        name = label.name
        if specific != retrieved_label:
            name = f"{specific}\n{name}"
        return specific, name
    filename = Path(reference.replace("\\", "/")).stem
    filename = re.sub(r"\s+", " ", filename.replace("_", " ").replace("-", " ")).strip()
    return retrieved_label, filename if len(filename) <= 52 else filename[:49] + "..."


def prepare_data(args):
    """Average runs within each seed, then seeds; never weight by chunk volume."""
    experiments = validate_config(args)
    pathologies = list(dict.fromkeys(args.pathologies))
    seeds = list(dict.fromkeys(args.seeds))
    paths = {
        "retrieval": find_accuracy_csv(
            args.retrieval_dir or args.results_dir,
            "retrieval_dir / results_dir",
            args.retrieval_pattern,
        ),
        "documents": find_accuracy_csv(
            args.document_dir or args.results_dir,
            "document_dir / results_dir",
            args.document_pattern,
        ),
    }
    retrieval = read_selected(paths["retrieval"], args, experiments, ["TotalChunks"])
    documents = read_selected(
        paths["documents"],
        args,
        experiments,
        ["DocumentReference", "RetrievedLabel", "ChunkCount"],
    )
    expected = pd.MultiIndex.from_product(
        [experiments, seeds, pathologies], names=RUN_KEYS
    )
    check_runs(retrieval, expected, "Retrieval proportions", unique=True)
    check_runs(documents, expected, "Document counts")
    for column in ("DocumentReference", "RetrievedLabel"):
        if (
            documents[column].isna().any()
            or not documents[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"{column} must contain nonempty strings.")
    if (documents.groupby("DocumentReference")["RetrievedLabel"].nunique() > 1).any():
        raise ValueError("A document has inconsistent RetrievedLabel values.")
    numeric_values(retrieval, ["TotalChunks"], "TotalChunks")
    numeric_values(documents, ["ChunkCount"], "ChunkCount")
    if retrieval["TotalChunks"].le(0).any():
        raise ValueError("Every selected run must have TotalChunks > 0.")
    if args.guideline_columns is None:
        available = {key[5:]: key for key in retrieval if key.startswith("Prop_")}
        columns = {
            label: available[label] for label in pathologies if label in available
        }
        columns.update(
            {label: key for label, key in available.items() if label not in columns}
        )
    elif isinstance(args.guideline_columns, DictConfig) and args.guideline_columns:
        columns = dict(args.guideline_columns)
    else:
        raise ValueError("guideline_columns must be a label-to-column mapping or null.")
    if not columns or set(columns.values()) - set(retrieval.columns):
        raise ValueError("Required guideline proportion columns are missing.")
    if any(not isinstance(label, str) or not label for label in columns) or len(
        set(columns.values())
    ) != len(columns):
        raise ValueError(
            "guideline_columns must use distinct columns and nonempty labels."
        )
    numeric_values(
        retrieval, list(columns.values()), "Guideline proportions", proportions=True
    )
    if not np.allclose(
        retrieval[list(columns.values())].sum(axis=1), 1, atol=1e-8, rtol=0
    ):
        raise ValueError("Guideline proportions must sum to 1 in every selected run.")
    # The exporter includes Prop_Other even for corpora containing no Other documents.
    if args.guideline_columns is None:
        columns = {
            label: key
            for label, key in columns.items()
            if label in pathologies
            or label in set(documents["RetrievedLabel"])
            or retrieval[key].ne(0).any()
        }
    unknown = set(documents["RetrievedLabel"]) - set(columns)
    if unknown:
        raise ValueError(
            f"Document labels have no guideline proportion column: {sorted(unknown)}."
        )
    counts = documents.groupby([*RUN_KEYS, "DocumentReference"], as_index=False)[
        "ChunkCount"
    ].sum()
    metadata = (
        documents[["DocumentReference", "RetrievedLabel"]]
        .drop_duplicates()
        .set_index("DocumentReference")
    )
    descriptions = [
        document_description(reference, row.RetrievedLabel, args)
        for reference, row in metadata.iterrows()
    ]
    metadata["SpecificPathology"] = [specific for specific, _ in descriptions]
    metadata["DisplayName"] = [name for _, name in descriptions]
    full_index = pd.MultiIndex.from_product(
        [experiments, seeds, pathologies, metadata.index],
        names=[*RUN_KEYS, "DocumentReference"],
    )
    counts = (
        counts.set_index([*RUN_KEYS, "DocumentReference"])["ChunkCount"]
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    totals = counts.groupby(RUN_KEYS)["ChunkCount"].sum()
    retrieval_totals = retrieval.set_index(RUN_KEYS)["TotalChunks"].reindex(
        totals.index
    )
    if (totals <= 0).any() or not np.allclose(
        totals, retrieval_totals, atol=1e-8, rtol=0
    ):
        raise ValueError("Document chunk totals disagree with retrieval TotalChunks.")
    counts = counts.merge(
        totals.rename("TotalRetrievedChunks").reset_index(),
        on=RUN_KEYS,
        validate="many_to_one",
    )
    counts["SharePct"] = 100 * counts["ChunkCount"] / counts["TotalRetrievedChunks"]
    # Both panels must describe the same chunks, not just similarly named runs.
    labeled_counts = counts.merge(
        metadata[["RetrievedLabel"]], on="DocumentReference", validate="many_to_one"
    )
    label_shares = (
        labeled_counts.groupby([*RUN_KEYS, "RetrievedLabel"])["SharePct"]
        .sum()
        .unstack(fill_value=0)
    )
    for label, column in columns.items():
        shares = label_shares.reindex(columns=[label], fill_value=0)[label]
        expected_shares = (
            retrieval.set_index(RUN_KEYS)[column].reindex(shares.index) * 100
        )
        if not np.allclose(shares, expected_shares, atol=1e-6, rtol=0):
            raise ValueError(f"Document counts disagree with {column} proportions.")
    guideline_seed = (
        retrieval.groupby(["Seed", "Pathology"])[list(columns.values())].mean() * 100
    )
    guideline_seed = guideline_seed.rename(
        columns={key: label for label, key in columns.items()}
    ).reindex(
        pd.MultiIndex.from_product([seeds, pathologies], names=["Seed", "Pathology"])
    )
    guideline_mean = guideline_seed.groupby("Pathology").mean().reindex(pathologies)
    guideline_sd = guideline_seed.groupby("Pathology").std(ddof=1).reindex(pathologies)
    document_seed = counts.groupby(
        ["Seed", "Pathology", "DocumentReference"], as_index=False
    )["SharePct"].mean()
    document_mean = document_seed.groupby(
        ["Pathology", "DocumentReference"], as_index=False
    ).agg(MeanSharePct=("SharePct", "mean"))
    heatmap = document_mean.pivot(
        index="DocumentReference", columns="Pathology", values="MeanSharePct"
    ).reindex(columns=pathologies)
    metadata["OverallMeanShare"] = heatmap.mean(axis=1)
    metadata["LabelOrder"] = metadata["RetrievedLabel"].map(
        {label: index for index, label in enumerate(columns)}
    )
    metadata = metadata.sort_values(
        ["LabelOrder", "OverallMeanShare"], ascending=[True, False], kind="stable"
    )
    heatmap = heatmap.reindex(metadata.index)
    mean_long = (
        guideline_mean.rename_axis(columns="GuidelineType")
        .stack()
        .rename("MeanPct")
        .reset_index()
    )
    sd_long = (
        guideline_sd.rename_axis(columns="GuidelineType")
        .stack(dropna=False)
        .rename("SDPct")
        .reset_index()
    )
    final_guidelines = mean_long.merge(
        sd_long, on=["Pathology", "GuidelineType"], validate="one_to_one"
    )
    meta_export = metadata.reset_index()
    return RetrievalData(
        guideline_mean,
        guideline_sd,
        heatmap,
        metadata,
        {
            "panel_a_guideline_seed_level_means": guideline_seed.reset_index(),
            "panel_a_guideline_final_mean_sd": final_guidelines,
            "panel_b_document_seed_level_means": document_seed.merge(
                meta_export, on="DocumentReference", validate="many_to_one"
            ),
            "panel_b_document_final_means": document_mean.merge(
                meta_export, on="DocumentReference", validate="many_to_one"
            ),
            "selected_retrieval_runs": retrieval,
        },
        paths,
        experiments,
        columns,
    )


def create_figure(args, data):
    """Draw one two-panel figure, adapting classes, pathologies and document rows."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch, Rectangle

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    pathologies = list(data.guideline_mean.index)
    labels = list(data.guideline_columns)
    colors = {label: args.pathology_colors.get(label, "#8B8B8B") for label in labels}
    lights = {
        label: args.pathology_light_colors.get(label, "#DCDCDC") for label in labels
    }
    nrows = (len(pathologies) + 1) // 2
    extra_classes = max(0, len(labels) - 4)
    bar_height = nrows * (1.0 + 0.12 * extra_classes)
    label_lines = max(name.count("\n") + 1 for name in data.documents["DisplayName"])
    # Multiline alternative-diagnosis labels need taller cells, not smaller text.
    heat_height = max(2.0, len(data.documents) * (0.15 + 0.09 * (label_lines - 1)))
    height = args.figure_size[1] or max(6.45, bar_height + heat_height + 1.65)
    fig = plt.figure(figsize=(args.figure_size[0], height), facecolor="white")
    grids = fig.add_gridspec(
        2,
        1,
        height_ratios=[bar_height, heat_height],
        # Keep panel separation near 0.9 inches when the document block grows.
        hspace=1.8 / max(height * 0.865 - 0.9, 0.9),
        left=0.35,
        right=0.94,
        top=0.935,
        bottom=0.07,
    )
    bar_grid = grids[0].subgridspec(nrows, 2, wspace=0.25, hspace=0.45)
    bar_axes = []
    errors = data.guideline_sd.fillna(0)
    endpoint = (data.guideline_mean + errors).to_numpy().max()
    xlim = max(80, np.ceil((endpoint + 12) / 10) * 10)
    for index, pathology in enumerate(pathologies):
        ax = fig.add_subplot(bar_grid[index // 2, index % 2])
        bar_axes.append(ax)
        means = data.guideline_mean.loc[pathology].to_numpy()
        sd = errors.loc[pathology].to_numpy()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        bars = ax.barh(
            np.arange(len(labels)),
            means,
            xerr=sd,
            height=0.68,
            color=[
                colors[label] if label == pathology else lights[label]
                for label in labels
            ],
            edgecolor="black",
            linewidth=0.45,
            error_kw={
                "elinewidth": 0.6,
                "capthick": 0.6,
                "capsize": 2.2,
                "ecolor": "#2E2E2E",
            },
        )
        for bar, label, value, deviation in zip(bars, labels, means, sd):
            if label != pathology:
                ax.add_patch(
                    Rectangle(
                        (bar.get_x(), bar.get_y()),
                        bar.get_width(),
                        bar.get_height(),
                        facecolor="none",
                        edgecolor="white",
                        hatch="///",
                        linewidth=0,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (bar.get_x(), bar.get_y()),
                        bar.get_width(),
                        bar.get_height(),
                        facecolor="none",
                        edgecolor="black",
                        linewidth=0.45,
                    )
                )
            ax.text(
                value + deviation + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                fontsize=5.1,
            )
        ax.set(
            xlim=(0, xlim), yticks=np.arange(len(labels)), title=f"{pathology} patients"
        )
        ax.set_xticks(np.arange(0, xlim + 1, 20))
        ax.set_yticklabels(labels if index % 2 == 0 else [""] * len(labels))
        ax.invert_yaxis()
        ax.tick_params(length=0, pad=2, labelsize=5.2)
        if index // 2 < nrows - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Retrieved guideline chunks (%)", labelpad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(
        0.195,
        (bar_axes[0].get_position().y1 + bar_axes[-1].get_position().y0) / 2,
        "Retrieved guideline type",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.5,
    )
    off_target = (
        Patch(facecolor="#C2C2C2", edgecolor="white", hatch="///", linewidth=0),
        Patch(facecolor="none", edgecolor="black", linewidth=0.45),
    )
    fig.legend(
        handles=[
            Patch(facecolor="#666666", edgecolor="black", linewidth=0.45),
            off_target,
        ],
        labels=["Ground-truth pathology", "Off-target pathology"],
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
        loc="upper center",
        bbox_to_anchor=(0.64, 0.995),
        ncol=2,
        frameon=False,
        fontsize=5.7,
    )
    fig.text(0.19, 0.972, "a", fontsize=8, fontweight="bold")
    ax = fig.add_subplot(grids[1])
    heat = data.heatmap.to_numpy()
    vmax = max(10, np.ceil(heat.max() / 5) * 5)
    image = ax.imshow(
        heat,
        cmap=args.heatmap_cmap,
        vmin=0,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(pathologies)))
    ax.set_xticklabels([f"{pathology}\npatients" for pathology in pathologies])
    ax.set_yticks(np.arange(len(data.documents)))
    ax.set_yticklabels(data.documents["DisplayName"], fontsize=5.2)
    ax.tick_params(length=2.5, width=0.6, pad=6)
    ax.set_xticks(np.arange(-0.5, len(pathologies), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.documents), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            ax.text(
                col,
                row,
                f"{heat[row, col]:.1f}",
                ha="center",
                va="center",
                fontsize=5,
                color="white" if heat[row, col] >= 0.55 * vmax else "black",
            )
        label = data.documents.iloc[row]["RetrievedLabel"]
        ax.add_patch(
            Rectangle(
                (-0.6, row - 0.5),
                0.1,
                1,
                facecolor=colors[label],
                edgecolor="none",
                clip_on=False,
            )
        )
    for label in labels:
        rows = np.where(data.documents["RetrievedLabel"].to_numpy() == label)[0]
        if not len(rows):
            continue
        last = rows.max()
        if last < len(data.documents) - 1:
            ax.axhline(last + 0.5, color="black", linewidth=0.75)
        if label in pathologies:
            ax.add_patch(
                Rectangle(
                    (pathologies.index(label) - 0.5, rows.min() - 0.5),
                    1,
                    len(rows),
                    fill=False,
                    edgecolor="black",
                    linewidth=0.85,
                )
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.027, pad=0.025)
    colorbar.set_label("Retrieved chunks (%)", fontsize=5.7)
    colorbar.ax.tick_params(labelsize=5.2, length=2.5, width=0.6)
    colorbar.outline.set_linewidth(0.6)
    ax.legend(
        handles=[
            Patch(facecolor=colors[label], label=label)
            for label in labels
            if label in set(data.documents["RetrievedLabel"])
        ],
        title="Document pathology",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=4,
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=5.2,
        title_fontsize=5.2,
        columnspacing=0.65,
        handlelength=0.9,
        handletextpad=0.3,
    )
    ax.set_title(args.heatmap_title, loc="left", pad=5)
    document_ylabel_x = 0.06 if label_lines > 1 else 0.195
    fig.text(
        document_ylabel_x,
        (ax.get_position().y0 + ax.get_position().y1) / 2,
        "Guideline document",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.5,
    )
    fig.text(
        0.19, ax.get_position().y1 + 0.3 / height, "b", fontsize=8, fontweight="bold"
    )
    return fig


def main(args: DictConfig):
    data = prepare_data(args)
    output = (
        Path(to_absolute_path(str(Path(args.output_dir).expanduser())))
        / args.output_subdir
    )
    if args.render:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with matplotlib.rc_context():
            figure = create_figure(args, data)
            try:
                for fmt in dict.fromkeys(args.formats):
                    folder = output / fmt
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{args.output_name}.{fmt}"
                    # Preserve physical canvas dimensions, as in the notebook.
                    figure.savefig(path, format=fmt, dpi=args.dpi, facecolor="white")
                    print(f"Saved {path}")
                if args.show:
                    plt.show()
            finally:
                plt.close(figure)
    if args.statistics:
        folder = output / "statistics"
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in data.tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
    output.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    for field in ("results_dir", "retrieval_dir", "document_dir", "output_dir"):
        if args[field]:
            resolved[field] = to_absolute_path(str(Path(args[field]).expanduser()))
    resolved.input_files = {key: str(path) for key, path in data.input_files.items()}
    resolved.selected_experiments = data.experiments
    resolved.resolved_guideline_columns = data.guideline_columns
    OmegaConf.save(resolved, output / "retrievals_figure_config.yaml", resolve=True)
    print(
        f"Prepared retrieval composition from {len(data.experiments)} experiments and {len(set(args.seeds))} seeds."
    )
    return data


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs"),
    config_name="figures",
    version_base=None,
)
def run(args: DictConfig):
    main(args)


if __name__ == "__main__":
    run()

"""Render the main manuscript figure and its statistics from inspection CSVs.

Use configs/figures.yaml, figures/<profile>.yaml, and Hydra key=value overrides. Importing this module
performs no plotting or file I/O. See docs/figures/main_figure.md for inputs and examples.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.stats import t as student_t, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


@dataclass
class FigureData:
    models: list[dict]
    pathologies: list[str]
    seeds: list[int]
    overall: pd.DataFrame
    pathology_plot: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    input_files: dict[str, Path]


def seed_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def mean_ci(values, confidence=0.95):
    """Student-t CI across independent backbones after within-backbone seed averaging."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan
    margin = (
        student_t.ppf((1 + confidence) / 2, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return mean, float(mean - margin), float(mean + margin)


def paired_wilcoxon(with_rag, without_rag):
    """Preserve the notebook's one-sided test, with a neutral all-zero convention."""
    if np.array_equal(with_rag, without_rag):
        return 0.0, 1.0
    result = wilcoxon(with_rag, without_rag, alternative="greater", method="auto")
    return float(result.statistic), float(result.pvalue)


def comparison_summary(name, subset, confidence):
    row = {"Group": name, "n backbones": len(subset)}
    for column, label in [
        ("without RAG", "without RAG"),
        ("with RAG", "with RAG"),
        ("Improvement", "Difference"),
    ]:
        mean, lower, upper = mean_ci(subset[column].to_numpy(), confidence)
        row[f"{label} mean" if label != "Difference" else label] = mean
        row[f"{label} CI lower"] = lower
        row[f"{label} CI upper"] = upper
    return row


def compute_statistics(model_df, pathology_model_df, pathologies, confidence):
    overall = comparison_summary("Overall", model_df, confidence)
    w, p = paired_wilcoxon(model_df["with RAG"], model_df["without RAG"])
    overall.update({"Wilcoxon W": w, "Raw P": p, "Confidence Level": confidence})
    if len(model_df) < 2:
        t, t_p = np.nan, np.nan
    elif np.array_equal(model_df["with RAG"], model_df["without RAG"]):
        t, t_p = 0.0, 1.0
    else:
        result = ttest_rel(
            model_df["with RAG"], model_df["without RAG"], alternative="greater"
        )
        t, t_p = float(result.statistic), float(result.pvalue)
    overall.update(
        {
            "Paired t": t,
            "Paired t P": t_p,
            "Paired t df": len(model_df) - 1,
            "Minimum Improvement": float(model_df["Improvement"].min()),
            "Maximum Improvement": float(model_df["Improvement"].max()),
        }
    )
    pathology_rows = []
    for pathology in pathologies:
        subset = pathology_model_df.loc[pathology_model_df["Pathology"].eq(pathology)]
        row = comparison_summary(pathology, subset, confidence)
        row["Pathology"] = row.pop("Group")
        w, p = paired_wilcoxon(subset["with RAG"], subset["without RAG"])
        row.update({"Wilcoxon W": w, "Raw P": p})
        pathology_rows.append(row)
    corrected = multipletests([row["Raw P"] for row in pathology_rows], method="holm")[
        1
    ]
    for row, p in zip(pathology_rows, corrected):
        row["Holm-adjusted P"] = float(p)
    subgroups = [
        comparison_summary(">=30B", model_df.loc[model_df["Size"].ge(30)], confidence),
        comparison_summary(
            "General-purpose", model_df.loc[~model_df["Medical"]], confidence
        ),
        comparison_summary(
            "Medical / clinical", model_df.loc[model_df["Medical"]], confidence
        ),
    ]
    return {
        "model_accuracy": model_df,
        "pathology_model_accuracy": pathology_model_df,
        "overall_statistics": pd.DataFrame([overall]),
        "pathology_statistics": pd.DataFrame(pathology_rows),
        "subgroup_statistics": pd.DataFrame(subgroups),
    }


def configured_model_pairs(args):
    """Link each selected raw CSV model key to its configured display name."""
    if not isinstance(args.models, DictConfig) or not args.models:
        raise ValueError("models must map raw model IDs to name/RAG/size metadata.")
    selected = args.selected_models
    if selected is None:
        selected = list(args.models)
    elif (
        not isinstance(selected, (list, tuple, ListConfig))
        or not selected
        or any(not isinstance(key, str) or not key for key in selected)
    ):
        raise ValueError(
            "selected_models must be a nonempty list of model keys, or null."
        )
    unknown = set(selected) - set(args.models)
    if unknown:
        raise ValueError(f"Unknown selected model keys: {sorted(unknown)}")
    pairs = []
    for key in dict.fromkeys(selected):
        entry = args.models[key]
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(entry, DictConfig)
        ):
            raise ValueError(
                "models must map nonempty CSV model IDs to metadata mappings."
            )
        for field in ("name", "rag", "size", "medical"):
            if field not in entry:
                raise ValueError(f"models[{key!r}] is missing {field!r}.")
        pair = OmegaConf.create(OmegaConf.to_container(entry, resolve=True))
        pair.label = pair.name
        pair.no_rag = key
        pairs.append(pair)
    return pairs


def validate_config(args):
    pairs = configured_model_pairs(args)
    labels = []
    experiments = []
    for pair in pairs:
        for key in ("label", "no_rag", "rag"):
            if not isinstance(pair[key], str) or not pair[key].strip():
                raise ValueError(f"Model {key} must be a nonempty string.")
        if (
            not isinstance(pair.size, (int, float))
            or isinstance(pair.size, bool)
            or not np.isfinite(pair.size)
            or pair.size <= 0
        ):
            raise ValueError("Model size must be a positive number in billions.")
        if type(pair.medical) is not bool:
            raise ValueError("Model medical flags must be true or false.")
        if pair.no_rag == pair.rag:
            raise ValueError(f"{pair.label}: baseline and RAG IDs must differ.")
        labels.append(pair.label)
        experiments.extend([pair.no_rag, pair.rag])
    if len(set(labels)) != len(labels) or len(set(experiments)) != len(experiments):
        raise ValueError("Model labels and experiment IDs must be unique across pairs.")
    if (
        type(args.base_seed) is not int
        or not args.seeds
        or any(type(seed) is not int for seed in args.seeds)
    ):
        raise ValueError(
            "base_seed must be an integer and seeds must be a nonempty integer list."
        )
    if (
        not args.pathologies
        or len(set(args.pathologies)) != len(args.pathologies)
        or any(
            not isinstance(p, str) or not p.strip() or p.lower() == "mean"
            for p in args.pathologies
        )
    ):
        raise ValueError(
            "pathologies must be a nonempty list of unique names, excluding Mean."
        )
    if (
        not isinstance(args.confidence, (int, float))
        or isinstance(args.confidence, bool)
        or not 0 < args.confidence < 1
    ):
        raise ValueError("confidence must be between 0 and 1.")
    if len(args.figure_size) != 2 or any(
        not isinstance(v, (int, float))
        or isinstance(v, bool)
        or not np.isfinite(v)
        or v <= 0
        for v in args.figure_size
    ):
        raise ValueError("figure_size must contain two positive dimensions in inches.")
    if type(args.dpi) is not int or args.dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if (
        type(args.render) is not bool
        or type(args.statistics) is not bool
        or type(args.show) is not bool
    ):
        raise ValueError("render, statistics, and show must be true or false.")
    if not args.render and not args.statistics:
        raise ValueError("Enable render or statistics.")
    if args.render and (
        not args.formats
        or any(fmt not in ("pdf", "svg", "png") for fmt in args.formats)
    ):
        raise ValueError("formats must select pdf, svg, and/or png.")
    if (
        not isinstance(args.output_name, str)
        or not args.output_name
        or Path(args.output_name).name != args.output_name
        or args.output_name in (".", "..")
    ):
        raise ValueError("output_name must be a filename stem.")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("output_dir must be a nonempty path.")
    if type(args.show_composition) is not bool:
        raise ValueError("show_composition must be true or false.")
    if args.render and args.show_composition:
        for group in ("cohort", "guidelines"):
            counts = args[group].counts
            if (
                not counts
                or any(
                    type(row.count) is not int or row.count < 0 or not row.label
                    for row in counts
                )
                or sum(row.count for row in counts) <= 0
            ):
                raise ValueError(
                    f"{group}.counts must contain nonnegative integer counts with a positive total."
                )


def find_accuracy_csv(value, name, pattern):
    """Find one matching CSV directly inside the specified inspection folder."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {name} to the folder containing the inspection CSV.")
    directory = Path(to_absolute_path(str(Path(value).expanduser())))
    if not directory.is_dir():
        raise NotADirectoryError(f"{name}: {directory}")
    if (
        not isinstance(pattern, str)
        or not pattern.strip()
        or "/" in pattern
        or "\\" in pattern
        or pattern in (".", "..")
        or not pattern.lower().endswith(".csv")
    ):
        raise ValueError(
            "csv_pattern must be a CSV filename pattern without directories."
        )
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No CSV matching {pattern!r} in {directory}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Multiple CSVs match {pattern!r} in {directory}: {names}. "
            "Set csv_pattern to select one."
        )
    return matches[0]


def read_accuracy(path, base_seed):
    source = pd.read_csv(path)
    required = {"Model", "Pathology", "Diagnostic Accuracy"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    identifiers = (
        source["Raw Model"].fillna(source["Model"])
        if "Raw Model" in source
        else source["Model"]
    )
    records = []
    for i, identifier in identifiers.items():
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{path}: empty model identifier in row {i}.")
        match = re.search(r"(?:_SEED=|\s+\(seed=)(-?\d+)\)?$", identifier)
        seed = int(match.group(1)) if match else base_seed
        experiment = identifier[: match.start()] if match else identifier
        metadata = source.at[i, "Seed"] if "Seed" in source else np.nan
        if pd.notna(metadata):
            try:
                numeric = float(metadata)
                if not np.isfinite(numeric) or numeric != int(numeric):
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(
                    f"{path}: invalid Seed metadata in row {i}: {metadata!r}"
                )
            if match and seed != int(numeric):
                raise ValueError(
                    f"{path}: seed suffix and Seed metadata disagree in row {i}."
                )
            seed = int(numeric)
        pathology = source.at[i, "Pathology"]
        if not isinstance(pathology, str):
            raise ValueError(f"{path}: empty pathology in row {i}.")
        records.append(
            {
                "Experiment": experiment,
                "Seed": seed,
                "Pathology": pathology.strip().capitalize(),
                "Diagnostic Accuracy": source.at[i, "Diagnostic Accuracy"],
            }
        )
    return pd.DataFrame(records)


def get_seed_values(source, experiment, pathology, seeds):
    selected = source.loc[
        source["Experiment"].eq(experiment)
        & source["Pathology"].eq(pathology)
        & source["Seed"].isin(seeds)
    ]
    if selected["Seed"].duplicated().any():
        raise ValueError(f"Duplicate rows for {experiment!r}, {pathology!r}.")
    values = pd.to_numeric(
        selected.set_index("Seed")["Diagnostic Accuracy"], errors="raise"
    ).reindex(seeds)
    if values.isna().any():
        raise ValueError(
            f"Missing rows for {experiment!r}, {pathology!r}; seeds: {values.index[values.isna()].tolist()}"
        )
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or ((array < 0) | (array > 100)).any():
        raise ValueError(
            f"Accuracy must be finite and between 0 and 100: {experiment!r}, {pathology!r}."
        )
    return array


def prepare_data(args):
    """Read each input once and validate every selected model/pathology/seed pair."""
    validate_config(args)
    overall_path = find_accuracy_csv(
        args.overall_dir or args.results_dir,
        "overall_dir / results_dir",
        args.csv_pattern,
    )
    pathology_path = find_accuracy_csv(
        args.pathology_dir or args.results_dir,
        "pathology_dir / results_dir",
        args.csv_pattern,
    )
    overall_source = read_accuracy(overall_path, args.base_seed)
    pathology_source = (
        overall_source
        if overall_path == pathology_path
        else read_accuracy(pathology_path, args.base_seed)
    )
    seeds = list(dict.fromkeys(args.seeds))
    pathologies = [p.strip().capitalize() for p in args.pathologies]
    models, overall_rows, model_rows, plot_rows, pathology_rows = [], [], [], [], []
    for pair in configured_model_pairs(args):
        models.append(
            {
                "Model": pair.label,
                "Size": pair.size,
                "Medical": pair.medical,
                "Display Label": pair.get("display_label") or pair.label,
                "Model Key": pair.no_rag,
            }
        )
        conditions = {}
        for condition, key in [("no RAG", "no_rag"), ("w RAG", "rag")]:
            experiment = pair[key]
            mean_values = get_seed_values(overall_source, experiment, "Mean", seeds)
            pathology_experiment = pair.get(f"pathology_{key}") or experiment
            rows = [
                get_seed_values(pathology_source, pathology_experiment, p, seeds)
                for p in pathologies
            ]
            reconstructed = np.mean(rows, axis=0)
            if not np.allclose(reconstructed, mean_values, rtol=0, atol=1e-10):
                raise ValueError(
                    f"{pair.label}, {condition}: pathology rows do not reproduce Mean for every seed; check CSVs and model IDs."
                )
            conditions[condition] = mean_values
            for pathology, values in zip(pathologies, rows):
                plot_rows.append(
                    {
                        "Model": pair.label,
                        "Condition": condition,
                        "Pathology": pathology,
                        "Accuracy": float(np.mean(values)),
                        "Seed SD": seed_sd(values),
                        "Medical": pair.medical,
                    }
                )
        no_rag, rag = conditions["no RAG"], conditions["w RAG"]
        overall_rows.append(
            {
                "Model": pair.label,
                "Size": pair.size,
                "Medical": pair.medical,
                "Acc. no RAG": no_rag,
                "Acc. w RAG": rag,
            }
        )
        model_rows.append(
            {
                "Model": pair.label,
                "Model Key": pair.no_rag,
                "Size": pair.size,
                "Medical": pair.medical,
                "without RAG": float(np.mean(no_rag)),
                "with RAG": float(np.mean(rag)),
                "Improvement": float(np.mean(rag - no_rag)),
                "without RAG seed SD": seed_sd(no_rag),
                "with RAG seed SD": seed_sd(rag),
            }
        )
    plot_df = pd.DataFrame(plot_rows)
    for model in models:
        for pathology in pathologies:
            selected = plot_df.loc[
                plot_df["Model"].eq(model["Model"]) & plot_df["Pathology"].eq(pathology)
            ].set_index("Condition")["Accuracy"]
            pathology_rows.append(
                {
                    "Model": model["Model"],
                    "Model Key": model["Model Key"],
                    "Pathology": pathology,
                    "without RAG": selected["no RAG"],
                    "with RAG": selected["w RAG"],
                    "Improvement": selected["w RAG"] - selected["no RAG"],
                }
            )
    model_df = pd.DataFrame(model_rows)
    tables = compute_statistics(
        model_df, pd.DataFrame(pathology_rows), pathologies, args.confidence
    )
    return FigureData(
        models,
        pathologies,
        seeds,
        pd.DataFrame(overall_rows),
        plot_df,
        tables,
        {"overall_csv": overall_path, "pathology_csv": pathology_path},
    )


def set_nature_style():
    """Apply Nature-compatible Matplotlib defaults for editable vector output."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_overall_accuracy(ax, data):
    """Draw paired macro accuracies with sample SD across selected seeds."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = data.overall.copy()
    df["RAG avg"] = df["Acc. w RAG"].apply(np.mean)
    df = df.sort_values(by=["Size", "RAG avg"], ascending=[True, True]).reset_index(
        drop=True
    )
    no_rag_avg = df["Acc. no RAG"].apply(np.mean).to_numpy()
    rag_avg = df["Acc. w RAG"].apply(np.mean).to_numpy()
    no_rag_uncertainty = (
        df["Acc. no RAG"].apply(lambda values: seed_sd(values)).to_numpy()
    )
    rag_uncertainty = df["Acc. w RAG"].apply(lambda values: seed_sd(values)).to_numpy()
    set_nature_style()
    size_palette_dark = {"1B": "#5478B3", "5–10B": "#7A57A8", "30B+": "#B3477A"}
    size_palette_light = {"1B": "#BAC5DE", "5–10B": "#C9B0D9", "30B+": "#E6B0C9"}
    medical_models = {model["Model"] for model in data.models if model["Medical"]}

    def get_size_group(size):
        if size == 1:
            return "1B"
        if size <= 10:
            return "5–10B"
        return "30B+"

    x = np.arange(len(df))
    bar_width = 0.34
    bar_gap = 0.0
    no_rag_x = x - (bar_width + bar_gap) / 2
    rag_x = x + (bar_width + bar_gap) / 2
    size_groups = [get_size_group(size) for size in df["Size"]]
    bar_colors_dark = [size_palette_dark[group] for group in size_groups]
    bar_colors_light = [size_palette_light[group] for group in size_groups]
    ax.set_facecolor("white")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.6)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3,
        width=0.6,
        colors="black",
        labelsize=6,
    )
    error_bar_style = {
        "ecolor": "black",
        "elinewidth": 0.6,
        "capsize": 2,
        "capthick": 0.6,
    }
    ax.bar(
        no_rag_x,
        no_rag_avg,
        width=bar_width,
        yerr=no_rag_uncertainty,
        error_kw=error_bar_style,
        color=bar_colors_light,
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    ax.bar(
        rag_x,
        rag_avg,
        width=bar_width,
        yerr=rag_uncertainty,
        error_kw=error_bar_style,
        color=bar_colors_dark,
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    overall_no_rag_avg = float(np.mean(no_rag_avg))
    overall_rag_avg = float(np.mean(rag_avg))
    value_label_padding = 1.0
    mean_line_clearance = 1.8

    def adjusted_bar_label_y(top_y):
        label_y = top_y + value_label_padding
        while (
            abs(label_y - overall_no_rag_avg) < mean_line_clearance
            or abs(label_y - overall_rag_avg) < mean_line_clearance
        ):
            label_y += mean_line_clearance
        return label_y

    for i in range(len(df)):
        ax.text(
            no_rag_x[i],
            adjusted_bar_label_y(no_rag_avg[i] + no_rag_uncertainty[i]),
            f"{no_rag_avg[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="black",
            zorder=5,
            clip_on=False,
        )
        ax.text(
            rag_x[i],
            adjusted_bar_label_y(rag_avg[i] + rag_uncertainty[i]),
            f"{rag_avg[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="black",
            zorder=5,
            clip_on=False,
        )

    def format_model_label(name):
        display = next(m["Display Label"] for m in data.models if m["Model"] == name)
        if name in medical_models:
            return f"{display}†"
        return display

    model_labels = [format_model_label(name) for name in df["Model"]]
    size_tick_positions = []
    size_tick_labels = []
    previous_size = None
    for idx, size in enumerate(df["Size"]):
        if size != previous_size:
            size_tick_positions.append(x[idx] - 0.5)
            size_tick_labels.append(f"{size}B")
            previous_size = size
    ax.set_xticks(size_tick_positions)
    ax.set_xticklabels(size_tick_labels, fontsize=6, ha="left")
    ax.tick_params(axis="x", length=3, width=0.6, pad=2)
    secax = ax.secondary_xaxis("bottom", functions=(lambda z: z, lambda z: z))
    secax.set_xticks(x)
    secax.set_xticklabels(model_labels, fontsize=5.5)
    secax.spines["bottom"].set_position(("outward", 18))
    secax.tick_params(axis="x", length=3, width=0.6, pad=1, labelrotation=35)
    for label, model_name in zip(secax.get_xticklabels(), df["Model"]):
        label.set_rotation(35)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")
        label.set_linespacing(1.15)
        if model_name in medical_models:
            label.set_fontweight("bold")
    line_x_start = -0.5
    line_x_end = len(df) - 0.55
    mean_label_x = len(df) - 0.47
    ax.hlines(
        overall_no_rag_avg,
        xmin=line_x_start,
        xmax=line_x_end,
        color="black",
        linestyle=(0, (3, 2)),
        linewidth=0.6,
        zorder=1,
    )
    ax.hlines(
        overall_rag_avg,
        xmin=line_x_start,
        xmax=line_x_end,
        color="black",
        linestyle="-",
        linewidth=0.6,
        zorder=1,
    )
    ax.text(
        mean_label_x,
        overall_no_rag_avg,
        f"without RAG\nmean {overall_no_rag_avg:.1f}%",
        ha="left",
        va="center",
        fontsize=5.5,
        color="black",
        clip_on=False,
        zorder=5,
    )
    ax.text(
        mean_label_x,
        overall_rag_avg,
        f"with RAG\nmean {overall_rag_avg:.1f}%",
        ha="left",
        va="center",
        fontsize=5.5,
        color="black",
        clip_on=False,
        zorder=5,
    )
    legend_handles = [
        Patch(
            facecolor=size_palette_light["1B"],
            edgecolor="black",
            linewidth=0.4,
            label="without RAG",
        ),
        Patch(
            facecolor=size_palette_dark["1B"],
            edgecolor="black",
            linewidth=0.4,
            label="with RAG",
        ),
        Patch(
            facecolor=size_palette_dark["1B"],
            edgecolor="black",
            linewidth=0.4,
            label="1B",
        ),
        Patch(
            facecolor=size_palette_dark["5–10B"],
            edgecolor="black",
            linewidth=0.4,
            label="5–10B",
        ),
        Patch(
            facecolor=size_palette_dark["30B+"],
            edgecolor="black",
            linewidth=0.4,
            label="30B+",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="$†$",
            markeredgecolor="black",
            markerfacecolor="black",
            markersize=5,
            label="medical model",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.3,
        handletextpad=0.4,
        borderaxespad=0,
    )
    ax.set_ylabel("Diagnostic accuracy (%)")
    upper_ylim = max(
        100,
        float(np.max(np.r_[no_rag_avg + no_rag_uncertainty, rag_avg + rag_uncertainty]))
        + 4.0,
    )
    ax.set_ylim(0, upper_ylim)
    ax.set_xlim(-0.5, len(df) - 0.02)


def plot_pathology_accuracy(ax, data):
    """Draw the paired pathology distributions and Holm-corrected comparisons."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    MODEL_CONFIG = data.models
    PATHOLOGIES = data.pathologies
    plot_df = data.pathology_plot.copy()
    pvals = (
        data.tables["pathology_statistics"]
        .set_index("Pathology")["Holm-adjusted P"]
        .to_dict()
    )
    pathologies = PATHOLOGIES.copy()
    set_nature_style()
    no_rag_fill = "#BAC5DE"
    rag_color = "#B3477A"
    rag_fill = "#E6B0C9"
    condition_offsets = {"no RAG": -0.15, "w RAG": 0.15}

    def grouped_arrays(dataframe, pathology_order):
        """Create arrays and y positions in no-RAG/RAG order."""
        arrays = []
        positions = []
        styles = []
        for pathology_index, pathology in enumerate(pathology_order):
            for condition in ["no RAG", "w RAG"]:
                values = dataframe.loc[
                    (dataframe["Pathology"] == pathology)
                    & (dataframe["Condition"] == condition),
                    "Accuracy",
                ].to_numpy()
                if len(values) == 0:
                    raise ValueError(
                        f"No plotted values for {pathology!r}, {condition!r}."
                    )
                arrays.append(values)
                positions.append(pathology_index + condition_offsets[condition])
                styles.append(condition)
        return (arrays, positions, styles)

    def p_to_label(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def add_pvalue_bracket(ax, x1, x2, y, cap_height, text):
        """Draw a compact bracket above a pair of vertical boxplots."""
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + cap_height, y + cap_height, y],
            color="black",
            linewidth=0.6,
            clip_on=False,
            zorder=7,
        )
        ax.text(
            (x1 + x2) / 2,
            y + cap_height + 0.45,
            text,
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="black",
            clip_on=False,
            zorder=8,
        )

    pathologies_ext = pathologies + ["Mean"]
    group_cols = ["Model", "Condition", "Medical"]
    avg_df = plot_df.groupby(group_cols, as_index=False).agg(
        Accuracy=("Accuracy", "mean"), **{"Seed SD": ("Seed SD", "mean")}
    )
    avg_df["Pathology"] = "Mean"
    avg_df = avg_df[plot_df.columns]
    plot_df_ext = pd.concat([plot_df, avg_df], ignore_index=True)
    pathology_to_y = {
        pathology: index for (index, pathology) in enumerate(pathologies_ext)
    }
    mean_df_ext = plot_df_ext.groupby(["Pathology", "Condition"], as_index=False).agg(
        mean_acc=("Accuracy", "mean")
    )
    mean_df_ext["y"] = mean_df_ext["Pathology"].map(pathology_to_y)
    mean_df_ext["y"] += mean_df_ext["Condition"].map(condition_offsets)
    (arrays, positions, styles) = grouped_arrays(plot_df_ext, pathologies_ext)
    ax.set_facecolor("white")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.6)
    vp = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.34,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(no_rag_fill if styles[i] == "no RAG" else rag_fill)
        body.set_edgecolor("none")
        body.set_alpha(0.22)
        body.set_zorder(2)
    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.24,
        vert=True,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.6},
        capprops={"color": "black", "linewidth": 0.6},
        manage_ticks=False,
        zorder=3,
    )
    for i, condition in enumerate(styles):
        if condition == "no RAG":
            bp["boxes"][i].set(
                facecolor=no_rag_fill, edgecolor="black", linewidth=0.8, alpha=0.55
            )
        else:
            bp["boxes"][i].set(
                facecolor=rag_fill, edgecolor="black", alpha=0.52, linewidth=0.8
            )
    model_order = [config["Model"] for config in MODEL_CONFIG]
    model_jitter = dict(zip(model_order, np.linspace(-0.035, 0.035, len(model_order))))
    for pathology_index, pathology in enumerate(pathologies_ext):
        paired = (
            plot_df_ext.loc[plot_df_ext["Pathology"].eq(pathology)]
            .pivot(index="Model", columns="Condition", values="Accuracy")
            .reindex(model_order)
        )
        for model_name, row in paired.iterrows():
            jitter = model_jitter[model_name]
            x_no = pathology_index + condition_offsets["no RAG"] + jitter
            x_rag = pathology_index + condition_offsets["w RAG"] + jitter
            ax.plot(
                [x_no, x_rag],
                [row["no RAG"], row["w RAG"]],
                color="#8A8A8A",
                linewidth=0.45,
                alpha=0.55,
                zorder=1,
            )
            ax.scatter(
                x_no,
                row["no RAG"],
                s=14,
                marker="o",
                facecolor=no_rag_fill,
                edgecolor="#4B4B4B",
                linewidth=0.55,
                zorder=4,
            )
            ax.scatter(
                x_rag,
                row["w RAG"],
                s=15,
                marker="s",
                facecolor=rag_color,
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
            )
    mean_df_ext["x"] = mean_df_ext["Pathology"].map(pathology_to_y)
    mean_df_ext["x"] += mean_df_ext["Condition"].map(condition_offsets)
    ax.scatter(
        mean_df_ext["x"],
        mean_df_ext["mean_acc"],
        s=18,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        zorder=6,
    )
    bracket_tops = []
    for i, pathology in enumerate(pathologies_ext):
        category_max = float(
            plot_df_ext.loc[plot_df_ext["Pathology"].eq(pathology), "Accuracy"].max()
        )
        if pathology != "Mean":
            bracket_y = category_max + 2.0
            bracket_tops.append(bracket_y + 2.0)
            add_pvalue_bracket(
                ax=ax,
                x1=i + condition_offsets["no RAG"],
                x2=i + condition_offsets["w RAG"],
                y=bracket_y,
                cap_height=0.9,
                text=p_to_label(pvals[pathology]),
            )
    mean_table = mean_df_ext.pivot(
        index="Condition", columns="Pathology", values="mean_acc"
    ).reindex(index=["no RAG", "w RAG"], columns=pathologies_ext)
    improvement = mean_table.loc["w RAG"] - mean_table.loc["no RAG"]
    table_values = [
        [f"{value:.1f}%" for value in mean_table.loc["no RAG"]],
        [f"{value:.1f}%" for value in mean_table.loc["w RAG"]],
        [f"{value:+.1f} pp" for value in improvement],
    ]
    values_table = ax.table(
        cellText=table_values,
        colLabels=pathologies_ext,
        rowLabels=["without RAG", "with RAG", "Improvement"],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.0, -0.47, 1.0, 0.32],
    )
    values_table.auto_set_font_size(False)
    values_table.set_fontsize(5.5)
    header_fill = "#E6E6E6"
    for (row, col), cell in values_table.get_celld().items():
        cell.set_edgecolor("#8C8C8C")
        cell.set_linewidth(0.4)
        cell.set_facecolor("white")
        cell.get_text().set_color("black")
        if row == 0:
            cell.set_facecolor(header_fill)
            cell.get_text().set_fontweight("bold")
        elif row == 3:
            cell.get_text().set_fontweight("bold")
        if col == -1:
            cell.get_text().set_ha("left")
            cell.get_text().set_fontweight("bold")
            if row == 1:
                cell.set_facecolor(no_rag_fill)
            elif row == 2:
                cell.set_facecolor(rag_fill)
    row_label_width = values_table[1, -1].get_width()
    header_height = values_table[0, 0].get_height()
    metric_header = values_table.add_cell(
        0,
        -1,
        width=row_label_width,
        height=header_height,
        text="Metric",
        loc="right",
        facecolor=header_fill,
        edgecolor="#8C8C8C",
    )
    metric_header.set_linewidth(0.4)
    metric_header.get_text().set_color("black")
    metric_header.get_text().set_fontweight("bold")
    metric_header.get_text().set_fontsize(5.5)
    y_max_data = float(plot_df_ext["Accuracy"].max())
    y_max_annotation = max(bracket_tops, default=y_max_data)
    ax.set_ylim(0, max(100, y_max_annotation + 1.5))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.55, len(pathologies_ext) - 0.45)
    ax.set_xticks(np.arange(len(pathologies_ext)))
    ax.set_xticklabels(pathologies_ext, fontsize=6)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3,
        width=0.6,
        colors="black",
        labelsize=6,
    )
    ax.set_ylabel("Diagnostic accuracy (%)")
    ax.set_xlabel("")
    legend_handles = [
        Patch(
            facecolor=no_rag_fill,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.55,
            label="without RAG box",
        ),
        Patch(
            facecolor=rag_fill,
            edgecolor="black",
            alpha=0.52,
            linewidth=0.8,
            label="with RAG box",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=no_rag_fill,
            markeredgecolor="#4B4B4B",
            markeredgewidth=0.55,
            markersize=4,
            label="model, without RAG",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=rag_color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4,
            label="model, with RAG",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=4,
            label="mean",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0,
    )


def plot_cohort_composition(ax, args):
    """Draw cohort and guideline compositions from explicit configured counts."""
    from matplotlib.patches import FancyBboxPatch

    cohort_composition = [
        (r.label, r.count, r.color) for r in args.cohort.counts if r.count
    ]
    guideline_composition = [
        (r.label, r.count, r.color) for r in args.guidelines.counts if r.count
    ]
    cohort_total = sum((r[1] for r in cohort_composition))
    guideline_total = sum((r[1] for r in guideline_composition))
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    def draw_stack(x0, width, y_top, total_height, gap, total_n, title, composition):
        usable_height = total_height - gap * (len(composition) - 1)
        current_top = y_top
        for label, count, color in composition:
            frac = count / total_n
            height = usable_height * frac
            y = current_top - height
            patch = FancyBboxPatch(
                (x0, y),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.035",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                mutation_aspect=1.0,
                clip_on=False,
                zorder=2,
            )
            ax.add_patch(patch)
            pct = 100 * frac
            fontsize = 5.8 if height >= 0.16 else 5.2
            ax.text(
                x0 + width / 2,
                y + height / 2,
                f"{label}\n{pct:.1f}% (n = {count:,})",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                zorder=3,
                linespacing=0.95,
            )
            current_top = y - gap
        ax.text(
            x0 + width / 2,
            1.015,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.2,
            color="black",
            clip_on=False,
        )

    draw_stack(
        x0=0.04,
        width=0.4,
        y_top=0.91,
        total_height=0.94,
        gap=0.012,
        total_n=cohort_total,
        title=f"{args.cohort.title}\n{cohort_total:,} {args.cohort.unit}",
        composition=cohort_composition,
    )
    draw_stack(
        x0=0.56,
        width=0.4,
        y_top=0.91,
        total_height=0.94,
        gap=0.012,
        total_n=guideline_total,
        title=f"{args.guidelines.title}\n{guideline_total:,} {args.guidelines.unit}",
        composition=guideline_composition,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def create_figure(args, data):
    """Build the original three-panel manuscript layout."""
    import matplotlib.pyplot as plt

    set_nature_style()
    fig = plt.figure(figsize=tuple(args.figure_size))
    gs = fig.add_gridspec(
        nrows=2, ncols=2, height_ratios=[1.55, 0.82], width_ratios=[3.25, 1.65]
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_cohort = fig.add_subplot(gs[1, 1])
    plot_overall_accuracy(ax_top, data)
    plot_pathology_accuracy(ax_bottom, data)
    if args.show_composition:
        plot_cohort_composition(ax_cohort, args)
    else:
        ax_cohort.set_visible(False)
    ax_top.text(
        -0.09,
        1.0,
        "a",
        transform=ax_top.transAxes,
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        ha="left",
        va="top",
        color="black",
    )
    ax_bottom.text(
        -0.15,
        1.1,
        "b",
        transform=ax_bottom.transAxes,
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        ha="left",
        va="top",
        color="black",
    )
    ax_cohort.text(
        -0.12,
        1.1,
        "c",
        transform=ax_cohort.transAxes,
        fontsize=8,
        fontweight="bold",
        fontstyle="normal",
        ha="left",
        va="top",
        color="black",
    )
    fig.subplots_adjust(
        left=0.12, right=0.89, top=0.94, bottom=0.2, hspace=0.42, wspace=0.22
    )
    c_pos = ax_cohort.get_position()
    extra_down = 0.055
    ax_cohort.set_position(
        [c_pos.x0, c_pos.y0 - extra_down, c_pos.width, c_pos.height + extra_down]
    )
    return fig


def statistics_report(data, confidence):
    """Provide the notebook's manuscript summary without IPython display state."""
    percent = 100 * confidence
    lines = [
        f"Seed-averaged accuracy across {len(data.models)} backbones; {percent:g}% Student-t CIs.",
        f"Seeds: {data.seeds}. The independent statistical unit is the backbone.",
    ]
    for name in (
        "overall_statistics",
        "model_accuracy",
        "pathology_statistics",
        "subgroup_statistics",
    ):
        lines.extend(
            [
                "",
                name.replace("_", " ").upper(),
                data.tables[name].to_string(
                    index=False, float_format=lambda v: f"{v:.6g}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Tests: one-sided paired Wilcoxon (with RAG > without RAG); paired t-test sensitivity analysis.",
            "Holm correction covers the configured pathology comparisons.",
            "CIs are undefined for fewer than two backbones. Subgroup CIs may extend beyond 0–100%.",
            "All-zero paired differences use Wilcoxon W=0, P=1.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: DictConfig):
    data = prepare_data(args)
    output = Path(to_absolute_path(str(Path(args.output_dir).expanduser())))
    if args.render:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with matplotlib.rc_context():
            fig = create_figure(args, data)
            try:
                for fmt in dict.fromkeys(args.formats):
                    directory = output / fmt
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / f"{args.output_name}.{fmt}"
                    fig.savefig(
                        destination,
                        format=fmt,
                        dpi=args.dpi,
                        bbox_inches="tight",
                        pad_inches=0.02,
                        facecolor="white",
                    )
                    print(f"Saved {destination}")
                if args.show:
                    plt.show()
            finally:
                plt.close(fig)
    if args.statistics:
        directory = output / "statistics"
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in data.tables.items():
            table.to_csv(directory / f"{name}.csv", index=False)
        report = statistics_report(data, args.confidence)
        (directory / "summary.txt").write_text(report, encoding="utf-8")
        print(report)
    output.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    resolved.overall_dir = str(data.input_files["overall_csv"].parent)
    resolved.pathology_dir = str(data.input_files["pathology_csv"].parent)
    if args.results_dir:
        resolved.results_dir = to_absolute_path(
            str(Path(args.results_dir).expanduser())
        )
    resolved.input_files = {key: str(path) for key, path in data.input_files.items()}
    resolved.output_dir = str(output)
    OmegaConf.save(resolved, output / "main_figure_config.yaml", resolve=True)
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

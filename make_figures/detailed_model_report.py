"""Generate configurable per-model six-panel appendix reports from inspection CSVs.

Use figures=detailed_model_report through configs/figures.yaml. See
docs/figures/detailed_model_report.md. Imports perform no file I/O or rendering.
"""

import re
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle
from omegaconf import DictConfig, ListConfig, OmegaConf

if __package__:
    from .main_figure import configured_model_pairs, find_accuracy_csv
    from .retrievals_figure import nonempty_strings
    from .table_common import split_identifier
else:
    from main_figure import configured_model_pairs, find_accuracy_csv
    from retrievals_figure import nonempty_strings
    from table_common import split_identifier
CONDITIONS = ["without RAG", "with RAG"]
ERROR_KW = {"ecolor": "black", "elinewidth": 0.55, "capsize": 1.8, "capthick": 0.55}


@dataclass
class ReportSettings:
    args: DictConfig
    models: list[dict]
    seeds: list[int]
    pathologies: list[str]
    counts: dict[str, int]
    treatments: dict[str, list[str]]
    labs: dict[str, list[str]]
    exams: list[str]
    imaging_modalities: list[str]
    guideline_columns: dict[str, str]
    guideline_types: list[str]
    treatment_labels: dict[str, str]
    exam_labels: dict[str, str]
    pathology_colors: dict[str, str]
    pathology_light_colors: dict[str, str]
    no_rag_color: str
    rag_color: str
    input_files: dict[str, Path]


def configure_report(args):
    """Resolve configuration once; no module globals are modified between reports."""
    if "report_name" not in args or "guideline_columns" not in args:
        raise ValueError(
            "Select the matching profile with figures=detailed_model_report."
        )
    for field in ("render", "statistics", "show", "individual_pages", "multipage_pdf"):
        if type(args[field]) is not bool:
            raise ValueError(f"{field} must be true or false.")
    if not args.render and not args.statistics:
        raise ValueError("Enable render or statistics.")
    if (
        not isinstance(args.seeds, (list, tuple, ListConfig))
        or not args.seeds
        or type(args.base_seed) is not int
        or any(type(seed) is not int for seed in args.seeds)
    ):
        raise ValueError("base_seed and seeds must be integers; seeds cannot be empty.")
    if args.render:
        if (
            not isinstance(args.formats, (list, tuple, ListConfig))
            or not args.formats
            or any(fmt not in ("pdf", "svg", "png", "jpg") for fmt in args.formats)
        ):
            raise ValueError("formats must select pdf, svg, png and/or jpg.")
        if not args.individual_pages and not (
            args.multipage_pdf and "pdf" in args.formats
        ):
            raise ValueError("Enable individual_pages or a multipage PDF export.")
    if type(args.dpi) is not int or args.dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if (
        not isinstance(args.figure_size, (list, tuple, ListConfig))
        or len(args.figure_size) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
            for value in args.figure_size
        )
    ):
        raise ValueError("figure_size must contain two positive finite dimensions.")
    for field in ("output_subdir", "report_name"):
        value = args[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or Path(value).name != value
            or value in (".", "..")
        ):
            raise ValueError(f"{field} must be a single directory/filename stem.")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("output_dir must be a nonempty path.")
    pairs = configured_model_pairs(args)
    names = [pair.label for pair in pairs]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Selected model names must be nonempty and unique.")
    filenames = [safe_filename(name) for name in names]
    if any(not filename for filename in filenames) or len(set(filenames)) != len(
        filenames
    ):
        raise ValueError(
            "Model names must produce nonempty, distinct output filenames."
        )
    models = [
        {
            "Model": pair.label,
            "No RAG experiment": pair.no_rag,
            "RAG experiment": pair.rag,
        }
        for pair in pairs
    ]
    ids = [
        identifier
        for model in models
        for identifier in (model["No RAG experiment"], model["RAG experiment"])
    ]
    if any(
        not isinstance(identifier, str) or not identifier.strip() for identifier in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError(
            "Selected baseline and RAG IDs must be distinct nonempty strings."
        )
    pathologies = nonempty_strings(args.pathologies, "pathologies")

    def category_mapping(value, label):
        if not isinstance(value, DictConfig):
            raise TypeError(f"{label} must map pathologies to category lists.")
        result = {}
        for pathology in pathologies:
            if pathology not in value:
                raise ValueError(f"{label} is missing {pathology}.")
            result[pathology] = nonempty_strings(
                value[pathology], f"{label}.{pathology}"
            )
        return result

    if (
        not isinstance(args.guideline_columns, DictConfig)
        or not args.guideline_columns
        or any(
            not isinstance(label, str)
            or not label
            or not isinstance(column, str)
            or not column
            for label, column in args.guideline_columns.items()
        )
        or len(set(args.guideline_columns.values())) != len(args.guideline_columns)
    ):
        raise ValueError(
            "guideline_columns must map class labels to distinct CSV columns."
        )

    def display_labels(value):
        if not isinstance(value, DictConfig) or any(
            not isinstance(text, str) for text in value.values()
        ):
            raise ValueError("Display labels must map raw categories to text.")
        return {key: text.replace("\\n", "\n") for key, text in value.items()}

    from matplotlib.colors import is_color_like

    for mapping in (
        args.pathology_colors,
        args.pathology_light_colors,
        args.condition_colors,
    ):
        if not isinstance(mapping, DictConfig) or any(
            not is_color_like(color) for color in mapping.values()
        ):
            raise ValueError("Color mappings must contain valid Matplotlib colors.")
    counts = {item.label: item.count for item in args.cohort.counts}
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError("cohort.counts must contain positive integer counts.")
    sources = [
        "diagnostic_accuracy",
        "retrieval",
        "retrieval_documents",
        "treatment",
        "physical_examination",
        "laboratory",
        "imaging",
    ]
    return ReportSettings(
        args=args,
        models=models,
        seeds=list(dict.fromkeys(args.seeds)),
        pathologies=pathologies,
        counts=counts,
        treatments=category_mapping(args.treatment_categories, "treatment_categories"),
        labs=category_mapping(args.laboratory_categories, "laboratory_categories"),
        exams=nonempty_strings(
            args.physical_examination_categories, "physical_examination_categories"
        ),
        imaging_modalities=nonempty_strings(
            args.imaging_modalities, "imaging_modalities"
        ),
        guideline_columns=dict(args.guideline_columns),
        guideline_types=list(args.guideline_columns),
        treatment_labels=display_labels(args.treatment_labels),
        exam_labels=display_labels(args.physical_examination_labels),
        pathology_colors={
            label: args.pathology_colors.get(label, "#8B8B8B")
            for label in args.guideline_columns
        },
        pathology_light_colors={
            label: args.pathology_light_colors.get(label, "#DCDCDC")
            for label in args.guideline_columns
        },
        no_rag_color=args.condition_colors["without RAG"],
        rag_color=args.condition_colors["with RAG"],
        input_files={
            source: find_accuracy_csv(
                args.input_dirs[source] or args.results_dir,
                f"input_dirs.{source} / results_dir",
                args.csv_patterns[source],
            )
            for source in sources
        },
    )


def set_nature_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 5.5,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
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


def style_axes(ax, percent=True, grid_axis=None):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.6)
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.set_axisbelow(True)
        ax.grid(
            axis=grid_axis, color="#E2E2E2", linewidth=0.45, linestyle="-", zorder=-10
        )
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.6, pad=1.5)
    if percent:
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))


def normalize_seed(value, settings):
    if pd.isna(value) or str(value).strip().lower() in ("", "default", "nan"):
        return settings.args.base_seed
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(f"Invalid seed metadata: {value!r}.")
    return int(numeric)


def add_identity(frame, settings):
    identifiers = (
        frame["Raw Model"].fillna(frame["Model"])
        if "Raw Model" in frame
        else frame["Model"]
    )
    identities = [
        split_identifier(
            identifier,
            normalize_seed(frame.at[index, "Seed"], settings)
            if "Seed" in frame and pd.notna(frame.at[index, "Seed"])
            else np.nan,
            settings.args.base_seed,
        )
        for index, identifier in identifiers.items()
    ]
    frame = frame.copy()
    frame["Experiment"] = [experiment for experiment, _ in identities]
    frame["Seed"] = [seed for _, seed in identities]
    return frame


def require_columns(df, columns, label):
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(sorted(missing))}"
        )


def seed_values(df, value_col, *, settings, **filters):
    selected = df
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if selected.empty:
        raise ValueError(
            "No rows matched the requested seed values:\n"
            + "\n".join(f"  {k} = {v!r}" for (k, v) in filters.items())
        )
    duplicate = selected.loc[selected["Seed"].duplicated(keep=False), "Seed"]
    if not duplicate.empty:
        raise ValueError(
            f"Duplicate rows were found for seeds {sorted(duplicate.astype(str).unique().tolist())}:\n"
            + "\n".join(f"  {k} = {v!r}" for (k, v) in filters.items())
        )
    values = selected.set_index("Seed")[value_col].reindex(settings.seeds)
    if values.isna().any():
        missing_seeds = values.index[values.isna()].tolist()
        raise ValueError(
            f"Missing seeds {missing_seeds} for:\n"
            + "\n".join(f"  {k} = {v!r}" for (k, v) in filters.items())
        )
    values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selected seed values must be finite.")
    return values


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    )


def safe_filename(text):
    return re.sub("[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def select_raw_model_seed_rows(
    source_df,
    *,
    category_column,
    categories_by_pathology,
    value_column,
    label,
    settings,
):
    """Select exact main baseline/RAG rows from one raw analysis CSV.

    The raw request-rate tables use:
      - Base Model = seed-independent experiment name
      - Seed       = blank/default for base_seed, otherwise a numeric seed
      - Model      = run-specific name (may include _SEED=...)

    The returned dataframe adds the concise manuscript model name and the
    condition label used by the plotting functions.
    """
    required = [
        "Model",
        "Base Model",
        "Seed",
        "Pathology",
        category_column,
        value_column,
    ]
    require_columns(source_df, required, label)
    source_df = source_df.copy()
    source_df["Seed label"] = source_df["Seed"].apply(
        lambda value: normalize_seed(value, settings)
    )
    source_df[value_column] = pd.to_numeric(source_df[value_column], errors="raise")
    selected_rows = []
    for config in settings.models:
        experiments = {
            "without RAG": config["No RAG experiment"],
            "with RAG": config["RAG experiment"],
        }
        for condition, experiment_name in experiments.items():
            experiment_df = source_df.loc[
                source_df["Base Model"].eq(experiment_name)
            ].copy()
            if experiment_df.empty:
                raise ValueError(
                    f"No {label.lower()} rows found for exact experiment:\n{experiment_name}"
                )
            for seed in settings.seeds:
                for pathology, categories in categories_by_pathology.items():
                    for category in categories:
                        hit = experiment_df.loc[
                            experiment_df["Seed label"].eq(seed)
                            & experiment_df["Pathology"].eq(pathology)
                            & experiment_df[category_column].eq(category)
                        ]
                        if len(hit) != 1:
                            available = experiment_df.loc[
                                experiment_df["Seed label"].eq(seed)
                                & experiment_df["Pathology"].eq(pathology),
                                [category_column, value_column],
                            ]
                            raise ValueError(
                                f"Expected exactly one {label.lower()} row for:\n  model = {config['Model']!r}\n  condition = {condition!r}\n  experiment = {experiment_name!r}\n  seed = {seed!r}\n  pathology = {pathology!r}\n  {category_column} = {category!r}\nFound {len(hit)} rows.\n\nRows available for this experiment/seed/pathology:\n{available.to_string(index=False)}"
                            )
                        row = hit.iloc[0].to_dict()
                        row["Run Model"] = row["Model"]
                        row["Model"] = config["Model"]
                        row["Model Key"] = config["No RAG experiment"]
                        row["Condition"] = condition
                        row["Seed"] = seed
                        selected_rows.append(row)
    selected_df = pd.DataFrame(selected_rows)
    expected_categories = sum(
        len(categories) for categories in categories_by_pathology.values()
    )
    expected_rows = (
        len(settings.models)
        * len(CONDITIONS)
        * len(settings.seeds)
        * expected_categories
    )
    if len(selected_df) != expected_rows:
        raise ValueError(
            f"Unexpected number of selected {label.lower()} rows: expected {expected_rows}, found {len(selected_df)}."
        )
    if not selected_df[value_column].between(0, 100, inclusive="both").all():
        bad = selected_df.loc[
            ~selected_df[value_column].between(0, 100, inclusive="both")
        ]
        raise ValueError(
            f"At least one selected {label.lower()} percentage lies outside 0-100%:\n"
            + bad.to_string(index=False)
        )
    return selected_df


def load_all_data(*, settings):
    diag_df = pd.read_csv(settings.input_files["diagnostic_accuracy"])
    require_columns(
        diag_df,
        ["Model", "Pathology", "Diagnostic Accuracy"],
        "Diagnostic-accuracy table",
    )
    diag_df = add_identity(diag_df, settings)
    diag_df["Diagnostic Accuracy"] = pd.to_numeric(
        diag_df["Diagnostic Accuracy"], errors="raise"
    )
    retrieval_df = pd.read_csv(settings.input_files["retrieval"])
    require_columns(
        retrieval_df,
        ["Model", "Pathology", "TotalChunks", *settings.guideline_columns.values()],
        "Retrieval metrics table",
    )
    retrieval_df = add_identity(retrieval_df, settings)
    retrieval_df["TotalChunks"] = pd.to_numeric(
        retrieval_df["TotalChunks"], errors="raise"
    )
    for column in settings.guideline_columns.values():
        retrieval_df[column] = pd.to_numeric(retrieval_df[column], errors="raise")
    retrieval_document_df = pd.read_csv(settings.input_files["retrieval_documents"])
    require_columns(
        retrieval_document_df,
        ["Model", "Pathology", "DocumentReference", "RetrievedLabel", "ChunkCount"],
        "Retrieval document-count table",
    )
    retrieval_document_df = add_identity(retrieval_document_df, settings)
    retrieval_document_df["ChunkCount"] = pd.to_numeric(
        retrieval_document_df["ChunkCount"], errors="raise"
    )
    treatment_source_df = pd.read_csv(settings.input_files["treatment"])
    require_columns(
        treatment_source_df,
        [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Treatment",
            "Treatment Cohort Count",
            "Requested Count",
            "Request Percentage",
        ],
        "Treatment table",
    )
    for column in ["Treatment Cohort Count", "Requested Count", "Request Percentage"]:
        treatment_source_df[column] = pd.to_numeric(
            treatment_source_df[column], errors="raise"
        )
    treatment_df = select_raw_model_seed_rows(
        treatment_source_df,
        category_column="Treatment",
        categories_by_pathology=settings.treatments,
        value_column="Request Percentage",
        label="Treatment table",
        settings=settings,
    )
    physical_source_df = pd.read_csv(
        settings.input_files["physical_examination"],
        keep_default_na=False,
        na_values={"Seed": [""]},
    )
    require_columns(
        physical_source_df,
        ["Model", "Base Model", "Seed", "Pathology", "Category", "N", "Percentage"],
        "Physical-examination table",
    )
    physical_source_df["N"] = pd.to_numeric(physical_source_df["N"], errors="raise")
    physical_source_df["Percentage"] = pd.to_numeric(
        physical_source_df["Percentage"], errors="raise"
    )
    physical_categories = OrderedDict(
        (pathology, settings.exams) for pathology in settings.pathologies
    )
    physical_df = select_raw_model_seed_rows(
        physical_source_df,
        category_column="Category",
        categories_by_pathology=physical_categories,
        value_column="Percentage",
        label="Physical-examination table",
        settings=settings,
    )
    lab_source_df = pd.read_csv(settings.input_files["laboratory"])
    require_columns(
        lab_source_df,
        [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Required Laboratory Test",
            "Requested Count",
            "Total Patients",
            "Percentage",
        ],
        "Laboratory table",
    )
    for column in ["Requested Count", "Total Patients", "Percentage"]:
        lab_source_df[column] = pd.to_numeric(lab_source_df[column], errors="raise")
    lab_df = select_raw_model_seed_rows(
        lab_source_df,
        category_column="Required Laboratory Test",
        categories_by_pathology=settings.labs,
        value_column="Percentage",
        label="Laboratory table",
        settings=settings,
    )
    imaging_source_df = pd.read_csv(
        settings.input_files["imaging"], keep_default_na=False, na_values={"Seed": [""]}
    )
    require_columns(
        imaging_source_df,
        [
            "Model",
            "Base Model",
            "Seed",
            "Pathology",
            "Modality",
            "Counts",
            "Total Patients",
            "Percentage",
        ],
        "Imaging table",
    )
    for column in ["Counts", "Total Patients", "Percentage"]:
        imaging_source_df[column] = pd.to_numeric(
            imaging_source_df[column], errors="raise"
        )
    available_modalities = imaging_source_df["Modality"].drop_duplicates().tolist()
    missing_modalities = [
        modality
        for modality in settings.imaging_modalities
        if modality not in available_modalities
    ]
    if missing_modalities:
        raise ValueError(
            "Expected imaging modalities are missing from ImagingPercentages: "
            + ", ".join(missing_modalities)
        )
    imaging_categories = OrderedDict(
        (pathology, settings.imaging_modalities) for pathology in settings.pathologies
    )
    imaging_df = select_raw_model_seed_rows(
        imaging_source_df,
        category_column="Modality",
        categories_by_pathology=imaging_categories,
        value_column="Percentage",
        label="Imaging table",
        settings=settings,
    )
    return {
        "diag": diag_df,
        "retrieval": retrieval_df,
        "retrieval_documents": retrieval_document_df,
        "treatment": treatment_df,
        "physical": physical_df,
        "lab": lab_df,
        "imaging": imaging_df,
        "imaging_modalities": settings.imaging_modalities.copy(),
    }


def add_panel_heading(fig, subplot_spec, letter, title):
    bbox = subplot_spec.get_position(fig)
    y = bbox.y1 + 0.014
    fig.text(
        bbox.x0 - 0.018,
        y,
        letter,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        bbox.x0 + 0.004,
        y,
        title,
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
    )


def pathology_axes(fig, subplot_spec, *, settings, wspace=0.32, hspace=0.55):
    rows = (len(settings.pathologies) + 1) // 2
    gs = subplot_spec.subgridspec(rows, 2, wspace=wspace, hspace=hspace)
    axes = np.array(
        [[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(rows)],
        dtype=object,
    )
    for ax in axes.ravel()[len(settings.pathologies) :]:
        ax.set_visible(False)
    return axes


def add_mini_pathology_title(ax, pathology, *, settings):
    n = settings.counts.get(pathology)
    title = pathology if n is None else f"{pathology} (n={n:,})"
    ax.set_title(title, fontsize=5.2, fontweight="bold", pad=2.5)


def plot_paired_seed_bars(
    ax,
    df,
    model_name,
    pathology,
    categories,
    category_column,
    value_column,
    display_labels=None,
    rotate=25,
    *,
    settings,
):
    x = np.arange(len(categories), dtype=float)
    width = 0.34
    offsets = {"without RAG": -0.18, "with RAG": 0.18}
    colors = {"without RAG": settings.no_rag_color, "with RAG": settings.rag_color}
    for condition in CONDITIONS:
        means = []
        sds = []
        for category in categories:
            values = seed_values(
                df,
                value_column,
                Model=model_name,
                Condition=condition,
                Pathology=pathology,
                **{category_column: category},
                settings=settings,
            )
            (mean, sd) = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        ax.bar(
            x + offsets[condition],
            means,
            width=width,
            yerr=sds,
            error_kw=ERROR_KW,
            color=colors[condition],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    style_axes(ax, percent=True, grid_axis=None)
    ax.set_xlim(-0.55, len(categories) - 0.45)
    ax.set_xticks(x)
    if display_labels is None:
        labels = categories
    else:
        labels = [display_labels.get(category, category) for category in categories]
    ax.set_xticklabels(labels, fontsize=4.6)
    for label in ax.get_xticklabels():
        label.set_rotation(rotate)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right" if rotate else "center")
        label.set_verticalalignment("top")
    add_mini_pathology_title(ax, pathology, settings=settings)


def plot_panel_a(ax, data, config, *, settings):
    diag_df = data["diag"]
    categories = [*settings.pathologies, "Mean"]
    x = np.arange(len(categories), dtype=float)
    width = 0.34
    condition_to_experiment = {
        "without RAG": config["No RAG experiment"],
        "with RAG": config["RAG experiment"],
    }
    colors = {"without RAG": settings.no_rag_color, "with RAG": settings.rag_color}
    offsets = {"without RAG": -0.18, "with RAG": 0.18}
    all_endpoints = []
    for condition in CONDITIONS:
        means = []
        sds = []
        experiment = condition_to_experiment[condition]
        for pathology in categories:
            values = seed_values(
                diag_df,
                "Diagnostic Accuracy",
                Experiment=experiment,
                Pathology=pathology,
                settings=settings,
            )
            (mean, sd) = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        means = np.asarray(means)
        sds = np.asarray(sds)
        all_endpoints.extend((means + sds).tolist())
        bars = ax.bar(
            x + offsets[condition],
            means,
            width=width,
            yerr=sds,
            error_kw=ERROR_KW,
            color=colors[condition],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        for bar, mean, sd in zip(bars, means, sds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + sd + 1.0,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=4.7,
                clip_on=False,
            )
    style_axes(ax, percent=True, grid_axis=None)
    upper = max(105.0, min(115.0, np.ceil((max(all_endpoints) + 5.0) / 5.0) * 5.0))
    ax.set_ylim(0, upper)
    ax.set_ylabel("Diagnostic accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=5.1)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")
    ax.set_xlim(-0.55, len(categories) - 0.45)
    ax.axvline(
        len(settings.pathologies) - 0.5,
        color="#BDBDBD",
        linewidth=0.55,
        linestyle=(0, (2, 2)),
        zorder=1,
    )


def plot_panel_b(fig, subplot_spec, data, config, *, settings):
    retrieval_df = data["retrieval"]
    rag_experiment = config["RAG experiment"]
    axes = pathology_axes(fig, subplot_spec, settings=settings, wspace=0.3, hspace=0.42)
    axes_flat = axes.ravel()
    for ax, true_pathology in zip(axes_flat, settings.pathologies):
        means = []
        sds = []
        for guideline in settings.guideline_types:
            values = (
                seed_values(
                    retrieval_df,
                    settings.guideline_columns[guideline],
                    Experiment=rag_experiment,
                    Pathology=true_pathology,
                    settings=settings,
                )
                * 100.0
            )
            (mean, sd) = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        means = np.asarray(means)
        sds = np.asarray(sds)
        y = np.arange(len(settings.guideline_types))
        colors = [
            settings.pathology_colors[g]
            if g == true_pathology
            else settings.pathology_light_colors[g]
            for g in settings.guideline_types
        ]
        bars = ax.barh(
            y,
            means,
            xerr=sds,
            height=0.66,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            error_kw={
                "ecolor": "#2E2E2E",
                "elinewidth": 0.5,
                "capsize": 1.6,
                "capthick": 0.5,
            },
            zorder=5,
        )
        for bar, guideline in zip(bars, settings.guideline_types):
            if guideline != true_pathology:
                hatch_overlay = Rectangle(
                    (bar.get_x(), bar.get_y()),
                    bar.get_width(),
                    bar.get_height(),
                    facecolor="none",
                    edgecolor="white",
                    hatch="///",
                    linewidth=0.0,
                    zorder=6,
                )
                ax.add_patch(hatch_overlay)
                outline = Rectangle(
                    (bar.get_x(), bar.get_y()),
                    bar.get_width(),
                    bar.get_height(),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=7,
                )
                ax.add_patch(outline)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_yticks(y)
        ax.set_yticklabels(settings.guideline_types, fontsize=4.5)
        ax.invert_yaxis()
        style_axes(ax, percent=False, grid_axis="x")
        ax.tick_params(axis="x", length=0, labelsize=4.6)
        ax.tick_params(axis="y", length=0, labelsize=4.4)
        ax.set_title(f"{true_pathology} patients", fontsize=5.1, pad=2.3)
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    for ax in axes[:-1, :].ravel():
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Retrieved chunks (%)", fontsize=5)
    return axes


def plot_panel_c(fig, subplot_spec, data, model_name, *, settings):
    axes = pathology_axes(
        fig, subplot_spec, settings=settings, wspace=0.34, hspace=0.62
    )
    for ax, (pathology, treatments) in zip(axes.ravel(), settings.treatments.items()):
        plot_paired_seed_bars(
            ax,
            data["treatment"],
            model_name,
            pathology,
            treatments,
            "Treatment",
            "Request Percentage",
            display_labels=settings.treatment_labels,
            rotate=30,
            settings=settings,
        )
    return axes


def plot_panel_d(fig, subplot_spec, data, model_name, *, settings):
    axes = pathology_axes(
        fig, subplot_spec, settings=settings, wspace=0.34, hspace=0.55
    )
    for ax, pathology in zip(axes.ravel(), settings.pathologies):
        plot_paired_seed_bars(
            ax,
            data["physical"],
            model_name,
            pathology,
            settings.exams,
            "Category",
            "Percentage",
            display_labels=settings.exam_labels,
            rotate=20,
            settings=settings,
        )
    return axes


def plot_panel_e(fig, subplot_spec, data, model_name, *, settings):
    axes = pathology_axes(fig, subplot_spec, settings=settings, wspace=0.34, hspace=0.6)
    for ax, (pathology, tests) in zip(axes.ravel(), settings.labs.items()):
        plot_paired_seed_bars(
            ax,
            data["lab"],
            model_name,
            pathology,
            tests,
            "Required Laboratory Test",
            "Percentage",
            display_labels=None,
            rotate=25,
            settings=settings,
        )
    return axes


def plot_panel_f(fig, subplot_spec, data, model_name, *, settings):
    axes = pathology_axes(
        fig, subplot_spec, settings=settings, wspace=0.34, hspace=0.62
    )
    modalities = data["imaging_modalities"]
    for ax, pathology in zip(axes.ravel(), settings.pathologies):
        plot_paired_seed_bars(
            ax,
            data["imaging"],
            model_name,
            pathology,
            modalities,
            "Modality",
            "Percentage",
            display_labels=None,
            rotate=30,
            settings=settings,
        )
    return axes


def label_nested_y_axes(axes, text="Request rate (%)"):
    for ax in axes[:, 0]:
        ax.set_ylabel(text, fontsize=5)
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", labelleft=False)


def build_model_page(data, config, *, settings):
    model_name = config["Model"]
    fig = plt.figure(figsize=settings.args.figure_size)
    outer = fig.add_gridspec(
        3, 2, left=0.075, right=0.985, bottom=0.055, top=0.925, wspace=0.19, hspace=0.27
    )
    ax_a = fig.add_subplot(outer[0, 0])
    plot_panel_a(ax_a, data, config, settings=settings)
    plot_panel_b(fig, outer[0, 1], data, config, settings=settings)
    axes_c = plot_panel_c(fig, outer[1, 0], data, model_name, settings=settings)
    axes_d = plot_panel_d(fig, outer[1, 1], data, model_name, settings=settings)
    axes_e = plot_panel_e(fig, outer[2, 0], data, model_name, settings=settings)
    axes_f = plot_panel_f(fig, outer[2, 1], data, model_name, settings=settings)
    label_nested_y_axes(axes_c, "Request rate (%)")
    label_nested_y_axes(axes_d, "Request rate (%)")
    label_nested_y_axes(axes_e, "Request rate (%)")
    label_nested_y_axes(axes_f, "Request rate (%)")
    add_panel_heading(fig, outer[0, 0], "a", "Diagnostic accuracy")
    add_panel_heading(fig, outer[0, 1], "b", "Retrieval composition")
    add_panel_heading(fig, outer[1, 0], "c", "Treatment requests")
    add_panel_heading(fig, outer[1, 1], "d", "Physical examinations")
    add_panel_heading(fig, outer[2, 0], "e", "Laboratory tests")
    add_panel_heading(fig, outer[2, 1], "f", "Imaging modalities")
    fig.text(
        0.075, 0.982, model_name, ha="left", va="top", fontsize=10, fontweight="bold"
    )
    fig.text(
        0.075,
        0.962,
        f"Detailed per-model report; bars show mean across {len(settings.seeds)} seeds, error bars show ±1 sample SD",
        ha="left",
        va="top",
        fontsize=5.8,
    )
    legend_handles = [
        Patch(
            facecolor=settings.no_rag_color,
            edgecolor="black",
            linewidth=0.45,
            label="without RAG",
        ),
        Patch(
            facecolor=settings.rag_color,
            edgecolor="black",
            linewidth=0.45,
            label="with RAG",
        ),
        Patch(
            facecolor="#EFEFEF",
            edgecolor="black",
            hatch="///",
            linewidth=0.45,
            label="off-target retrieval",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.986),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0,
    )
    return fig


def validate_retrieval(data, config, *, settings):
    retrieval_df = data["retrieval"]
    rag_experiment = config["RAG experiment"]
    selected = retrieval_df.loc[
        retrieval_df["Experiment"].eq(rag_experiment)
        & retrieval_df["Seed"].isin(settings.seeds)
        & retrieval_df["Pathology"].isin(settings.pathologies)
    ].copy()
    expected_index = pd.MultiIndex.from_product(
        [settings.seeds, settings.pathologies], names=["Seed", "Pathology"]
    )
    actual_index = pd.MultiIndex.from_frame(selected[["Seed", "Pathology"]])
    if selected.duplicated(["Seed", "Pathology"], keep=False).any():
        raise ValueError(
            f"Duplicate retrieval rows for main RAG experiment {rag_experiment!r}."
        )
    missing = expected_index.difference(actual_index)
    if len(missing) > 0:
        raise ValueError(
            f"Missing retrieval rows for main RAG experiment {rag_experiment!r}: {missing.tolist()}"
        )
    if (selected["TotalChunks"] <= 0).any():
        raise ValueError(
            f"At least one selected retrieval row has TotalChunks <= 0 for {rag_experiment!r}."
        )
    prop_sums = selected[list(settings.guideline_columns.values())].sum(axis=1)
    if not np.allclose(prop_sums.to_numpy(dtype=float), 1.0, atol=1e-08):
        raise ValueError(
            f"Retrieved guideline proportions do not sum to 1.0 for every selected row of {rag_experiment!r}."
        )
    document_df = data["retrieval_documents"]
    document_selected = document_df.loc[
        document_df["Experiment"].eq(rag_experiment)
        & document_df["Seed"].isin(settings.seeds)
        & document_df["Pathology"].isin(settings.pathologies)
    ].copy()
    document_totals = document_selected.groupby(
        ["Seed", "Pathology"], as_index=False, sort=False
    ).agg(DocumentTotalChunks=("ChunkCount", "sum"))
    metric_totals = selected[["Seed", "Pathology", "TotalChunks"]].copy()
    retrieval_crosscheck = metric_totals.merge(
        document_totals, on=["Seed", "Pathology"], how="left", validate="one_to_one"
    )
    if retrieval_crosscheck["DocumentTotalChunks"].isna().any():
        missing_doc_runs = retrieval_crosscheck.loc[
            retrieval_crosscheck["DocumentTotalChunks"].isna(), ["Seed", "Pathology"]
        ].to_dict("records")
        raise ValueError(
            f"Missing RetrievalDocumentCounts rows for {rag_experiment!r}: {missing_doc_runs}"
        )
    if not np.allclose(
        retrieval_crosscheck["TotalChunks"].to_numpy(dtype=float),
        retrieval_crosscheck["DocumentTotalChunks"].to_numpy(dtype=float),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(
            "RetrievalMetrics TotalChunks does not match summed RetrievalDocumentCounts ChunkCount for:\n"
            + retrieval_crosscheck.to_string(index=False)
        )


def prepare_data(settings):
    """Validate every selected page and calculate the values before export."""
    data = load_all_data(settings=settings)
    all_ids = [
        identifier
        for model in settings.models
        for identifier in (model["No RAG experiment"], model["RAG experiment"])
    ]
    rag_ids = [model["RAG experiment"] for model in settings.models]
    for name in ("diag", "retrieval", "retrieval_documents"):
        frame = data[name]
        ids = all_ids if name == "diag" else rag_ids
        pathologies = (
            [*settings.pathologies, "Mean"] if name == "diag" else settings.pathologies
        )
        data[name] = frame.loc[
            frame["Experiment"].isin(ids)
            & frame["Seed"].isin(settings.seeds)
            & frame["Pathology"].isin(pathologies)
        ].copy()
    numeric = [
        ("diag", ["Diagnostic Accuracy"], 100),
        ("retrieval", list(settings.guideline_columns.values()), 1),
        ("retrieval", ["TotalChunks"], np.inf),
        ("retrieval_documents", ["ChunkCount"], np.inf),
    ]
    for name, columns, upper in numeric:
        values = data[name][columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > upper)).any():
            raise ValueError(
                f"{name}: values must be finite and within their valid range."
            )
    for name, denominator, count, percentage in [
        (
            "treatment",
            "Treatment Cohort Count",
            "Requested Count",
            "Request Percentage",
        ),
        ("lab", "Total Patients", "Requested Count", "Percentage"),
        ("imaging", "Total Patients", "Counts", "Percentage"),
        ("physical", "N", None, "Percentage"),
    ]:
        frame = data[name]
        columns = [denominator] + ([count] if count else [])
        values = frame[columns].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or frame[denominator].le(0).any()
            or not (frame[columns] % 1).eq(0).all().all()
        ):
            raise ValueError(
                f"{name}: patient counts must be finite integers and denominators positive."
            )
        if count and (
            frame[count].gt(frame[denominator]).any()
            or not np.allclose(
                frame[percentage], 100 * frame[count] / frame[denominator], atol=1e-8
            )
        ):
            raise ValueError(
                f"{name}: percentage is inconsistent with requested count / denominator."
            )
    summary_rows = []

    def add_summary(model, panel, condition, pathology, category, values):
        summary_rows.append(
            {
                "Model": model["Model"],
                "Model Key": model["No RAG experiment"],
                "Panel": panel,
                "Condition": condition,
                "Pathology": pathology,
                "Category": category,
                "Mean (%)": float(np.mean(values)),
                "Seed SD (%)": float(np.std(values, ddof=1))
                if len(values) > 1
                else np.nan,
                "N seeds": len(values),
            }
        )

    request_specs = [
        ("treatment", "Treatment", "Request Percentage", settings.treatments),
        (
            "physical",
            "Category",
            "Percentage",
            {pathology: settings.exams for pathology in settings.pathologies},
        ),
        ("lab", "Required Laboratory Test", "Percentage", settings.labs),
        (
            "imaging",
            "Modality",
            "Percentage",
            {
                pathology: settings.imaging_modalities
                for pathology in settings.pathologies
            },
        ),
    ]
    for model in settings.models:
        for condition, identifier in zip(
            CONDITIONS, (model["No RAG experiment"], model["RAG experiment"])
        ):
            for pathology in [*settings.pathologies, "Mean"]:
                values = seed_values(
                    data["diag"],
                    "Diagnostic Accuracy",
                    Experiment=identifier,
                    Pathology=pathology,
                    settings=settings,
                )
                add_summary(
                    model,
                    "diagnostic_accuracy",
                    condition,
                    pathology,
                    "Diagnostic accuracy",
                    values,
                )
            for panel, category_column, value_column, categories in request_specs:
                for pathology, items in categories.items():
                    for category in items:
                        values = seed_values(
                            data[panel],
                            value_column,
                            Model=model["Model"],
                            Condition=condition,
                            Pathology=pathology,
                            settings=settings,
                            **{category_column: category},
                        )
                        add_summary(
                            model, panel, condition, pathology, category, values
                        )
        validate_retrieval(data, model, settings=settings)
        for pathology in settings.pathologies:
            for label, column in settings.guideline_columns.items():
                values = (
                    seed_values(
                        data["retrieval"],
                        column,
                        Experiment=model["RAG experiment"],
                        Pathology=pathology,
                        settings=settings,
                    )
                    * 100
                )
                add_summary(model, "retrieval", "with RAG", pathology, label, values)
    data["panel_summary"] = pd.DataFrame(summary_rows)
    return data


def main(args: DictConfig):
    settings = configure_report(args)
    data = prepare_data(settings)
    output = (
        Path(to_absolute_path(str(Path(args.output_dir).expanduser())))
        / args.output_subdir
    )
    if args.render:
        if not args.show:
            mpl.use("Agg")
        multipage = args.multipage_pdf and "pdf" in args.formats
        if multipage:
            output.mkdir(parents=True, exist_ok=True)
        context = (
            PdfPages(output / f"{args.report_name}.pdf")
            if multipage
            else nullcontext(None)
        )
        with mpl.rc_context(), context as pdf:
            set_nature_style()
            for model in settings.models:
                print(f"Building report for {model['Model']} ...")
                figure = build_model_page(data, model, settings=settings)
                try:
                    if pdf is not None:
                        pdf.savefig(figure, facecolor="white")
                    if args.individual_pages:
                        for fmt in dict.fromkeys(args.formats):
                            folder = output / fmt
                            folder.mkdir(parents=True, exist_ok=True)
                            figure.savefig(
                                folder
                                / f"{safe_filename(model['Model'])}_detailed_report.{fmt}",
                                format=fmt,
                                dpi=args.dpi,
                                facecolor="white",
                            )
                    if args.show:
                        plt.show()
                finally:
                    plt.close(figure)
    if args.statistics:
        folder = output / "statistics"
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in data.items():
            if isinstance(frame, pd.DataFrame):
                frame.to_csv(folder / f"{name}.csv", index=False)
    output.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    resolved.input_files = {
        source: str(path) for source, path in settings.input_files.items()
    }
    resolved.selected_model_keys = [
        model["No RAG experiment"] for model in settings.models
    ]
    for field in ("results_dir", "output_dir"):
        if args[field]:
            resolved[field] = to_absolute_path(str(Path(args[field]).expanduser()))
    for key, value in args.input_dirs.items():
        if value:
            resolved.input_dirs[key] = to_absolute_path(str(Path(value).expanduser()))
    OmegaConf.save(resolved, output / "detailed_model_report_config.yaml", resolve=True)
    print(f"Saved reports for {len(settings.models)} models to {output}")
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

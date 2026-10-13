"""Standalone laboratory, imaging, physical-exam and treatment request figures.

Use figures=figures_orders_tests_treatments with configs/figures.yaml. See
docs/figures/figures_orders_tests_treatments.md. Imports perform no I/O or rendering.
"""

from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
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
DOCTORS = "MIMIC Doctors"
PLOT_SOURCES = {
    "laboratory": "laboratory",
    "imaging_patients": "imaging",
    "imaging_stacked_percentages": "imaging",
    "imaging_stacked_counts": "imaging",
    "physical_examination": "physical_examination",
    "treatment_all_diagnoses": "treatment_all_diagnoses",
    "treatment_correct_diagnosis": "treatment_correct_diagnosis",
}
SOURCE_COLUMNS = {
    "laboratory": (
        "Required Laboratory Test",
        "Percentage",
        "Requested Count",
        "Total Patients",
    ),
    "imaging": ("Modality", "Percentage", "Counts", "Total Patients"),
    "physical_examination": ("Category", "Percentage", None, "N"),
    "treatment_all_diagnoses": (
        "Treatment",
        "Request Percentage",
        "Requested Count",
        "Treatment Cohort Count",
    ),
    "treatment_correct_diagnosis": (
        "Treatment",
        "Request Percentage",
        "Requested Count",
        "Correctly Diagnosed Count",
    ),
}
RUN_KEYS = ["Experiment", "Seed", "Pathology"]
GROUP_KEYS = ["Condition", "Pathology", "Category"]


@dataclass
class RequestData:
    categories: dict[str, list[str]]
    model_order: list[str]
    tables: dict[str, pd.DataFrame]
    input_files: dict[str, Path]


def validate_config(args):
    if "csv_patterns" not in args or "laboratory_categories" not in args:
        raise ValueError("Select figures=figures_orders_tests_treatments.")
    plots = nonempty_strings(args.plots, "plots")
    unknown = set(plots) - set(PLOT_SOURCES)
    if unknown:
        raise ValueError(f"Unknown plots: {sorted(unknown)}.")
    for field in (
        "render",
        "statistics",
        "show",
        "include_doctors",
        "show_values_table",
        "show_violins",
    ):
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
    nonempty_strings(args.pathologies, "pathologies")
    if args.render and (
        not isinstance(args.formats, (list, tuple, ListConfig))
        or not args.formats
        or any(fmt not in ("pdf", "svg", "png", "jpg") for fmt in args.formats)
    ):
        raise ValueError("formats must select pdf, svg, png and/or jpg.")
    for field in ("dpi", "panel_columns"):
        if type(args[field]) is not int or args[field] <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    for field in ("figure_width", "panel_height", "stacked_height"):
        value = args[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{field} must be a positive finite number.")
    for field in ("output_subdir", "output_prefix"):
        value = args[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or Path(value).name != value
            or value in (".", "..")
        ):
            raise ValueError(f"{field} must be a single directory/filename name.")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("output_dir must be a nonempty path.")
    if not isinstance(args.doctor_model, str) or not args.doctor_model:
        raise ValueError("doctor_model must be a nonempty CSV model ID.")
    pairs = configured_model_pairs(args)
    names = [pair.label for pair in pairs]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Selected model names must be nonempty and unique.")
    ids = [identifier for pair in pairs for identifier in (pair.no_rag, pair.rag)]
    if len(set(ids)) != len(ids) or args.doctor_model in ids:
        raise ValueError("Baseline, RAG and doctor CSV IDs must be distinct.")
    return plots, pairs


def cohort_counts(args):
    counts = {item.label: item.count for item in args.cohort.counts}
    for pathology in args.pathologies:
        if (
            pathology not in counts
            or type(counts[pathology]) is not int
            or counts[pathology] <= 0
        ):
            raise ValueError(
                f"cohort.counts needs a positive integer count for {pathology}."
            )
    return counts


def read_source(path, args):
    # Preserve categorical strings such as imaging modality 'None'.
    frame = pd.read_csv(path, keep_default_na=False, na_values={"Seed": [""]})
    missing = {"Model", "Base Model", "Seed", "Pathology"} - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}.")
    identities = []
    for identifier, seed in zip(frame["Base Model"], frame["Seed"]):
        metadata = (
            np.nan
            if pd.isna(seed) or str(seed).strip().lower() in ("", "default", "nan")
            else seed
        )
        identities.append(split_identifier(identifier, metadata, args.base_seed))
    frame["Experiment"] = [identifier for identifier, _ in identities]
    frame["Seed"] = [seed for _, seed in identities]
    return frame


def categories_for(args, source):
    if source == "laboratory":
        mapping = args.laboratory_categories
    elif source.startswith("treatment_"):
        mapping = args.treatment_categories
    else:
        values = (
            args.imaging_modalities
            if source == "imaging"
            else args.physical_examination_categories
        )
        mapping = {pathology: values for pathology in args.pathologies}
    result = {}
    for pathology in dict.fromkeys(args.pathologies):
        if pathology not in mapping:
            raise ValueError(f"Missing {source} categories for {pathology}.")
        result[pathology] = nonempty_strings(
            mapping[pathology], f"{source} categories for {pathology}"
        )
    return result


def select_grid(frame, args, pairs, categories, doctors=False):
    definitions = pd.DataFrame(
        [
            {
                "Experiment": identifier,
                "Model": pair.label,
                "Model Key": pair.no_rag,
                "Condition": condition,
            }
            for pair in pairs
            for condition, identifier in zip(CONDITIONS, (pair.no_rag, pair.rag))
        ]
    )
    category_grid = pd.DataFrame(
        [
            {"Pathology": pathology, "Category": category}
            for pathology, values in categories.items()
            for category in values
        ]
    )
    if doctors:
        selected = frame.loc[frame["Experiment"].eq(args.doctor_model)].copy()
        keys = ["Pathology", "Category"]
        selected["Condition"] = DOCTORS
        expected = category_grid
    else:
        selected = frame.loc[
            frame["Experiment"].isin(definitions["Experiment"])
            & frame["Seed"].isin(args.seeds)
        ].copy()
        selected = selected.drop(columns=["Model"], errors="ignore").merge(
            definitions, on="Experiment", validate="many_to_one"
        )
        keys = [*RUN_KEYS, "Category"]
        expected = (
            definitions[["Experiment"]]
            .merge(pd.DataFrame({"Seed": list(dict.fromkeys(args.seeds))}), how="cross")
            .merge(category_grid, how="cross")
        )
    selected = selected.merge(
        category_grid, on=["Pathology", "Category"], how="inner", validate="many_to_one"
    )
    if selected.duplicated(keys).any():
        raise ValueError(
            f"Duplicate selected {'doctor' if doctors else 'model'} rows for {keys}."
        )
    indexed = selected.set_index(keys)
    missing = pd.MultiIndex.from_frame(expected[keys]).difference(indexed.index)
    if len(missing):
        raise ValueError(
            f"Missing selected {'doctor' if doctors else 'model'} rows: {missing.tolist()}."
        )
    return selected.reset_index(drop=True)


def check_values(frame, source):
    columns = ["Percentage", "Denominator Count", "Requested Count"]
    if source == "treatment_correct_diagnosis":
        columns.append("Treatment Cohort Count")
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{source}: values must be finite and nonnegative.")
    if frame["Percentage"].gt(100).any() or frame["Denominator Count"].le(0).any():
        raise ValueError(
            f"{source}: percentages must be 0–100 and denominators positive."
        )
    if frame["Requested Count"].gt(frame["Denominator Count"]).any():
        raise ValueError(f"{source}: requested count exceeds denominator.")
    integer_columns = ["Denominator Count"]
    if source != "physical_examination":
        integer_columns.append("Requested Count")
    if source == "treatment_correct_diagnosis":
        integer_columns.append("Treatment Cohort Count")
        if frame["Denominator Count"].gt(frame["Treatment Cohort Count"]).any():
            raise ValueError(
                "Correctly diagnosed count exceeds treatment cohort count."
            )
    if not (frame[integer_columns] % 1).eq(0).all().all():
        raise ValueError(f"{source}: patient counts must be integers.")
    if source != "physical_examination" and not np.allclose(
        frame["Percentage"],
        100 * frame["Requested Count"] / frame["Denominator Count"],
        atol=1e-8,
    ):
        raise ValueError(f"{source}: percentage disagrees with count / denominator.")


def aggregate(selected, doctors):
    """Keep backbone estimates for boxplots and separate seed estimates for stacks."""
    per_model = selected.groupby(
        ["Model", "Model Key", *GROUP_KEYS], as_index=False, sort=False
    ).agg(
        Mean_percentage_across_seeds=("Percentage", "mean"),
        Seed_SD=("Percentage", "std"),
        Mean_requested_count_across_seeds=("Requested Count", "mean"),
        Mean_denominator_N_across_seeds=("Denominator Count", "mean"),
        Min_denominator_N_across_seeds=("Denominator Count", "min"),
        Max_denominator_N_across_seeds=("Denominator Count", "max"),
        N_seeds=("Seed", "nunique"),
    )
    summary = per_model.groupby(GROUP_KEYS, as_index=False, sort=False).agg(
        Mean_across_models=("Mean_percentage_across_seeds", "mean"),
        SD_across_models=("Mean_percentage_across_seeds", "std"),
        Median_across_models=("Mean_percentage_across_seeds", "median"),
        Q1_across_models=(
            "Mean_percentage_across_seeds",
            lambda values: values.quantile(0.25),
        ),
        Q3_across_models=(
            "Mean_percentage_across_seeds",
            lambda values: values.quantile(0.75),
        ),
        N_models=("Model", "nunique"),
    )
    per_seed = selected.groupby(["Seed", *GROUP_KEYS], as_index=False, sort=False).agg(
        Mean_percentage_across_models=("Percentage", "mean"),
        Mean_count_across_models=("Requested Count", "mean"),
    )
    ranges = selected.groupby(GROUP_KEYS, as_index=False, sort=False).agg(
        Minimum_source_denominator=("Denominator Count", "min"),
        Maximum_source_denominator=("Denominator Count", "max"),
        Mean_source_denominator=("Denominator Count", "mean"),
        Number_source_rows=("Denominator Count", "size"),
    )
    improvements = summary.pivot(
        index=["Pathology", "Category"],
        columns="Condition",
        values="Mean_across_models",
    )
    improvements["Improvement (pp)"] = (
        improvements["with RAG"] - improvements["without RAG"]
    )
    if not doctors.empty:
        improvements[DOCTORS] = doctors.set_index(["Pathology", "Category"])[
            "Percentage"
        ]
    return {
        "model_seed_values": selected,
        "model_seed_averages": per_model,
        "model_summary": summary.merge(ranges, on=GROUP_KEYS, validate="one_to_one"),
        "seed_level_means": per_seed,
        "denominator_ranges": ranges,
        "mimic_doctors": doctors,
        "mean_improvements": improvements.reset_index(),
    }


def prepare_source(args, pairs, source, path):
    categories = categories_for(args, source)
    frame = read_source(path, args)
    category, percentage, count, denominator = SOURCE_COLUMNS[source]
    required = {category, percentage, denominator}
    if count:
        required.add(count)
    if source == "treatment_correct_diagnosis":
        required.add("Treatment Cohort Count")
    if required - set(frame):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame))}.")
    frame = frame.rename(
        columns={
            category: "Category",
            percentage: "Percentage",
            denominator: "Denominator Count",
        }
    )
    if count:
        frame = frame.rename(columns={count: "Requested Count"})
    else:
        frame["Requested Count"] = (
            pd.to_numeric(frame["Percentage"], errors="raise")
            * pd.to_numeric(frame["Denominator Count"], errors="raise")
            / 100
        )
    selected = select_grid(frame, args, pairs, categories)
    doctors = (
        select_grid(frame, args, pairs, categories, doctors=True)
        if args.include_doctors and source in ("laboratory", "imaging")
        else pd.DataFrame()
    )
    check_values(selected, source)
    if not doctors.empty:
        check_values(doctors, source)
    tables = aggregate(selected, doctors)
    if source == "physical_examination":
        nominal = selected["Pathology"].map(cohort_counts(args))
        if selected["Denominator Count"].gt(nominal).any():
            raise ValueError(
                "Physical-examination denominator exceeds configured cohort size."
            )
        reduced = selected.loc[selected["Denominator Count"].lt(nominal)].copy()
        reduced["Nominal N"] = nominal.loc[reduced.index]
        reduced["Missing cases"] = reduced["Nominal N"] - reduced["Denominator Count"]
        tables["reduced_denominators"] = reduced
        tables["source_mean_rows"] = frame.loc[
            frame["Pathology"].eq("Mean")
            & frame["Seed"].isin(args.seeds)
            & frame["Experiment"].isin(selected["Experiment"])
        ].copy()
    return RequestData(
        categories, [pair.label for pair in pairs], tables, {source: path}
    )


def prepare_imaging_patients(args, pairs, imaging, path):
    """Count each patient once per modality; the denominator is the full cohort."""
    frame = read_source(path, args)
    if {"Patient ID", "Modality"} - set(frame):
        raise ValueError(f"{path}: Patient ID and Modality columns are required.")
    categories = imaging.categories
    counts_by_pathology = cohort_counts(args)
    template = imaging.tables["model_seed_values"].copy()
    raw_selected = frame.loc[
        frame["Experiment"].isin(template["Experiment"])
        & frame["Seed"].isin(args.seeds)
        & frame["Pathology"].isin(categories)
    ]
    raw_doctors = frame.loc[
        frame["Experiment"].eq(args.doctor_model) & frame["Pathology"].isin(categories)
    ]
    for records, keys in [(raw_selected, RUN_KEYS)] + (
        [(raw_doctors, ["Pathology"])] if args.include_doctors else []
    ):
        if (
            records["Patient ID"].isna().any()
            or records["Patient ID"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Imaging requests need nonempty Patient ID values.")
        patient_totals = records.groupby(keys)["Patient ID"].nunique()
        pathology_index = patient_totals.index.get_level_values("Pathology")
        if (
            patient_totals.to_numpy()
            > pathology_index.map(counts_by_pathology).to_numpy()
        ).any():
            raise ValueError(
                "Imaging patient IDs exceed the configured full cohort size."
            )

    def patient_counts(rows, keys):
        return (
            rows.groupby([*keys, "Modality"])["Patient ID"]
            .nunique()
            .rename("Requested Count")
            .reset_index()
            .rename(columns={"Modality": "Category"})
        )

    counts = patient_counts(raw_selected, RUN_KEYS)
    template = template.drop(
        columns=["Requested Count", "Percentage", "Denominator Count"]
    ).merge(counts, on=[*RUN_KEYS, "Category"], how="left", validate="one_to_one")
    template["Requested Count"] = template["Requested Count"].fillna(0)
    template["Denominator Count"] = template["Pathology"].map(counts_by_pathology)
    template["Percentage"] = (
        100 * template["Requested Count"] / template["Denominator Count"]
    )
    doctors = imaging.tables["mimic_doctors"].copy()
    if not doctors.empty:
        counts = patient_counts(raw_doctors, ["Pathology"])
        doctors = doctors.drop(
            columns=["Requested Count", "Percentage", "Denominator Count"]
        ).merge(counts, on=["Pathology", "Category"], how="left", validate="one_to_one")
        doctors["Requested Count"] = doctors["Requested Count"].fillna(0)
        doctors["Denominator Count"] = doctors["Pathology"].map(counts_by_pathology)
        doctors["Percentage"] = (
            100 * doctors["Requested Count"] / doctors["Denominator Count"]
        )
        check_values(doctors, "imaging_patients")
    check_values(template, "imaging_patients")
    tables = aggregate(template, doctors)
    tables["selected_imaging_requests"] = raw_selected
    return RequestData(
        categories,
        imaging.model_order,
        tables,
        {**imaging.input_files, "imaging_requests": path},
    )


def add_stack_tables(data):
    per_seed = data.tables["seed_level_means"]
    summary = per_seed.groupby(GROUP_KEYS, as_index=False, sort=False).agg(
        Mean_percentage=("Mean_percentage_across_models", "mean"),
        Seed_SD_percentage=("Mean_percentage_across_models", "std"),
        Mean_count=("Mean_count_across_models", "mean"),
        Seed_SD_count=("Mean_count_across_models", "std"),
    )
    doctors = data.tables["mimic_doctors"]
    if not doctors.empty:
        doctor_summary = doctors[GROUP_KEYS + ["Percentage", "Requested Count"]].rename(
            columns={"Percentage": "Mean_percentage", "Requested Count": "Mean_count"}
        )
        summary = pd.concat([summary, doctor_summary], ignore_index=True)
    totals = per_seed.groupby(
        ["Condition", "Seed", "Pathology"], as_index=False, sort=False
    )[["Mean_percentage_across_models", "Mean_count_across_models"]].sum()
    total_summary = totals.groupby(
        ["Condition", "Pathology"], as_index=False, sort=False
    ).agg(
        Total_mean_percentage=("Mean_percentage_across_models", "mean"),
        Total_seed_SD_percentage=("Mean_percentage_across_models", "std"),
        Total_mean_count=("Mean_count_across_models", "mean"),
        Total_seed_SD_count=("Mean_count_across_models", "std"),
    )
    data.tables.update(
        stack_summary=summary,
        stack_seed_totals=totals,
        stack_total_summary=total_summary,
    )


def prepare_data(args):
    plots, pairs = validate_config(args)
    sources = list(dict.fromkeys(PLOT_SOURCES[plot] for plot in plots))
    prepared = {}
    for source in sources:
        path = find_accuracy_csv(
            args.input_dirs[source] or args.results_dir,
            f"input_dirs.{source} / results_dir",
            args.csv_patterns[source],
        )
        prepared[source] = prepare_source(args, pairs, source, path)
    if "imaging_patients" in plots:
        path = find_accuracy_csv(
            args.input_dirs.imaging_requests or args.results_dir,
            "input_dirs.imaging_requests / results_dir",
            args.csv_patterns.imaging_requests,
        )
        prepared["imaging_patients"] = prepare_imaging_patients(
            args, pairs, prepared["imaging"], path
        )
    if any(plot.startswith("imaging_stacked_") for plot in plots):
        add_stack_tables(prepared["imaging"])
    return {
        plot: prepared[
            "imaging_patients" if plot == "imaging_patients" else PLOT_SOURCES[plot]
        ]
        for plot in plots
    }


def set_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.5,
        }
    )


def render_distribution(args, data, plot):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    set_style()
    columns = min(args.panel_columns, len(data.categories))
    rows = (len(data.categories) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(args.figure_width, args.panel_height * rows),
        squeeze=False,
    )
    model_values = data.tables["model_seed_averages"]
    doctors = data.tables["mimic_doctors"]
    offsets = [-0.17, 0.10] if not doctors.empty else [-0.15, 0.15]
    jitter = dict(
        zip(data.model_order, np.linspace(-0.03, 0.03, len(data.model_order)))
    )
    for index, (pathology, categories) in enumerate(data.categories.items()):
        ax = axes.flat[index]
        for category_index, category in enumerate(categories):
            paired = []
            for condition, offset in zip(CONDITIONS, offsets):
                values = (
                    model_values.loc[
                        model_values["Condition"].eq(condition)
                        & model_values["Pathology"].eq(pathology)
                        & model_values["Category"].eq(category)
                    ]
                    .set_index("Model")
                    .reindex(data.model_order)["Mean_percentage_across_seeds"]
                    .to_numpy()
                )
                paired.append(values)
                position = category_index + offset
                if args.show_violins and len(values) > 1 and np.ptp(values) > 1e-12:
                    violin = ax.violinplot(
                        [values], positions=[position], widths=0.3, showextrema=False
                    )
                    violin["bodies"][0].set(
                        facecolor=args.condition_colors[condition],
                        edgecolor="none",
                        alpha=0.22,
                    )
                box = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=0.2,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                box["boxes"][0].set(
                    facecolor=args.condition_colors[condition],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.55,
                )
                for part in ("medians", "whiskers", "caps"):
                    for line in box[part]:
                        line.set(color="black", linewidth=0.6)
                ax.scatter(
                    position,
                    np.mean(values),
                    marker="o",
                    s=18,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=7,
                )
            for model_index, model in enumerate(data.model_order):
                xs = [category_index + offset + jitter[model] for offset in offsets]
                ys = [values[model_index] for values in paired]
                ax.plot(xs, ys, color="#8A8A8A", linewidth=0.45, alpha=0.55, zorder=1)
                for condition, x, y, marker in zip(CONDITIONS, xs, ys, ("o", "s")):
                    ax.scatter(
                        x,
                        y,
                        marker=marker,
                        s=14,
                        facecolor=args.point_colors[condition],
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=5,
                    )
            if not doctors.empty:
                value = doctors.loc[
                    doctors["Pathology"].eq(pathology)
                    & doctors["Category"].eq(category),
                    "Percentage",
                ].iloc[0]
                ax.scatter(
                    category_index + 0.36,
                    value,
                    marker="D",
                    s=23,
                    facecolor=args.condition_colors[DOCTORS],
                    edgecolor="black",
                    linewidth=0.55,
                    zorder=6,
                )
        names = [
            args.category_labels.get(category, category).replace("\\n", "\n")
            for category in categories
        ]
        ax.set(
            xticks=np.arange(len(categories)),
            ylim=(0, 105),
            yticks=np.arange(0, 101, 20),
            title=pathology,
            xlim=(-0.5, len(categories) - 0.4),
        )
        ax.set_xticklabels(names, rotation=25, ha="right", rotation_mode="anchor")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(
            -0.10,
            1.04,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
        )
        if args.show_values_table:
            improvements = (
                data.tables["mean_improvements"]
                .set_index(["Pathology", "Category"])
                .loc[pathology]
                .reindex(categories)
            )
            table_columns = [*CONDITIONS, "Improvement (pp)"] + (
                [DOCTORS] if not doctors.empty else []
            )
            text = [
                [
                    f"{value:+.1f} pp" if key == "Improvement (pp)" else f"{value:.1f}%"
                    for value in improvements[key]
                ]
                for key in table_columns
            ]
            table = ax.table(
                cellText=text,
                rowLabels=table_columns,
                colLabels=names,
                cellLoc="center",
                rowLoc="right",
                bbox=[0, -0.60, 1, 0.36],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(4.8)
            for (row, col), cell in table.get_celld().items():
                cell.set(
                    edgecolor="#8C8C8C",
                    linewidth=0.4,
                    facecolor="#E6E6E6" if row == 0 else "white",
                )
                if row == 3:
                    cell.set_facecolor("#FFF3C4")
        ax.tick_params(length=3, width=0.6)
    for ax in list(axes.flat)[len(data.categories) :]:
        ax.set_visible(False)
    ylabel = {
        "laboratory": "Patients with laboratory test requested (%)",
        "imaging_patients": "Patients with imaging modality requested (%)",
        "physical_examination": "Patients with physical examination requested (%)",
        "treatment_all_diagnoses": "Treatment requested, all diagnoses (%)",
        "treatment_correct_diagnosis": "Treatment requested, correct diagnosis (%)",
    }[plot]
    fig.supylabel(ylabel, x=0.02, fontsize=7)
    handles = [
        Patch(
            facecolor=args.condition_colors[condition],
            edgecolor="black",
            label=condition,
        )
        for condition in CONDITIONS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label="mean across models",
        )
    )
    if not doctors.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor=args.condition_colors[DOCTORS],
                markeredgecolor="black",
                label=DOCTORS,
            )
        )
    fig.legend(
        handles=handles, loc="upper center", ncol=min(4, len(handles)), frameon=False
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        top=0.91,
        bottom=0.24 if args.show_values_table else 0.12,
        hspace=1.35 if args.show_values_table else 0.55,
        wspace=0.55,
    )
    return fig


def render_stacked(args, data, plot):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    set_style()
    unit = "count" if plot == "imaging_stacked_counts" else "percentage"
    summary = data.tables["stack_summary"]
    totals = data.tables["stack_total_summary"]
    conditions = CONDITIONS + (
        [DOCTORS] if not data.tables["mimic_doctors"].empty else []
    )
    pathologies = list(data.categories)
    modalities = next(iter(data.categories.values()))
    fig, ax = plt.subplots(figsize=(args.figure_width, args.stacked_height))
    positions = np.arange(len(pathologies))
    maximum = 0
    for condition_index, condition in enumerate(conditions):
        x = positions + (condition_index - (len(conditions) - 1) / 2) * 0.23
        bottom = np.zeros(len(pathologies))
        for modality in modalities:
            values = (
                summary.loc[
                    summary["Condition"].eq(condition)
                    & summary["Category"].eq(modality)
                ]
                .set_index("Pathology")
                .reindex(pathologies)[f"Mean_{unit}"]
                .to_numpy()
            )
            ax.bar(
                x,
                values,
                width=0.23,
                bottom=bottom,
                color=args.condition_colors[condition],
                edgecolor="black",
                linewidth=0.5,
                hatch=args.modality_hatches.get(modality, ""),
            )
            bottom += values
        errors = (
            np.zeros(len(pathologies))
            if condition == DOCTORS
            else totals.loc[totals["Condition"].eq(condition)]
            .set_index("Pathology")
            .reindex(pathologies)[f"Total_seed_SD_{unit}"]
            .fillna(0)
            .to_numpy()
        )
        if condition != DOCTORS:
            ax.errorbar(
                x,
                bottom,
                yerr=errors,
                fmt="none",
                ecolor="black",
                elinewidth=0.6,
                capsize=2,
                capthick=0.6,
            )
        for location, total, error in zip(x, bottom, errors):
            ax.text(
                location,
                total + error,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )
        maximum = max(maximum, (bottom + errors).max())
    ax.set(
        xticks=positions,
        xticklabels=pathologies,
        ylim=(0, max(1, maximum * 1.14)),
        xlabel="Pathology",
        ylabel="Cases by imaging request category (mean count)"
        if unit == "count"
        else "Cases by imaging request category (%)",
    )
    if unit == "count":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(
        handles=[
            Patch(
                facecolor=args.condition_colors[condition],
                edgecolor="black",
                label=condition,
            )
            for condition in conditions
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(conditions),
        frameon=False,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=args.modality_hatches.get(modality, ""),
                label=modality,
            )
            for modality in modalities
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(modalities),
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.16)
    return fig


def main(args: DictConfig):
    prepared = prepare_data(
        args
    )  # Validate all requested inputs before writing artifacts.
    output = (
        Path(to_absolute_path(str(Path(args.output_dir).expanduser())))
        / args.output_subdir
    )
    if args.render:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for plot, data in prepared.items():
            with matplotlib.rc_context():
                renderer = (
                    render_stacked
                    if plot.startswith("imaging_stacked_")
                    else render_distribution
                )
                figure = renderer(args, data, plot)
                try:
                    for fmt in dict.fromkeys(args.formats):
                        folder = output / fmt
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f"{args.output_prefix}_{plot}.{fmt}"
                        figure.savefig(
                            path,
                            format=fmt,
                            dpi=args.dpi,
                            bbox_inches="tight",
                            pad_inches=0.02,
                            facecolor="white",
                        )
                        print(f"Saved {path}")
                    if args.show:
                        plt.show()
                finally:
                    plt.close(figure)
    if args.statistics:
        for plot, data in prepared.items():
            folder = output / "statistics" / plot
            folder.mkdir(parents=True, exist_ok=True)
            for name, table in data.tables.items():
                table.to_csv(folder / f"{name}.csv", index=False)
    output.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    resolved.input_files = {
        key: str(path)
        for data in prepared.values()
        for key, path in data.input_files.items()
    }
    for field in ("results_dir", "output_dir"):
        if args[field]:
            resolved[field] = to_absolute_path(str(Path(args[field]).expanduser()))
    for key, value in args.input_dirs.items():
        if value:
            resolved.input_dirs[key] = to_absolute_path(str(Path(value).expanduser()))
    OmegaConf.save(
        resolved, output / "figures_orders_tests_treatments_config.yaml", resolve=True
    )
    return prepared


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs"),
    config_name="figures",
    version_base=None,
)
def run(args: DictConfig):
    main(args)


if __name__ == "__main__":
    run()

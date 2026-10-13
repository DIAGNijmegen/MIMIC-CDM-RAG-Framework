"""Shared CSV loading, model selection and exports for benchmark table scripts.

See docs/figures/table_common.md for the helper API and configuration.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import ListConfig, OmegaConf

if __package__:
    from .main_figure import configured_model_pairs, find_accuracy_csv
else:
    from main_figure import configured_model_pairs, find_accuracy_csv

METRICS = [
    "Diagnostic Accuracy",
    "Alignment",
    "End-to-End Treatment",
    "Precision@6_equiv",
    "Brier",
]
OPTIONAL_METRIC = "Conditional Treatment"
PERFORMANCE_COLUMNS = [
    "Diagnostic\naccuracy %",
    "Test-ordering\nalignment %",
    "Treatment\nalignment %",
    "Retrieval\nP@6 %",
    "Retrieval spread\n(Brier score)",
]
RANK_COLUMN = "Rank by\nmedian rank"
DIRECTIONS = {
    RANK_COLUMN: "lower",
    **{key: "higher" for key in PERFORMANCE_COLUMNS[:-1]},
    PERFORMANCE_COLUMNS[-1]: "lower",
}
VALUE_FORMATS = {
    RANK_COLUMN: "{:.0f}",
    **{key: "{:.2f}" for key in PERFORMANCE_COLUMNS[:-1]},
    PERFORMANCE_COLUMNS[-1]: "{:.3f}",
}
CI_COLUMNS = {key: key + "__ci" for key in PERFORMANCE_COLUMNS}
SOURCES = {
    "diagnostic_accuracy": {
        "Diagnostic Accuracy": [
            "Diagnostic Accuracy",
            "Diagnostic_Accuracy",
            "DiagnosticAccuracy",
            "Accuracy",
            "Mean",
        ]
    },
    "alignment": {"Alignment": ["Mean", "Alignment", "Test-ordering Alignment"]},
    "retrieval": {
        "Precision@6_equiv": ["Precision@6_equiv", "Precision@6"],
        "Brier": ["Brier"],
    },
    "treatment": {
        "End-to-End Treatment": [
            "End-to-End Treatment Success (%)",
            "End-to-End Treatment",
            "End_to_End_Treatment",
        ],
        OPTIONAL_METRIC: ["Conditional Treatment Recall (%)", "Conditional Treatment"],
    },
}


@dataclass
class TableData:
    table: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    input_files: dict[str, Path]
    section_order: list[str]
    model_names: dict[str, str]


def validate_settings(args, profile_field, profile_name):
    if profile_field not in args:
        raise ValueError(
            f"Select the matching Hydra profile with figures={profile_name}."
        )
    if (
        not isinstance(args.seeds, (ListConfig, list, tuple))
        or not args.seeds
        or type(args.base_seed) is not int
        or any(type(seed) is not int for seed in args.seeds)
    ):
        raise ValueError(
            "base_seed and selected seeds must be integers; seeds cannot be empty."
        )
    if (
        not isinstance(args.confidence, (int, float))
        or isinstance(args.confidence, bool)
        or not 0 < args.confidence < 1
    ):
        raise ValueError("confidence must be between 0 and 1.")
    for field in ("render", "statistics", "show", "zero_as_missing"):
        if type(args[field]) is not bool:
            raise ValueError(f"{field} must be true or false.")
    if not args.render and not args.statistics:
        raise ValueError("Enable render or statistics.")
    if args.render and (
        not args.formats
        or any(fmt not in ("pdf", "png", "svg") for fmt in args.formats)
    ):
        raise ValueError("formats must select pdf, png and/or svg.")
    if type(args.dpi) is not int or args.dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    for field in ("figure_width", "height_scale", "min_height"):
        value = args[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{field} must be a positive finite number.")
    if args.duplicate_rows not in ("mean", "error") or args.missing_rows not in (
        "keep",
        "error",
    ):
        raise ValueError(
            "duplicate_rows must be mean/error; missing_rows must be keep/error."
        )
    if (
        not isinstance(args.output_name, str)
        or not args.output_name
        or Path(args.output_name).name != args.output_name
        or args.output_name in (".", "..")
    ):
        raise ValueError("output_name must be a filename stem.")
    if (
        not isinstance(args.output_subdir, str)
        or not args.output_subdir
        or Path(args.output_subdir).name != args.output_subdir
        or args.output_subdir in (".", "..")
    ):
        raise ValueError("output_subdir must be a single subdirectory name.")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("output_dir must be a nonempty path.")
    pairs = configured_model_pairs(args)
    if any(
        not isinstance(pair.label, str) or not pair.label.strip() for pair in pairs
    ) or len({pair.label for pair in pairs}) != len(pairs):
        raise ValueError("Selected model names must be nonempty and unique.")
    return pairs


def split_identifier(identifier, metadata, base_seed):
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("CSV model identifiers must be nonempty strings.")
    identifier = identifier.strip()
    match = re.search(r"(?:_SEED=|\s+\(seed=)(-?\d+)\)?$", identifier)
    seed = int(match.group(1)) if match else base_seed
    configuration = identifier[: match.start()] if match else identifier
    if pd.notna(metadata):
        value = float(metadata)
        if not np.isfinite(value) or value != int(value) or isinstance(metadata, bool):
            raise ValueError(f"Invalid Seed metadata: {metadata!r}")
        if match and seed != int(value):
            raise ValueError(
                f"Seed suffix disagrees with Seed metadata: {identifier!r}"
            )
        seed = int(value)
    return configuration, seed


def read_metric_source(path, source, args, wanted_ids):
    frame = pd.read_csv(path)
    if "Model" not in frame:
        raise ValueError(f"{path}: missing Model column.")
    if source in ("diagnostic_accuracy", "retrieval"):
        if "Pathology" not in frame:
            raise ValueError(f"{path}: missing Pathology column.")
        frame = frame.loc[
            frame["Pathology"].astype(str).str.strip().str.lower().eq("mean")
        ].copy()
    identifiers = (
        frame["Raw Model"].fillna(frame["Model"])
        if "Raw Model" in frame
        else frame["Model"]
    )
    identity_rows = [
        split_identifier(
            identifier,
            frame.at[index, "Seed"] if "Seed" in frame else np.nan,
            args.base_seed,
        )
        for index, identifier in identifiers.items()
    ]
    frame["ConfigID"] = [
        args.model_aliases.get(configuration, configuration)
        for configuration, _ in identity_rows
    ]
    frame["Seed"] = [seed for _, seed in identity_rows]
    frame = frame.loc[
        frame["ConfigID"].isin(wanted_ids) & frame["Seed"].isin(args.seeds)
    ].copy()
    result = frame[["ConfigID", "Seed"]].copy()
    for metric, candidates in SOURCES[source].items():
        column = next((key for key in candidates if key in frame), None)
        if column is None:
            if metric == OPTIONAL_METRIC:
                continue
            raise ValueError(
                f"{path}: missing {metric!r}; expected one of {candidates}"
            )
        values = pd.to_numeric(frame[column], errors="raise")
        valid = values.dropna()
        upper = (
            1
            if metric == "Precision@6_equiv"
            else (np.inf if metric == "Brier" else 100)
        )
        if not np.isfinite(valid).all() or ((valid < 0) | (valid > upper)).any():
            raise ValueError(
                f"{path}: {metric} contains nonfinite or out-of-range values."
            )
        result[metric] = values
    keys = ["ConfigID", "Seed"]
    if result.duplicated(keys).any() and args.duplicate_rows == "error":
        raise ValueError(f"{path}: duplicate configuration/seed rows.")
    return result.groupby(keys, as_index=False, sort=False).mean()


def load_metrics(args, configurations, baseline_ids):
    """Align the four sources by exact configuration ID and seed before aggregation."""
    if len(set(configurations)) != len(configurations):
        raise ValueError("Configured source IDs must be unique.")
    seeds = list(dict.fromkeys(args.seeds))
    merged = pd.MultiIndex.from_product(
        [configurations, seeds], names=["ConfigID", "Seed"]
    ).to_frame(index=False)
    paths = {}
    for source in SOURCES:
        directory = args.metric_dirs.get(source) or args.results_dir
        paths[source] = find_accuracy_csv(
            directory,
            f"metric_dirs.{source} / results_dir",
            args.metric_patterns[source],
        )
        values = read_metric_source(paths[source], source, args, set(configurations))
        merged = merged.merge(
            values, on=["ConfigID", "Seed"], how="left", validate="one_to_one"
        )
    if OPTIONAL_METRIC not in merged:
        merged[OPTIONAL_METRIC] = np.nan
    missing = merged[METRICS].isna()
    missing.loc[
        merged["ConfigID"].isin(baseline_ids), ["Precision@6_equiv", "Brier"]
    ] = False
    report_rows = [
        {"ConfigID": row["ConfigID"], "Seed": row["Seed"], "Metric": metric}
        for index, row in merged.iterrows()
        for metric in METRICS
        if missing.at[index, metric]
    ]
    report = pd.DataFrame(report_rows, columns=["ConfigID", "Seed", "Metric"])
    if not report.empty and args.missing_rows == "error":
        raise ValueError(
            "Missing configured metric values:\n" + report.to_string(index=False)
        )
    if not report.empty:
        print(
            f"Missing metric values: {len(report)}; included as empty cells and recorded in missing_metrics.csv."
        )
    return merged, paths, report


def ranked_metrics(frame):
    ranks = pd.DataFrame(
        {
            metric: frame[metric].rank(
                method="average",
                ascending=DIRECTIONS[metric] == "lower",
                na_option="keep",
            )
            for metric in PERFORMANCE_COLUMNS
        }
    )
    median = ranks.median(axis=1, skipna=True)
    return median, median.rank(method="min", ascending=True, na_option="keep")


def metric_plot_values(frame):
    output = pd.DataFrame(index=frame.index)
    for metric, label in zip(METRICS, PERFORMANCE_COLUMNS):
        output[label] = frame[metric] * (100 if metric == "Precision@6_equiv" else 1)
    return output


def export_table(args, data, renderer, profile_name):
    """Write profile-specific images/tables, with no notebook display state."""
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
            matplotlib.rcParams.update(
                {
                    "pdf.fonttype": 42,
                    "svg.fonttype": "none",
                    "font.family": "sans-serif",
                    "font.sans-serif": [
                        "Arial",
                        "Helvetica",
                        "Liberation Sans",
                        "DejaVu Sans",
                    ],
                }
            )
            fig, _ = renderer(data, args)
            try:
                for fmt in dict.fromkeys(args.formats):
                    folder = output / fmt
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{args.output_name}.{fmt}"
                    fig.savefig(
                        path,
                        format=fmt,
                        dpi=args.dpi,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    print(f"Saved {path}")
                if args.show:
                    plt.show()
            finally:
                plt.close(fig)
    if args.statistics:
        folder = output / "statistics"
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in data.tables.items():
            table = table.rename(
                columns=lambda key: (
                    key.replace("\n", " ") if isinstance(key, str) else key
                )
            )
            table.to_csv(folder / f"{name}.csv", index=False)
    output.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    resolved.input_files = {name: str(path) for name, path in data.input_files.items()}
    if args.results_dir:
        resolved.results_dir = to_absolute_path(
            str(Path(args.results_dir).expanduser())
        )
    resolved.output_dir = to_absolute_path(str(Path(args.output_dir).expanduser()))
    OmegaConf.save(resolved, output / f"{profile_name}_config.yaml", resolve=True)
    return data

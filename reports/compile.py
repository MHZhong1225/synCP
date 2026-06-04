from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reports.output import write_results


KEY_METHODS = [
    "Standard CP",
    "Classwise CP",
    "Known-group CP",
    "Error-slice",
    "K-means",
    "Random halfspace",
    "GMM-unaware",
    "Coverage-Domino",
    "DOCP-MVP",
    "DOCP",
]

METHOD_LABELS = {
    "Standard CP": ("Baseline", "Standard CP"),
    "Classwise CP": ("Baseline", "Classwise CP"),
    "Known-group CP": ("Baseline", "Known-group CP"),
    "Error-slice": ("Baseline", "Error-slice"),
    "K-means": ("Baseline", "K-means"),
    "Random halfspace": ("Ablation", "Random halfspace"),
    "GMM-unaware": ("Ablation", "GMM-unaware"),
    "Coverage-Domino": ("Ablation", "Coverage-Domino"),
    "DOCP-MVP": ("Ablation", "DOCP-MVP"),
    "DOCP": ("Ours", "DOCP"),
}

GROUP_ORDER = {"Baseline": 0, "Ablation": 1, "Ours": 2}

METRIC_COLUMNS = [
    "marginal_coverage",
    "worst_class_coverage",
    "worst_known_group_coverage",
    "worst_hidden_group_coverage",
    "worst_discovered_slice_coverage",
    "avg_set_size",
    "max_undercoverage_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a cross-experiment DOCP result report.")
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional list of result directories. Defaults to every child directory with summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (args.root / "compiled_report")
    outdir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = _find_experiments(args.root, args.experiments)
    if not experiment_dirs:
        raise FileNotFoundError(f"No summary.csv files found under {args.root}")

    overall = _load_overall(experiment_dirs)
    comparison = _make_comparison_table(overall)

    write_results(overall, outdir / "overall_key_metrics.csv")
    write_results(comparison, outdir / "baseline_vs_ours.csv")
    (outdir / "baseline_vs_ours.md").write_text(_markdown_table(comparison), encoding="utf-8")

    report = _render_markdown(comparison)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote compiled report to {outdir}")


def _find_experiments(root: Path, experiments: list[str] | None) -> list[Path]:
    if experiments:
        paths = [Path(name) if Path(name).is_absolute() else root / name for name in experiments]
    else:
        paths = [path for path in sorted(root.iterdir()) if path.is_dir()]
    return [path for path in paths if (path / "summary.csv").exists()]


def _load_overall(experiment_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for exp_dir in experiment_dirs:
        frame = pd.read_csv(exp_dir / "summary.csv")
        frame = frame[frame["method"].isin(KEY_METHODS)].copy()
        frame.insert(0, "experiment", exp_dir.name)
        if "num_runs" not in frame.columns:
            frame["num_runs"] = 1
        for column in METRIC_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frames.append(frame[["experiment", "method", "num_runs", *METRIC_COLUMNS]])
    return pd.concat(frames, ignore_index=True)


def _make_comparison_table(overall: pd.DataFrame) -> pd.DataFrame:
    frame = overall.copy()
    frame[["group", "method"]] = frame["method"].apply(
        lambda name: pd.Series(METHOD_LABELS.get(name, ("Other", name)))
    )
    frame["_group_order"] = frame["group"].map(GROUP_ORDER).fillna(99)
    columns = [
        "experiment",
        "group",
        "method",
        "marginal_coverage",
        "worst_class_coverage",
        "worst_known_group_coverage",
        "worst_hidden_group_coverage",
        "worst_discovered_slice_coverage",
        "avg_set_size",
        "max_undercoverage_gap",
    ]
    frame = frame[columns + ["_group_order"]]
    frame["_method_order"] = frame["method"].map({"DOCP-MVP": 90, "DOCP": 99}).fillna(50)
    frame = frame.sort_values(["experiment", "_group_order", "_method_order", "method"])
    frame = frame.drop(columns=["_group_order", "_method_order"])
    return _round_numeric(frame)


def _render_markdown(comparison: pd.DataFrame) -> str:
    lines = [
        "# DOCP Experiment Report",
        "",
        "Target coverage is 0.90 unless otherwise stated in the per-run summaries.",
        "",
        "## Baselines vs Ours",
        "",
        _markdown_table(comparison),
    ]
    return "\n".join(lines) + "\n"


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    numeric = rounded.select_dtypes(include="number").columns
    rounded[numeric] = rounded[numeric].round(5)
    return rounded


def _markdown_table(frame: pd.DataFrame) -> str:
    return frame.fillna("").to_markdown(index=False)


if __name__ == "__main__":
    main()

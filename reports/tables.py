from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reports.output import write_results


METHOD_LABELS = {
    "Standard CP": ("Baseline", "Standard CP"),
    "Classwise CP": ("Baseline", "Classwise CP"),
    "Known-group CP": ("Baseline", "Known-group CP"),
    "Error-slice": ("Baseline", "Error-slice"),
    "Error-slice GCP": ("Baseline", "Error-slice GCP"),
    "K-means": ("Baseline", "K-means"),
    "K-means GCP": ("Baseline", "K-means GCP"),
    "Random halfspace": ("Ablation", "Random halfspace"),
    "GMM-unaware": ("Ablation", "GMM-unaware"),
    "Coverage-Domino": ("Ablation", "Coverage-Domino"),
    "Random halfspace GCP": ("Ablation", "Random halfspace GCP"),
    "GMM-unaware GCP": ("Ablation", "GMM-unaware GCP"),
    "Coverage-Domino GCP": ("Ablation", "Coverage-Domino GCP"),
    "DOCP-MVP": ("Ablation", "DOCP-MVP"),
    "DOCP": ("Ours", "DOCP"),
}

GROUP_ORDER = {"Baseline": 0, "Ablation": 1, "Ours": 2}

TABLE1_COLUMNS = [
    "group",
    "method",
    "marginal_coverage",
    "worst_class_coverage",
    "worst_known_group_coverage",
    "worst_discovered_slice_coverage",
    "avg_set_size",
    "max_undercoverage_gap",
]

TABLE3_COLUMNS = [
    "group",
    "method",
    "worst_discovered_slice_coverage",
    "worst_class_coverage",
    "avg_set_size",
    "max_undercoverage_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style tables from aggregated DOCP outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (args.root / "tables")
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.root / "summary.csv")
    summary = _format_methods(summary)
    table1 = _select_columns(summary, TABLE1_COLUMNS)
    table3 = _select_columns(summary, TABLE3_COLUMNS)
    _write_table(table1, outdir / "table1_overall")
    _write_table(table1, outdir / "baseline_vs_ours")
    _write_table(table3, outdir / "table3_efficiency_tradeoff")

    print(f"Wrote tables to {outdir}")


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in frame.columns]
    return frame[existing].copy()


def _format_methods(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    groups = []
    labels = []
    for method in formatted["method"]:
        group, label = METHOD_LABELS.get(method, ("Other", method))
        groups.append(group)
        labels.append(label)
    formatted["group"] = groups
    formatted["method"] = labels
    formatted["_group_order"] = formatted["group"].map(GROUP_ORDER).fillna(99)
    formatted["_method_order"] = formatted["method"].map({"DOCP-MVP": 90, "DOCP": 99}).fillna(50)
    formatted = formatted.sort_values(["_group_order", "_method_order", "method"])
    formatted = formatted.drop(columns=["_group_order", "_method_order"])
    return formatted


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    rounded = write_results(frame, stem.with_suffix(".csv"))
    stem.with_suffix(".md").write_text(rounded.fillna("").to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reports.output import write_results


DATASETS = {
    "bracs": "BRACS",
    "chest_xray": "Chest X-ray",
    "nih": "NIH Effusion",
    "rsna_5k": "RSNA Breast 5k",
}

PARAM_SETS = {
    "default": "Default",
    "tuned": "Tuned",
}

DOCP_METRICS = [
    "marginal_coverage",
    "avg_set_size",
    "max_undercoverage_gap",
    "worst_discovered_slice_coverage",
    "worst_class_coverage",
    "worst_known_group_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare default and tuned DOCP parameter sets.")
    parser.add_argument("--root", type=Path, default=Path("results/param_compare"))
    parser.add_argument("--outdir", type=Path, default=Path("results/param_compare"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = load_all_results(args.root)
    write_results(full, args.outdir / "all_methods_mean_std.csv")

    docp = select_docp_rows(full)
    write_results(docp, args.outdir / "docp_param_comparison.csv")

    winners = choose_best_params(docp)
    write_results(winners, args.outdir / "best_param_set_by_dataset.csv")

    print("\nDOCP parameter comparison:")
    print(docp.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nBest parameter set by dataset:")
    print(winners.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nwrote: {args.outdir / 'all_methods_mean_std.csv'}")
    print(f"wrote: {args.outdir / 'docp_param_comparison.csv'}")
    print(f"wrote: {args.outdir / 'best_param_set_by_dataset.csv'}")


def load_all_results(root: Path) -> pd.DataFrame:
    frames = []
    for dataset_key, dataset_name in DATASETS.items():
        for param_key, param_name in PARAM_SETS.items():
            path = root / dataset_key / param_key / "summary.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing result file: {path}")
            frame = pd.read_csv(path)
            frame.insert(0, "dataset", dataset_name)
            frame.insert(1, "param_set", param_name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def select_docp_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "param_set", "method"]
    for metric in DOCP_METRICS:
        if metric in frame.columns:
            columns.append(metric)
        std_col = f"{metric}_std"
        if std_col in frame.columns:
            columns.append(std_col)
    return frame[frame["method"].eq("DOCP")][columns].reset_index(drop=True)


def choose_best_params(docp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in docp.groupby("dataset", sort=False):
        ranked = group.copy()
        slice_cov = ranked["worst_discovered_slice_coverage"].fillna(ranked["marginal_coverage"])
        ranked["_coverage_ok"] = (ranked["marginal_coverage"] >= 0.9) & (slice_cov >= 0.9)
        if ranked["_coverage_ok"].any():
            ranked = ranked[ranked["_coverage_ok"]].sort_values(
                ["avg_set_size", "max_undercoverage_gap", "worst_discovered_slice_coverage", "marginal_coverage"],
                ascending=[True, True, False, False],
            )
        else:
            ranked = ranked.sort_values(
                ["worst_discovered_slice_coverage", "max_undercoverage_gap", "marginal_coverage", "avg_set_size"],
                ascending=[False, True, False, True],
            )
        best = ranked.iloc[0].drop(labels=["_coverage_ok"]).to_dict()
        best["selection_rule"] = "min set size among coverage-ok rows" if ranked.iloc[0]["_coverage_ok"] else "best available slice coverage"
        rows.append(best)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

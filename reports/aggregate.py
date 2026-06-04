from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reports.output import print_results, write_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DOCP experiment summaries.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in sorted(args.root.glob("**/summary.csv"))]
    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {args.root}")
    all_results = pd.concat(frames, ignore_index=True)
    metric_cols = [col for col in all_results.columns if col != "method"]
    mean = all_results.groupby("method")[metric_cols].mean(numeric_only=True)
    counts = all_results.groupby("method").size().rename("num_runs")
    summary = pd.concat([counts, mean], axis=1).reset_index()
    out = args.out or (args.root / "summary.csv")
    cleaned = write_results(summary, out)
    print_results(cleaned)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()

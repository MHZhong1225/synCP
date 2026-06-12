from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from runners.benchmark_tasks import expand_tasks
from runners.run_final_consensus_benchmark import run_task


DEFAULT_TASKS = ["isic", "ham10000"]
DEFAULT_SCORES = ["thr", "aps", "randomized_aps", "raps"]
RETAINED_METHODS = ["Standard CP", "Consensus-CPScore"]
SUMMARY_METRICS = [
    "marginal_coverage",
    "avg_set_size",
    "worst_class_coverage",
    "worst_known_group_coverage",
    "max_common_undercoverage_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare THR, APS, randomized APS, and RAPS on cached multiclass tasks.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--scores", nargs="+", default=DEFAULT_SCORES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--repeat-cache-root", type=Path, default=Path("results/npz/repeats"))
    parser.add_argument("--outdir", type=Path, default=Path("results/score_ablation"))
    parser.add_argument("--raps-penalty", type=float, default=0.01)
    parser.add_argument("--raps-regularization-rank", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for seed in args.seeds:
        cache_root = args.repeat_cache_root / f"seed_{seed:02d}"
        for task, family in expand_tasks(args.tasks, []):
            for score in args.scores:
                print(f"[seed {seed:02d}] task={task} score={score}", flush=True)
                results, _ = run_task(
                    task,
                    family,
                    seed,
                    cache_root,
                    score_override=score,
                    raps_penalty=args.raps_penalty,
                    raps_regularization_rank=args.raps_regularization_rank,
                )
                retained = results[results["method"].isin(RETAINED_METHODS)].copy()
                retained.insert(0, "score", score)
                retained.insert(0, "run_seed", seed)
                rows.append(retained)
    all_results = pd.concat(rows, ignore_index=True, sort=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.outdir / "all_results_by_seed.csv", index=False, float_format="%.5f")
    summary = summarize(all_results)
    summary.to_csv(args.outdir / "summary_mean_std.csv", index=False, float_format="%.5f")
    (args.outdir / "analysis.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in SUMMARY_METRICS if metric in results]
    groups = ["task", "score", "method"]
    mean = results.groupby(groups, as_index=False)[metrics].mean(numeric_only=True)
    std = results.groupby(groups, as_index=False)[metrics].std(numeric_only=True).fillna(0.0)
    mean = mean.rename(columns={metric: f"{metric}_mean" for metric in metrics})
    std = std.rename(columns={metric: f"{metric}_std" for metric in metrics})
    return mean.merge(std, on=groups, how="left")


if __name__ == "__main__":
    main()

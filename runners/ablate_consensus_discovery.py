from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from core.conformal import conformal_quantile, covered, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_threshold_model
from core.rectification import fit_batch_gcp
from experiments import images as image_experiment
from reports.output import round_numbers
from runners.run_full_benchmark import _expand_tasks, _format_result_columns, _load_task
from variants.consensus_latent.consensus import discover_consensus_groups


DISCOVERY_METHODS = (
    "coverage_domino",
    "undercoverage_risk",
    "kmeans",
    "random_halfspace",
)
DISPLAY_NAMES = {
    "coverage_domino": "Coverage-Domino",
    "undercoverage_risk": "Undercoverage-Risk",
    "kmeans": "K-means",
    "random_halfspace": "Random-Halfspace",
}
DEFAULT_SEEDS = tuple(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate the discovery source used by Consensus-Hard.")
    parser.add_argument("--tasks", nargs="+", default=["bracs", "nih_all", "isic", "ham10000", "rsna_5k", "mimic_care"])
    parser.add_argument("--nih-targets", nargs="+", default=["Effusion"])
    parser.add_argument("--methods", nargs="+", choices=DISCOVERY_METHODS, default=["kmeans", "undercoverage_risk"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=Path("results/consensus_discovery_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results, runtimes = [], []
    for task, family in _expand_tasks(args.tasks, args.nih_targets):
        print(f"running consensus discovery ablation: task={task}", flush=True)
        task_results, task_runtimes = run_task(task, family, args)
        results.append(task_results)
        runtimes.append(task_runtimes)
        _write(task_results, args.outdir / task / "results.csv")
        _write(task_runtimes, args.outdir / task / "runtimes.csv")
    all_results = pd.concat(results, ignore_index=True)
    all_runtimes = pd.concat(runtimes, ignore_index=True)
    _write(all_results, args.outdir / "all_results.csv")
    _write(all_runtimes, args.outdir / "all_runtimes.csv")
    _write(_aggregate_results(all_results), args.outdir / "aggregate_results.csv")
    _write(_aggregate_runtimes(all_runtimes), args.outdir / "aggregate_runtimes.csv")
    (args.outdir / "config.json").write_text(
        json.dumps(
            {
                "tasks": args.tasks,
                "nih_targets": args.nih_targets,
                "methods": args.methods,
                "seeds": args.seeds,
                "split_seed": args.split_seed,
                "consensus": "same Consensus-Hard aggregation and BatchGCP rectification for every discovery source",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_task(task: str, family: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_args, data, probs, embeddings, score_name = _load_task(task, family, args.split_seed)
    score_matrix = score_matrix_from_probs(probs, score_name)
    uncertainty = observable_uncertainty(probs)
    base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), task_args.alpha)
    failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    inputs = image_experiment.DiscoveryInputs(
        embeddings[data.discovery_idx],
        uncertainty[data.discovery_idx],
        failures,
        probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx],
    )
    discovered = {}
    timing_parts = {}
    for method in args.methods:
        start = perf_counter()
        groups_by_seed = {}
        for seed in args.seeds:
            seed_args = argparse.Namespace(**vars(task_args))
            seed_args.seed = seed
            groups_by_seed[(method, seed)] = image_experiment.discover_slice_method(method, seed_args, inputs)
        discovery_seconds = perf_counter() - start
        start = perf_counter()
        groups, _, _ = discover_consensus_groups(
            groups_by_seed,
            embeddings[data.discovery_idx],
            uncertainty[data.discovery_idx],
            failures,
            embeddings[data.mvcal_idx],
            uncertainty[data.mvcal_idx],
            task_args.alpha,
            effect_threshold=task_args.epsilon,
            max_groups=task_args.pool_groups,
        )
        timing_parts[method] = (discovery_seconds, perf_counter() - start)
        discovered[method] = groups

    test = data.test_idx
    known_test = data.known_group[test] if data.known_group is not None else None
    shared_groups = [group for groups in discovered.values() for group in groups]
    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    rows, runtimes = [], []
    for method in args.methods:
        discovery_seconds, aggregation_seconds = timing_parts[method]
        start = perf_counter()
        model = fit_batch_gcp(
            base_q,
            discovered[method],
            embeddings[data.mvcal_idx],
            mv_scores,
            task_args.alpha,
            observed_scores=probs[data.mvcal_idx],
            min_size=task_args.rectifier_min_size,
        )
        rectification_seconds = perf_counter() - start
        start = perf_counter()
        row = evaluate_threshold_model(
            "Consensus-Hard",
            model,
            embeddings[test],
            score_matrix[test],
            data.labels[test],
            None,
            known_test,
            shared_groups,
            task_args.alpha,
            observed_scores=probs[test],
            eval_observed_scores=probs[test],
            own_groups=discovered[method],
        )
        evaluation_seconds = perf_counter() - start
        row.update({"discovery_method": DISPLAY_NAMES[method], "task": task, "family": family, "num_selected_slices": len(discovered[method])})
        rows.append(row)
        runtimes.append(
            {
                "discovery_method": DISPLAY_NAMES[method],
                "task": task,
                "family": family,
                "discovery_seconds": discovery_seconds,
                "aggregation_seconds": aggregation_seconds,
                "rectification_seconds": rectification_seconds,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": discovery_seconds + aggregation_seconds + rectification_seconds + evaluation_seconds,
            }
        )
    formatted = _format_result_columns(pd.DataFrame(rows), task, family)
    formatted.insert(1, "discovery_method", [DISPLAY_NAMES[method] for method in args.methods])
    return formatted, pd.DataFrame(runtimes)


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in frame if column not in {"method", "discovery_method", "task", "family"}]
    return frame.groupby("discovery_method", as_index=False)[metrics].mean(numeric_only=True)


def _aggregate_runtimes(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["discovery_seconds", "aggregation_seconds", "rectification_seconds", "evaluation_seconds", "total_seconds"]
    return frame.groupby("discovery_method", as_index=False)[metrics].mean(numeric_only=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numbers(frame).to_csv(path, index=False, float_format="%.5f")


if __name__ == "__main__":
    main()

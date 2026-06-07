from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.conformal import (
    conformal_quantile,
    covered,
    observable_uncertainty,
    score_matrix_from_probs,
    set_sizes,
    true_label_scores,
)
from core.rectification import fit_slice_quantile
from experiments import images as image_experiment
from reports.output import write_results
from slices.discovery import GroupFunction


DEFAULT_DATASETS = ("ham10000", "nih", "mimic_effusion", "isic", "rsna_5k")
DEFAULT_SEEDS = tuple(range(1, 11))


@dataclass
class SeedRun:
    seed: int
    groups: list[GroupFunction]
    discovery_memberships: np.ndarray
    anchor_memberships: np.ndarray
    union_membership: np.ndarray
    row: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure within-method slice instability while holding features and data splits fixed."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--method", choices=image_experiment.DISCOVERY_METHODS, default="coverage_domino")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=Path("results/seed_stability"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_runs = []
    all_pairs = []
    all_slices = []
    dataset_summaries = []

    for dataset in args.datasets:
        print(f"analyzing {dataset}: method={args.method}, discovery_seeds={args.seeds}", flush=True)
        run_rows, pair_rows, slice_rows, summary = analyze_dataset(
            dataset,
            args.method,
            args.seeds,
            args.split_seed,
        )
        all_runs.extend(run_rows)
        all_pairs.extend(pair_rows)
        all_slices.extend(slice_rows)
        dataset_summaries.append(summary)

    runs = write_results(pd.DataFrame(all_runs), args.outdir / "per_seed_results.csv")
    pairs = write_results(pd.DataFrame(all_pairs), args.outdir / "pairwise_stability.csv")
    slices = write_results(pd.DataFrame(all_slices), args.outdir / "per_slice_results.csv")
    summary = write_results(pd.DataFrame(dataset_summaries), args.outdir / "summary.csv")
    _write_markdown(summary, runs, pairs, args.outdir / "report.md")
    config = {
        "datasets": args.datasets,
        "method": args.method,
        "discovery_seeds": args.seeds,
        "fixed_split_seed": args.split_seed,
        "control": "fixed data split, labels, probabilities, embeddings, base CP threshold, and rectification split; only discovery algorithm seed changes",
    }
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote: {args.outdir}", flush=True)


def analyze_dataset(
    dataset: str,
    method: str,
    seeds: list[int],
    split_seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    image_args = _image_args(dataset, split_seed)
    data = image_experiment.load_experiment_data(image_args)
    probs, embeddings = image_experiment._load_feature_cache(
        image_args.cache_file,
        len(data.class_names),
        image_args.backbone,
    )
    score_name = image_experiment._resolve_score(len(data.class_names))
    score_matrix = score_matrix_from_probs(probs, score_name)
    uncertainty = observable_uncertainty(probs)
    base_q = conformal_quantile(
        true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]),
        image_args.alpha,
    )
    discovery_failures = ~covered(
        score_matrix[data.discovery_idx],
        data.labels[data.discovery_idx],
        base_q,
    )
    inputs = image_experiment.DiscoveryInputs(
        embeddings=embeddings[data.discovery_idx],
        uncertainty=uncertainty[data.discovery_idx],
        coverage_failures=discovery_failures,
        classification_errors=probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx],
    )

    runs = []
    slice_rows = []
    for seed in seeds:
        seed_args = argparse.Namespace(**vars(image_args))
        seed_args.seed = seed
        groups = image_experiment.discover_slice_method(method, seed_args, inputs)
        discovery_memberships = _membership_matrix(
            groups,
            embeddings[data.discovery_idx],
            uncertainty[data.discovery_idx],
        )
        anchor_memberships = _membership_matrix(
            groups,
            embeddings[data.test_idx],
            uncertainty[data.test_idx],
        )
        union_membership = anchor_memberships.any(axis=1) if anchor_memberships.shape[1] else np.zeros(len(data.test_idx), dtype=bool)
        row, seed_slice_rows = _evaluate_seed(
            dataset,
            method,
            seed,
            groups,
            anchor_memberships,
            union_membership,
            data,
            embeddings,
            probs,
            score_matrix,
            uncertainty,
            base_q,
            image_args,
        )
        runs.append(
            SeedRun(
                seed=seed,
                groups=groups,
                discovery_memberships=discovery_memberships,
                anchor_memberships=anchor_memberships,
                union_membership=union_membership,
                row=row,
            )
        )
        slice_rows.extend(seed_slice_rows)

    pair_rows = _pairwise_rows(dataset, method, runs)
    run_rows = [run.row for run in runs]
    summary = _summarize(dataset, method, run_rows, pair_rows)
    return run_rows, pair_rows, slice_rows, summary


def _image_args(dataset: str, split_seed: int) -> argparse.Namespace:
    if dataset not in image_experiment.DATASET_PRESETS:
        raise KeyError(f"Unknown dataset preset: {dataset}")
    preset = image_experiment.DATASET_PRESETS[dataset]
    hyperparams = image_experiment._load_fixed_hyperparams(image_experiment.DEFAULT_HYPERPARAMS, dataset)
    return argparse.Namespace(
        preset=dataset,
        dataset_root=preset["dataset_root"],
        dataset_type=preset.get("dataset_type", "imagefolder"),
        split_mode="official_test",
        nih_target=preset.get("nih_target", "Effusion"),
        mimic_target=preset.get("mimic_target", "Pleural Effusion"),
        mimic_iv_root=preset.get("mimic_iv_root", Path("../dataset/mimic-iv-3.1")),
        mimic_ed_root=preset.get("mimic_ed_root", Path("../dataset/mimic-iv-ed-2.2")),
        known_group=preset.get("known_group", "gender"),
        max_samples=preset.get("max_samples"),
        cache_file=preset["cache_file"],
        backbone="resnet18",
        seed=split_seed,
        alpha=0.10,
        components=int(hyperparams["components"]),
        min_size=int(hyperparams["min_size"]),
        rectifier_min_size=int(hyperparams["rectifier_min_size"]),
        pool_groups=int(hyperparams["pool_groups"]),
        epsilon=float(hyperparams["epsilon"]),
        enrichment=float(hyperparams["enrichment"]),
    )


def _membership_matrix(
    groups: list[GroupFunction],
    embeddings: np.ndarray,
    uncertainty: np.ndarray,
) -> np.ndarray:
    if not groups:
        return np.zeros((len(embeddings), 0), dtype=bool)
    return np.column_stack([group.membership(embeddings, uncertainty) for group in groups])


def _evaluate_seed(
    dataset: str,
    method: str,
    seed: int,
    groups: list[GroupFunction],
    anchor_memberships: np.ndarray,
    union_membership: np.ndarray,
    data,
    embeddings: np.ndarray,
    probs: np.ndarray,
    score_matrix: np.ndarray,
    uncertainty: np.ndarray,
    base_q: float,
    image_args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    test_scores = score_matrix[data.test_idx]
    test_y = data.labels[data.test_idx]
    standard_covered = covered(test_scores, test_y, base_q)
    standard_sizes = set_sizes(test_scores, base_q)
    mv_true_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    rect = fit_slice_quantile(
        base_q,
        groups,
        embeddings[data.mvcal_idx],
        mv_true_scores,
        image_args.alpha,
        observed_scores=uncertainty[data.mvcal_idx],
        min_size=image_args.rectifier_min_size,
    )
    thresholds = rect.thresholds(
        embeddings[data.test_idx],
        observed_scores=uncertainty[data.test_idx],
        score_matrix=test_scores,
    )
    rect_covered = covered(test_scores, test_y, thresholds)
    rect_sizes = set_sizes(test_scores, thresholds)
    slice_coverages = [
        float(standard_covered[anchor_memberships[:, index]].mean())
        for index in range(anchor_memberships.shape[1])
        if anchor_memberships[:, index].sum() >= image_args.min_size
    ]
    rect_slice_coverages = [
        float(rect_covered[anchor_memberships[:, index]].mean())
        for index in range(anchor_memberships.shape[1])
        if anchor_memberships[:, index].sum() >= image_args.min_size
    ]
    row = {
        "dataset": dataset,
        "method": method,
        "discovery_seed": seed,
        "num_slices": len(groups),
        "anchor_union_size": int(union_membership.sum()),
        "anchor_union_fraction": float(union_membership.mean()),
        "standard_marginal_coverage": float(standard_covered.mean()),
        "rectified_marginal_coverage": float(rect_covered.mean()),
        "standard_avg_set_size": float(standard_sizes.mean()),
        "rectified_avg_set_size": float(rect_sizes.mean()),
        "worst_slice_coverage_before": min(slice_coverages) if slice_coverages else np.nan,
        "worst_slice_coverage_after": min(rect_slice_coverages) if rect_slice_coverages else np.nan,
    }
    slice_rows = []
    for index, group in enumerate(groups):
        mask = anchor_memberships[:, index]
        if not mask.any():
            coverage_before = np.nan
            coverage_after = np.nan
        else:
            coverage_before = float(standard_covered[mask].mean())
            coverage_after = float(rect_covered[mask].mean())
        slice_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "discovery_seed": seed,
                "slice_index": index,
                "slice_name": group.name,
                "anchor_size": int(mask.sum()),
                "anchor_fraction": float(mask.mean()),
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
            }
        )
    return row, slice_rows


def _pairwise_rows(dataset: str, method: str, runs: list[SeedRun]) -> list[dict]:
    rows = []
    for left_index in range(len(runs)):
        for right_index in range(left_index + 1, len(runs)):
            left = runs[left_index]
            right = runs[right_index]
            anchor_similarity = _slice_set_similarity(left.anchor_memberships, right.anchor_memberships)
            discovery_similarity = _slice_set_similarity(left.discovery_memberships, right.discovery_memberships)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed_a": left.seed,
                    "seed_b": right.seed,
                    "anchor_union_jaccard": _jaccard(left.union_membership, right.union_membership),
                    "anchor_mean_best_match_jaccard": anchor_similarity["mean_best"],
                    "anchor_min_best_match_jaccard": anchor_similarity["min_best"],
                    "anchor_exact_match_rate": anchor_similarity["exact_rate"],
                    "discovery_mean_best_match_jaccard": discovery_similarity["mean_best"],
                    "discovery_min_best_match_jaccard": discovery_similarity["min_best"],
                }
            )
    return rows


def _slice_set_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape[1] == 0 and right.shape[1] == 0:
        return {"mean_best": 1.0, "min_best": 1.0, "exact_rate": 1.0}
    if left.shape[1] == 0 or right.shape[1] == 0:
        return {"mean_best": 0.0, "min_best": 0.0, "exact_rate": 0.0}
    similarities = np.empty((left.shape[1], right.shape[1]), dtype=np.float64)
    for i in range(left.shape[1]):
        for j in range(right.shape[1]):
            similarities[i, j] = _jaccard(left[:, i], right[:, j])
    best_left = similarities.max(axis=1)
    best_right = similarities.max(axis=0)
    all_best = np.concatenate([best_left, best_right])
    return {
        "mean_best": float(all_best.mean()),
        "min_best": float(all_best.min()),
        "exact_rate": float((all_best >= 0.95).mean()),
    }


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _summarize(dataset: str, method: str, run_rows: list[dict], pair_rows: list[dict]) -> dict:
    runs = pd.DataFrame(run_rows)
    pairs = pd.DataFrame(pair_rows)
    return {
        "dataset": dataset,
        "method": method,
        "num_seeds": len(runs),
        "num_slices_mean": runs["num_slices"].mean(),
        "num_slices_min": runs["num_slices"].min(),
        "num_slices_max": runs["num_slices"].max(),
        "anchor_union_fraction_mean": runs["anchor_union_fraction"].mean(),
        "anchor_union_fraction_std": runs["anchor_union_fraction"].std(ddof=1),
        "anchor_union_jaccard_mean": pairs["anchor_union_jaccard"].mean(),
        "anchor_union_jaccard_min": pairs["anchor_union_jaccard"].min(),
        "anchor_mean_best_match_jaccard": pairs["anchor_mean_best_match_jaccard"].mean(),
        "anchor_min_best_match_jaccard": pairs["anchor_min_best_match_jaccard"].min(),
        "anchor_exact_match_rate": pairs["anchor_exact_match_rate"].mean(),
        "rectified_marginal_coverage_mean": runs["rectified_marginal_coverage"].mean(),
        "rectified_marginal_coverage_std": runs["rectified_marginal_coverage"].std(ddof=1),
        "rectified_avg_set_size_mean": runs["rectified_avg_set_size"].mean(),
        "rectified_avg_set_size_std": runs["rectified_avg_set_size"].std(ddof=1),
        "worst_slice_coverage_after_mean": runs["worst_slice_coverage_after"].mean(),
        "worst_slice_coverage_after_std": runs["worst_slice_coverage_after"].std(ddof=1),
    }


def _write_markdown(summary: pd.DataFrame, runs: pd.DataFrame, pairs: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Within-method Discovery Seed Stability",
        "",
        "Only the discovery algorithm seed changes. Data splits, cached model outputs, embeddings, base conformal calibration, and rectification data remain fixed.",
        "",
        "## Summary",
        "",
        summary.fillna("").to_markdown(index=False),
        "",
        "## Per-seed Results",
        "",
        runs.fillna("").to_markdown(index=False),
        "",
        "## Pairwise Stability",
        "",
        pairs.fillna("").to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

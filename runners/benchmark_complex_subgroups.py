from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from core.conformal import conformal_quantile, covered, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_standard, fit_target_aware_gcp
from experiments import images as image_experiment
from reports.output import write_results
from runners.analyze_seed_stability import _image_args
from variants.consensus_latent.consensus import discover_consensus_groups
from variants.consensus_latent.run import _naive_docp_pool
from slices.discovery import class_conditional_groups, select_group_pool
from variants.ee_style.aggregation import aggregate_ee_slices


DEFAULT_DATASETS = ("ham10000", "nih", "mimic_effusion", "isic", "rsna_5k")
DEFAULT_SEEDS = tuple(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark discovery under complex hidden subgroup score shifts.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--shift-delta", type=float, default=0.12)
    parser.add_argument("--outdir", type=Path, default=Path("results/complex_subgroups"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results, recovery, groups = [], [], []
    for dataset in args.datasets:
        print(f"running complex subgroup benchmark: dataset={dataset}", flush=True)
        dataset_results, dataset_recovery, dataset_groups = run_dataset(dataset, args)
        results.append(dataset_results)
        recovery.append(dataset_recovery)
        groups.append(dataset_groups)
        dataset_dir = args.outdir / dataset
        write_results(dataset_results, dataset_dir / "summary.csv")
        write_results(dataset_recovery, dataset_dir / "recovery.csv")
        write_results(dataset_groups, dataset_dir / "hidden_groups.csv")
    all_results = pd.concat(results, ignore_index=True)
    all_recovery = pd.concat(recovery, ignore_index=True)
    all_groups = pd.concat(groups, ignore_index=True)
    write_results(all_results, args.outdir / "all_results.csv")
    write_results(all_recovery, args.outdir / "all_recovery.csv")
    write_results(all_groups, args.outdir / "all_hidden_groups.csv")
    aggregate = _aggregate(all_results)
    write_results(aggregate, args.outdir / "aggregate_summary.csv")
    (args.outdir / "config.json").write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "seeds": args.seeds,
                "split_seed": args.split_seed,
                "shift_delta": args.shift_delta,
                "hidden_groups": ["xor", "annulus", "disjoint_union"],
                "shifted_splits": ["discovery", "mvcal", "test"],
                "unshifted_splits": ["train", "calib0"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {args.outdir}", flush=True)


def run_dataset(dataset: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_args = _image_args(dataset, args.split_seed)
    data = image_experiment.load_experiment_data(image_args)
    probs, embeddings = image_experiment._load_feature_cache(image_args.cache_file, len(data.class_names), image_args.backbone)
    uncertainty = observable_uncertainty(probs)
    score_matrix = score_matrix_from_probs(probs, image_experiment._resolve_score(len(data.class_names)))
    hidden = _complex_hidden_groups(embeddings, uncertainty, data.discovery_idx)
    shifted_scores = _shift_scores(score_matrix, hidden, data, args.shift_delta)
    base_q = conformal_quantile(true_label_scores(shifted_scores[data.calib0_idx], data.labels[data.calib0_idx]), image_args.alpha)
    discovery_failures = ~covered(shifted_scores[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    inputs = image_experiment.DiscoveryInputs(
        embeddings=embeddings[data.discovery_idx],
        uncertainty=uncertainty[data.discovery_idx],
        coverage_failures=discovery_failures,
        classification_errors=probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx],
    )
    groups_by_seed = {}
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(image_args))
        seed_args.seed = seed
        groups_by_seed[seed] = image_experiment.discover_slice_method("coverage_domino", seed_args, inputs)

    _, ee_groups, _ = aggregate_ee_slices(
        groups_by_seed,
        embeddings[data.discovery_idx],
        uncertainty[data.discovery_idx],
        discovery_failures,
        embeddings[data.mvcal_idx],
        uncertainty[data.mvcal_idx],
        image_args.alpha,
        max_groups=image_args.pool_groups,
    )
    consensus_groups, _, _ = discover_consensus_groups(
        {("coverage_domino", seed): groups for seed, groups in groups_by_seed.items()},
        embeddings[data.discovery_idx],
        uncertainty[data.discovery_idx],
        discovery_failures,
        embeddings[data.mvcal_idx],
        uncertainty[data.mvcal_idx],
        image_args.alpha,
        effect_threshold=image_args.epsilon,
        max_groups=image_args.pool_groups,
    )
    docp_groups = _naive_docp_pool(image_args, inputs, embeddings, probs, data, discovery_failures)
    conditional_groups = select_group_pool(
        {"consensus": class_conditional_groups(consensus_groups, len(data.class_names))},
        embeddings[data.discovery_idx],
        probs[data.discovery_idx],
        discovery_failures,
        image_args.alpha,
        methods=("consensus",),
        min_size=image_args.min_size,
        epsilon=image_args.epsilon,
        enrichment=image_args.enrichment,
        max_groups=image_args.pool_groups,
    )
    mv_scores = true_label_scores(shifted_scores[data.mvcal_idx], data.labels[data.mvcal_idx])
    models = {
        "Standard CP": fit_standard(base_q),
        "DOCP": fit_batch_gcp(base_q, docp_groups, embeddings[data.mvcal_idx], mv_scores, image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size),
        "EE-TargetAware": fit_target_aware_gcp(base_q, ee_groups, embeddings[data.mvcal_idx], mv_scores, shifted_scores[data.mvcal_idx], image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size),
        "Consensus-Hard": fit_batch_gcp(base_q, consensus_groups, embeddings[data.mvcal_idx], mv_scores, image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size),
        "Consensus-Conditional": fit_batch_gcp(base_q, conditional_groups, embeddings[data.mvcal_idx], mv_scores, image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size),
        "Consensus-TargetAware": fit_target_aware_gcp(base_q, consensus_groups, embeddings[data.mvcal_idx], mv_scores, shifted_scores[data.mvcal_idx], image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size),
    }
    method_groups = {
        "Standard CP": [],
        "DOCP": docp_groups,
        "EE-TargetAware": ee_groups,
        "Consensus-Hard": consensus_groups,
        "Consensus-Conditional": conditional_groups,
        "Consensus-TargetAware": consensus_groups,
    }
    test = data.test_idx
    hidden_test = hidden[test]
    rows = []
    for name, model in models.items():
        row = evaluate_threshold_model(
            name,
            model,
            embeddings[test],
            shifted_scores[test],
            data.labels[test],
            hidden_test,
            data.known_group[test] if data.known_group is not None else None,
            docp_groups + ee_groups + consensus_groups,
            image_args.alpha,
            observed_scores=probs[test],
            eval_observed_scores=probs[test],
            own_groups=method_groups[name],
        )
        row.update(
            {
                "dataset": dataset,
                "target_abs_error": abs(float(row["marginal_coverage"]) - (1.0 - image_args.alpha)),
                "hidden_deficit": max(0.0, (1.0 - image_args.alpha) - float(row["worst_hidden_group_coverage"])),
                "repair_scale": getattr(model, "repair_scale", 1.0),
                "num_selected_slices": len(method_groups[name]),
            }
        )
        rows.append(row)
    recovery = []
    for name, selected in method_groups.items():
        if name == "Standard CP":
            continue
        for group_index in range(hidden.shape[1]):
            recovery.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "hidden_group": group_index,
                    "best_discovery_jaccard": _best_jaccard(selected, embeddings[data.discovery_idx], probs[data.discovery_idx], hidden[data.discovery_idx, group_index]),
                    "best_test_jaccard": _best_jaccard(selected, embeddings[test], probs[test], hidden_test[:, group_index]),
                }
            )
    group_rows = []
    for group_index, name in enumerate(["xor", "annulus", "disjoint_union"]):
        group_rows.append({"dataset": dataset, "hidden_group": group_index, "definition": name, "discovery_size": int(hidden[data.discovery_idx, group_index].sum()), "mvcal_size": int(hidden[data.mvcal_idx, group_index].sum()), "test_size": int(hidden[test, group_index].sum())})
    return pd.DataFrame(rows), pd.DataFrame(recovery), pd.DataFrame(group_rows)


def _complex_hidden_groups(embeddings: np.ndarray, uncertainty: np.ndarray, discovery_idx: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=3, random_state=0).fit(embeddings[discovery_idx])
    coordinates = pca.transform(embeddings)
    disc = coordinates[discovery_idx]
    q = lambda values, value: float(np.quantile(values, value))
    xor = np.logical_xor(coordinates[:, 0] > q(disc[:, 0], 0.50), coordinates[:, 1] > q(disc[:, 1], 0.50))
    xor &= uncertainty > q(uncertainty[discovery_idx], 0.75)
    radius = np.linalg.norm(coordinates[:, :2] - np.median(disc[:, :2], axis=0), axis=1)
    annulus = (radius >= q(radius[discovery_idx], 0.70)) & (radius <= q(radius[discovery_idx], 0.82))
    disjoint_union = ((coordinates[:, 0] > q(disc[:, 0], 0.85)) & (coordinates[:, 2] < q(disc[:, 2], 0.35))) | ((coordinates[:, 1] < q(disc[:, 1], 0.15)) & (uncertainty > q(uncertainty[discovery_idx], 0.75)))
    return np.column_stack([xor, annulus, disjoint_union])


def _shift_scores(score_matrix: np.ndarray, hidden: np.ndarray, data, delta: float) -> np.ndarray:
    shifted = score_matrix.copy()
    affected = hidden.any(axis=1)
    for indices in [data.discovery_idx, data.mvcal_idx, data.test_idx]:
        mask = affected[indices]
        shifted[indices[mask]] = np.clip(shifted[indices[mask]] + delta, 0.0, 1.0)
    return shifted


def _best_jaccard(groups, embeddings, scores, hidden) -> float:
    values = []
    for group in groups:
        membership = group.membership(embeddings, scores)
        union = np.logical_or(membership, hidden).sum()
        values.append(float(np.logical_and(membership, hidden).sum() / union) if union else 0.0)
    return max(values, default=0.0)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["marginal_coverage", "target_abs_error", "worst_hidden_group_coverage", "hidden_deficit", "worst_class_coverage", "worst_known_group_coverage", "avg_set_size", "repair_scale", "num_selected_slices"]
    result = frame.groupby("method", as_index=False)[metrics].mean(numeric_only=True)
    result.insert(0, "dataset", "mean_across_datasets")
    return result


if __name__ == "__main__":
    main()

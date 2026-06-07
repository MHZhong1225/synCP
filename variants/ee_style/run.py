from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.conformal import (
    classwise_quantiles,
    conformal_quantile,
    covered,
    groupwise_quantiles,
    observable_uncertainty,
    score_matrix_from_probs,
    true_label_scores,
)
from core.metrics import evaluate_classwise_model, evaluate_groupwise_model, evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_standard, fit_target_aware_gcp
from experiments import images as image_experiment
from reports.output import write_results
from runners.analyze_seed_stability import _image_args
from slices.discovery import GroupFunction, class_conditional_groups, select_group_pool
from variants.ee_style.aggregation import CandidateEvidence, aggregate_ee_slices


DEFAULT_DATASETS = ("ham10000", "nih", "mimic_effusion", "isic", "rsna_5k")
DEFAULT_SEEDS = tuple(range(1, 11))

METHOD_GROUPS = {
    "Standard CP": "Baseline",
    "Classwise CP": "Baseline",
    "Known-group CP": "Baseline",
    "DOCP": "Original",
    "EE-Frequency": "Ours",
    "EE-Score": "Ours",
    "EE-TargetAware": "Ours",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated EE-style slice aggregation experiments.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--frequency-threshold", type=float, default=0.50)
    parser.add_argument("--outdir", type=Path, default=Path("results/variants/ee_style"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    evidence_frames = []
    pool_frames = []
    for dataset in args.datasets:
        print(f"running EE-style aggregation: dataset={dataset}", flush=True)
        summary, evidence, pools = run_dataset(dataset, args)
        summaries.append(summary)
        evidence_frames.append(evidence)
        pool_frames.append(pools)
        dataset_dir = args.outdir / dataset
        write_results(summary, dataset_dir / "summary.csv")
        write_results(evidence, dataset_dir / "candidate_evidence.csv")
        write_results(pools, dataset_dir / "selected_slices.csv")
        _write_table(summary, dataset_dir / "baseline_vs_ours.md")

    all_results = pd.concat(summaries, ignore_index=True)
    all_evidence = pd.concat(evidence_frames, ignore_index=True)
    all_pools = pd.concat(pool_frames, ignore_index=True)
    write_results(all_results, args.outdir / "all_results.csv")
    write_results(all_evidence, args.outdir / "all_candidate_evidence.csv")
    write_results(all_pools, args.outdir / "all_selected_slices.csv")
    aggregate = _aggregate_results(all_results)
    write_results(aggregate, args.outdir / "aggregate_summary.csv")
    _write_table(all_results, args.outdir / "baseline_vs_ours.md")
    _write_table(aggregate, args.outdir / "aggregate_summary.md")
    (args.outdir / "config.json").write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "discovery_method": "coverage_domino",
                "discovery_seeds": args.seeds,
                "fixed_split_seed": args.split_seed,
                "jaccard_threshold": args.jaccard_threshold,
                "frequency_threshold": args.frequency_threshold,
                "anchor_split": "mvcal membership only",
                "rectifier": "batch_gcp",
                "max_groups": "dataset-specific DOCP pool_groups",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {args.outdir}", flush=True)


def run_dataset(dataset: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_args = _image_args(dataset, args.split_seed)
    data = image_experiment.load_experiment_data(image_args)
    probs, embeddings = image_experiment._load_feature_cache(
        image_args.cache_file,
        len(data.class_names),
        image_args.backbone,
    )
    score_matrix = score_matrix_from_probs(probs, image_experiment._resolve_score(len(data.class_names)))
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

    groups_by_seed = {}
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(image_args))
        seed_args.seed = seed
        groups_by_seed[seed] = image_experiment.discover_slice_method("coverage_domino", seed_args, inputs)

    frequency_groups, score_groups, evidence = aggregate_ee_slices(
        groups_by_seed,
        embeddings[data.discovery_idx],
        uncertainty[data.discovery_idx],
        discovery_failures,
        embeddings[data.mvcal_idx],
        uncertainty[data.mvcal_idx],
        image_args.alpha,
        jaccard_threshold=args.jaccard_threshold,
        frequency_threshold=args.frequency_threshold,
        max_groups=image_args.pool_groups,
    )
    naive_pool = _naive_docp_pool(image_args, inputs, embeddings, probs, data, discovery_failures)
    shared_groups = _shared_groups(naive_pool, frequency_groups, score_groups)
    summary = _evaluate_methods(
        dataset,
        image_args,
        data,
        embeddings,
        probs,
        score_matrix,
        uncertainty,
        base_q,
        shared_groups,
        naive_pool,
        frequency_groups,
        score_groups,
    )
    evidence_frame = _evidence_frame(dataset, evidence, frequency_groups, score_groups)
    pool_frame = _pool_frame(dataset, naive_pool, frequency_groups, score_groups, embeddings[data.test_idx], probs[data.test_idx])
    return summary, evidence_frame, pool_frame


def _naive_docp_pool(
    image_args: argparse.Namespace,
    inputs: image_experiment.DiscoveryInputs,
    embeddings: np.ndarray,
    probs: np.ndarray,
    data,
    discovery_failures: np.ndarray,
) -> list[GroupFunction]:
    seed_args = argparse.Namespace(**vars(image_args))
    discovery_outputs = {
        method: image_experiment.discover_slice_method(method, seed_args, inputs)
        for method in image_experiment.DISCOVERY_METHODS
    }
    source_outputs = {
        method: class_conditional_groups(groups, len(data.class_names))
        for method, groups in discovery_outputs.items()
    }
    return select_group_pool(
        source_outputs,
        embeddings[data.discovery_idx],
        probs[data.discovery_idx],
        discovery_failures,
        image_args.alpha,
        methods=tuple(image_experiment.POOL_METHODS),
        min_size=image_args.min_size,
        epsilon=image_args.epsilon,
        enrichment=image_args.enrichment,
        max_groups=image_args.pool_groups,
    )


def _evaluate_methods(
    dataset: str,
    image_args: argparse.Namespace,
    data,
    embeddings: np.ndarray,
    probs: np.ndarray,
    score_matrix: np.ndarray,
    uncertainty: np.ndarray,
    base_q: float,
    shared_groups: list[GroupFunction],
    naive_pool: list[GroupFunction],
    frequency_groups: list[GroupFunction],
    score_groups: list[GroupFunction],
) -> pd.DataFrame:
    test = data.test_idx
    known_test = data.known_group[test] if data.known_group is not None else None
    rows = []
    rows.append(
        evaluate_threshold_model(
            "Standard CP",
            fit_standard(base_q),
            embeddings[test],
            score_matrix[test],
            data.labels[test],
            None,
            known_test,
            shared_groups,
            image_args.alpha,
            observed_scores=probs[test],
            eval_observed_scores=probs[test],
        )
    )
    class_q = classwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], image_args.alpha)
    rows.append(
        evaluate_classwise_model(
            "Classwise CP",
            class_q,
            embeddings[test],
            score_matrix[test],
            data.labels[test],
            None,
            known_test,
            shared_groups,
            image_args.alpha,
            observed_scores=probs[test],
            eval_observed_scores=probs[test],
        )
    )
    if data.known_group is not None:
        known_q = groupwise_quantiles(
            score_matrix[data.calib0_idx],
            data.labels[data.calib0_idx],
            data.known_group[data.calib0_idx],
            image_args.alpha,
        )
        rows.append(
            evaluate_groupwise_model(
                "Known-group CP",
                known_q,
                embeddings[test],
                score_matrix[test],
                data.labels[test],
                data.known_group[test],
                None,
                known_test,
                shared_groups,
                image_args.alpha,
                observed_scores=probs[test],
                eval_observed_scores=probs[test],
            )
        )
    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    for name, groups in [
        ("DOCP", naive_pool),
        ("EE-Frequency", frequency_groups),
        ("EE-Score", score_groups),
    ]:
        rectifier = fit_batch_gcp(
            base_q,
            groups,
            embeddings[data.mvcal_idx],
            mv_scores,
            image_args.alpha,
            observed_scores=probs[data.mvcal_idx],
            min_size=image_args.rectifier_min_size,
        )
        rows.append(
            evaluate_threshold_model(
                name,
                rectifier,
                embeddings[test],
                score_matrix[test],
                data.labels[test],
                None,
                known_test,
                shared_groups,
                image_args.alpha,
                observed_scores=probs[test],
                eval_observed_scores=probs[test],
                own_groups=groups,
            )
        )
    target_aware = fit_target_aware_gcp(
        base_q,
        score_groups,
        embeddings[data.mvcal_idx],
        mv_scores,
        score_matrix[data.mvcal_idx],
        image_args.alpha,
        observed_scores=probs[data.mvcal_idx],
        min_size=image_args.rectifier_min_size,
    )
    rows.append(
        evaluate_threshold_model(
            "EE-TargetAware",
            target_aware,
            embeddings[test],
            score_matrix[test],
            data.labels[test],
            None,
            known_test,
            shared_groups,
            image_args.alpha,
            observed_scores=probs[test],
            eval_observed_scores=probs[test],
            own_groups=score_groups,
        )
    )
    frame = pd.DataFrame(rows)
    frame.insert(0, "dataset", dataset)
    frame.insert(1, "group", frame["method"].map(METHOD_GROUPS))
    frame["num_selected_slices"] = frame["method"].map(
        {"DOCP": len(naive_pool), "EE-Frequency": len(frequency_groups), "EE-Score": len(score_groups), "EE-TargetAware": len(score_groups)}
    )
    frame["repair_scale"] = frame["method"].map({"EE-TargetAware": target_aware.repair_scale})
    return frame


def _shared_groups(*pools: list[GroupFunction]) -> list[GroupFunction]:
    groups = []
    seen = set()
    for pool in pools:
        for group in pool:
            identity = id(group)
            if identity in seen:
                continue
            seen.add(identity)
            groups.append(group)
    return groups


def _evidence_frame(
    dataset: str,
    evidence: list[CandidateEvidence],
    frequency_groups: list[GroupFunction],
    score_groups: list[GroupFunction],
) -> pd.DataFrame:
    frequency_ids = {id(group) for group in frequency_groups}
    score_ids = {id(group) for group in score_groups}
    rows = []
    for item in evidence:
        rows.append(
            {
                "dataset": dataset,
                "source_seed": item.source_seed,
                "source_index": item.source_index,
                "slice_name": item.group.name,
                "frequency": item.frequency,
                "median_gap": item.median_gap,
                "discovery_size": item.discovery_size,
                "anchor_size": item.anchor_size,
                "ee_score": item.ee_score,
                "selected_frequency": id(item.group) in frequency_ids,
                "selected_score": id(item.group) in score_ids,
            }
        )
    return pd.DataFrame(rows)


def _pool_frame(
    dataset: str,
    naive_pool: list[GroupFunction],
    frequency_groups: list[GroupFunction],
    score_groups: list[GroupFunction],
    test_embeddings: np.ndarray,
    test_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for method, groups in [("DOCP", naive_pool), ("EE-Frequency", frequency_groups), ("EE-Score", score_groups)]:
        for index, group in enumerate(groups):
            mask = group.membership(test_embeddings, test_scores)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "slice_index": index,
                    "slice_name": group.name,
                    "test_size": int(mask.sum()),
                    "test_fraction": float(mask.mean()),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "marginal_coverage",
        "worst_class_coverage",
        "worst_known_group_coverage",
        "worst_shared_discovered_slice_coverage",
        "worst_own_pool_coverage",
        "avg_set_size",
        "max_undercoverage_gap",
        "num_selected_slices",
        "repair_scale",
    ]
    aggregate = frame.groupby(["group", "method"], as_index=False)[metrics].mean(numeric_only=True)
    aggregate.insert(0, "dataset", "mean_across_datasets")
    return aggregate


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "group",
        "method",
        "marginal_coverage",
        "worst_class_coverage",
        "worst_known_group_coverage",
        "worst_shared_discovered_slice_coverage",
        "worst_own_pool_coverage",
        "avg_set_size",
        "max_undercoverage_gap",
        "num_selected_slices",
        "repair_scale",
    ]
    existing = [column for column in columns if column in frame.columns]
    path.write_text(frame[existing].round(5).fillna("").to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()

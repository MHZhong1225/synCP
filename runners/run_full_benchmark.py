from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.conformal import classwise_quantiles, conformal_quantile, covered, groupwise_quantiles, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_classwise_model, evaluate_groupwise_model, evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_standard, fit_target_aware_gcp
from data.mimic_care import load_mimic_care_experiment
from experiments import images as image_experiment
from reports.output import write_results
from runners.analyze_seed_stability import _image_args
from runners.run_nih_ovr import NIH_TARGETS, target_suffix
from slices.discovery import class_conditional_groups, select_group_pool
from variants.consensus_latent.consensus import discover_consensus_groups
from variants.ee_style.aggregation import aggregate_ee_slices


DEFAULT_SEEDS = tuple(range(1, 11))
IMAGE_TASKS = ("bracs", "ham10000", "isic", "rsna_5k")
METHODS = ("Standard CP", "Classwise CP", "Known-group CP", "DOCP", "EE-Score", "EE-TargetAware", "Consensus-Hard", "Consensus-TargetAware")
CORE_RESULT_COLUMNS = (
    "method",
    "marginal_coverage",
    "marginal_undercoverage",
    "avg_set_size",
    "worst_class_coverage",
    "worst_known_group_coverage",
    "max_undercoverage_gap",
    "worst_hidden_group_coverage",
    "task",
    "family",
    "num_selected_slices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete DOCP benchmark with per-method runtimes.")
    parser.add_argument("--tasks", nargs="+", default=["bracs", "nih_all", "isic", "ham10000", "rsna_5k", "mimic_care"])
    parser.add_argument("--nih-targets", nargs="+", default=NIH_TARGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=Path("results/full_benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    task_specs = _expand_tasks(args.tasks, args.nih_targets)
    result_frames, timing_frames = [], []
    for task_name, family in task_specs:
        print(f"running full benchmark: task={task_name}", flush=True)
        results, timings = run_task(task_name, family, args)
        result_frames.append(results)
        timing_frames.append(timings)
        task_dir = args.outdir / task_name
        write_results(results, task_dir / "results.csv")
        write_results(timings, task_dir / "runtimes.csv")
    all_results = _order_result_columns(pd.concat(result_frames, ignore_index=True))
    all_timings = pd.concat(timing_frames, ignore_index=True)
    write_results(all_results, args.outdir / "all_results.csv")
    write_results(all_timings, args.outdir / "all_runtimes.csv")
    write_results(_aggregate_results(all_results), args.outdir / "aggregate_results.csv")
    write_results(_aggregate_timings(all_timings), args.outdir / "aggregate_runtimes.csv")
    write_results(_family_results(all_results), args.outdir / "family_results.csv")
    write_results(_family_timings(all_timings), args.outdir / "family_runtimes.csv")
    (args.outdir / "config.json").write_text(
        json.dumps({"tasks": task_specs, "discovery_seeds": args.seeds, "split_seed": args.split_seed, "feature_extraction": "cached; excluded from method runtime", "runtime_unit": "seconds"}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote: {args.outdir}", flush=True)


def run_task(task_name: str, family: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_args, data, probs, embeddings, score_name = _load_task(task_name, family, args.split_seed)
    score_matrix = score_matrix_from_probs(probs, score_name)
    uncertainty = observable_uncertainty(probs)
    base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), task_args.alpha)
    discovery_failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    inputs = image_experiment.DiscoveryInputs(embeddings[data.discovery_idx], uncertainty[data.discovery_idx], discovery_failures, probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx])

    start = perf_counter()
    outputs = {method: image_experiment.discover_slice_method(method, task_args, inputs) for method in image_experiment.DISCOVERY_METHODS}
    pool_sources = {method: class_conditional_groups(groups, len(data.class_names)) for method, groups in outputs.items()}
    docp_groups = select_group_pool(pool_sources, embeddings[data.discovery_idx], probs[data.discovery_idx], discovery_failures, task_args.alpha, methods=tuple(image_experiment.POOL_METHODS), min_size=task_args.min_size, epsilon=task_args.epsilon, enrichment=task_args.enrichment, max_groups=task_args.pool_groups)
    docp_discovery_seconds = perf_counter() - start

    start = perf_counter()
    groups_by_seed = {}
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(task_args))
        seed_args.seed = seed
        groups_by_seed[seed] = image_experiment.discover_slice_method("coverage_domino", seed_args, inputs)
    coverage_discovery_seconds = perf_counter() - start

    start = perf_counter()
    _, ee_groups, _ = aggregate_ee_slices(groups_by_seed, embeddings[data.discovery_idx], uncertainty[data.discovery_idx], discovery_failures, embeddings[data.mvcal_idx], uncertainty[data.mvcal_idx], task_args.alpha, max_groups=task_args.pool_groups)
    ee_aggregation_seconds = perf_counter() - start

    start = perf_counter()
    consensus_groups, _, _ = discover_consensus_groups({("coverage_domino", seed): groups for seed, groups in groups_by_seed.items()}, embeddings[data.discovery_idx], uncertainty[data.discovery_idx], discovery_failures, embeddings[data.mvcal_idx], uncertainty[data.mvcal_idx], task_args.alpha, effect_threshold=task_args.epsilon, max_groups=task_args.pool_groups)
    consensus_aggregation_seconds = perf_counter() - start

    shared_groups = docp_groups + ee_groups + consensus_groups
    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    test = data.test_idx
    known_test = data.known_group[test] if data.known_group is not None else None
    rows, timings = [], []

    def record(method, discovery_seconds, aggregation_seconds, fit_fn, eval_fn, num_slices):
        fit_start = perf_counter()
        fitted = fit_fn()
        fit_seconds = perf_counter() - fit_start
        eval_start = perf_counter()
        row = eval_fn(fitted)
        evaluation_seconds = perf_counter() - eval_start
        row.update({"task": task_name, "family": family, "num_selected_slices": num_slices})
        rows.append(row)
        timings.append({"task": task_name, "family": family, "method": method, "discovery_seconds": discovery_seconds, "aggregation_seconds": aggregation_seconds, "rectification_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds, "total_method_seconds": discovery_seconds + aggregation_seconds + fit_seconds + evaluation_seconds})

    threshold_eval = lambda method, own: lambda model: evaluate_threshold_model(method, model, embeddings[test], score_matrix[test], data.labels[test], None, known_test, shared_groups, task_args.alpha, observed_scores=probs[test], eval_observed_scores=probs[test], own_groups=own)
    record("Standard CP", 0.0, 0.0, lambda: fit_standard(base_q), threshold_eval("Standard CP", None), 0)
    record("Classwise CP", 0.0, 0.0, lambda: classwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], task_args.alpha), lambda q: evaluate_classwise_model("Classwise CP", q, embeddings[test], score_matrix[test], data.labels[test], None, known_test, shared_groups, task_args.alpha, observed_scores=probs[test], eval_observed_scores=probs[test]), 0)
    if data.known_group is not None:
        record("Known-group CP", 0.0, 0.0, lambda: groupwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], data.known_group[data.calib0_idx], task_args.alpha), lambda q: evaluate_groupwise_model("Known-group CP", q, embeddings[test], score_matrix[test], data.labels[test], data.known_group[test], None, known_test, shared_groups, task_args.alpha, observed_scores=probs[test], eval_observed_scores=probs[test]), 0)
    record("DOCP", docp_discovery_seconds, 0.0, lambda: fit_batch_gcp(base_q, docp_groups, embeddings[data.mvcal_idx], mv_scores, task_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=task_args.rectifier_min_size), threshold_eval("DOCP", docp_groups), len(docp_groups))
    record("EE-Score", coverage_discovery_seconds, ee_aggregation_seconds, lambda: fit_batch_gcp(base_q, ee_groups, embeddings[data.mvcal_idx], mv_scores, task_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=task_args.rectifier_min_size), threshold_eval("EE-Score", ee_groups), len(ee_groups))
    record("EE-TargetAware", coverage_discovery_seconds, ee_aggregation_seconds, lambda: fit_target_aware_gcp(base_q, ee_groups, embeddings[data.mvcal_idx], mv_scores, score_matrix[data.mvcal_idx], task_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=task_args.rectifier_min_size), threshold_eval("EE-TargetAware", ee_groups), len(ee_groups))
    record("Consensus-Hard", coverage_discovery_seconds, consensus_aggregation_seconds, lambda: fit_batch_gcp(base_q, consensus_groups, embeddings[data.mvcal_idx], mv_scores, task_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=task_args.rectifier_min_size), threshold_eval("Consensus-Hard", consensus_groups), len(consensus_groups))
    record("Consensus-TargetAware", coverage_discovery_seconds, consensus_aggregation_seconds, lambda: fit_target_aware_gcp(base_q, consensus_groups, embeddings[data.mvcal_idx], mv_scores, score_matrix[data.mvcal_idx], task_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=task_args.rectifier_min_size), threshold_eval("Consensus-TargetAware", consensus_groups), len(consensus_groups))
    return _format_result_columns(pd.DataFrame(rows), task_name, family), pd.DataFrame(timings)


def _load_task(task_name: str, family: str, split_seed: int):
    if family == "mimic_care":
        args = argparse.Namespace(alpha=0.10, components=24, min_size=40, rectifier_min_size=20, pool_groups=8, epsilon=0.02, enrichment=1.10, seed=split_seed)
        data = load_mimic_care_experiment(Path("../dataset/mimic-iv-3.1"), Path("../dataset/mimic-iv-ed-2.2"), known_group="race_acuity", feature_mode="text_triage", seed=split_seed)
        cache = np.load("results/mimic_care/features.npz")
        normalized = SimpleNamespace(labels=data.labels, class_names=data.class_names, calib0_idx=data.calib0_idx, discovery_idx=data.discovery_idx, mvcal_idx=data.mvcal_idx, test_idx=data.test_idx, known_group=data.known_group)
        return args, normalized, cache["probs"], cache["embeddings"], "inverse_probability"
    preset = "nih" if family == "nih" else task_name
    task_args = _image_args(preset, split_seed)
    if family == "nih":
        target = task_name.removeprefix("nih_").replace("_", " ").title().replace("Pleural Thickening", "Pleural_Thickening")
        task_args.nih_target = target
        task_args.cache_file = Path("results") / task_name / "features.npz"
    data = image_experiment.load_experiment_data(task_args)
    probs, embeddings = image_experiment._load_feature_cache(task_args.cache_file, len(data.class_names), task_args.backbone)
    return task_args, data, probs, embeddings, image_experiment._resolve_score(len(data.class_names))


def _expand_tasks(tasks: list[str], nih_targets: list[str]) -> list[tuple[str, str]]:
    expanded = []
    for task in tasks:
        if task == "nih_all":
            expanded.extend((f"nih_{target_suffix(target)}", "nih") for target in nih_targets)
        elif task == "mimic_care":
            expanded.append((task, "mimic_care"))
        elif task in IMAGE_TASKS:
            expanded.append((task, task))
        else:
            raise ValueError(f"Unknown task: {task}")
    return expanded


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in frame.columns if column not in {"method", "task", "family"}]
    return frame.groupby("method", as_index=False)[metrics].mean(numeric_only=True)


def _aggregate_timings(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["discovery_seconds", "aggregation_seconds", "rectification_seconds", "evaluation_seconds", "total_method_seconds"]
    grouped = frame.groupby("method")[metrics].agg(["mean", "median", "sum"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _family_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in frame.columns if column not in {"method", "task", "family"}]
    return frame.groupby(["family", "method"], as_index=False)[metrics].mean(numeric_only=True)


def _format_result_columns(frame: pd.DataFrame, task_name: str, family: str) -> pd.DataFrame:
    attribute = _known_group_attribute(task_name, family)
    generic_prefix = "known_group_coverage_group_"
    if attribute is not None:
        frame = frame.rename(
            columns={
                column: column.replace(generic_prefix, f"known_group_coverage_{attribute}_group_", 1)
                for column in frame.columns
                if column.startswith(generic_prefix)
            }
        )
    return _order_result_columns(frame)


def _order_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    known_group_columns = sorted(column for column in frame.columns if column.startswith("known_group_coverage_"))
    columns = [column for column in CORE_RESULT_COLUMNS[:5] if column in frame]
    columns.extend(known_group_columns)
    columns.extend(column for column in CORE_RESULT_COLUMNS[5:] if column in frame)
    return frame.loc[:, columns]


def _known_group_attribute(task_name: str, family: str) -> str | None:
    if family == "nih":
        return "gender"
    return {
        "ham10000": "sex",
        "isic": "source",
        "mimic_care": "race_acuity",
        "rsna_5k": "site_laterality_view",
    }.get(task_name)


def _family_timings(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["family", "method"])["total_method_seconds"].agg(["mean", "median", "sum"])
    return grouped.rename(columns={"mean": "total_seconds_mean", "median": "total_seconds_median", "sum": "total_seconds_sum"}).reset_index()


if __name__ == "__main__":
    main()

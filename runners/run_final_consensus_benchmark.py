from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from core.consensus import discover_consensus_groups
from core.conformal import classwise_quantiles, conformal_quantile, covered, groupwise_quantiles, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_classwise_model, evaluate_groupwise_model, evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_standard
from experiments import images as image_experiment
from reports.final_benchmark_report import write_final_benchmark_report
from reports.main_tables_by_dataset import write_main_tables
from reports.output import round_numbers
from reports.subgroup_fairness import write_subgroup_reports
from runners.benchmark_tasks import NIH_TARGETS, expand_tasks, format_result_columns, load_task


METHOD_SEEDS = {
    "cp_score_kmeans": tuple(range(1, 11)),
}
DISPLAY_NAMES = {
    "cp_score_kmeans": "Consensus-CPScore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final consensus benchmark across all datasets.")
    parser.add_argument("--tasks", nargs="+", default=["nih_all", "isic", "ham10000", "rsna_5k", "mimic_care", "mimic_cxr_all"])
    parser.add_argument("--nih-targets", nargs="+", default=NIH_TARGETS)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--cache-root", type=Path, default=Path("results/npz"), help="Directory containing cached .npz files for this full-system run.")
    parser.add_argument("--outdir", type=Path, default=Path("results/final_consensus_benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results, runtimes = [], []
    for task, family in expand_tasks(args.tasks, args.nih_targets):
        print(f"running final consensus benchmark: task={task}", flush=True)
        task_results, task_runtimes = run_task(task, family, args.split_seed, args.cache_root)
        results.append(task_results)
        runtimes.append(task_runtimes)
        _write(task_results, args.outdir / task / "results.csv")
        _write(task_runtimes, args.outdir / task / "runtimes.csv")

    all_results = pd.concat(results, ignore_index=True)
    all_runtimes = pd.concat(runtimes, ignore_index=True)
    _write(all_results, args.outdir / "all_results.csv")
    _write(all_runtimes, args.outdir / "all_runtimes.csv")
    _write(_aggregate_results(all_results), args.outdir / "aggregate_results.csv")
    _write(_family_results(all_results), args.outdir / "family_results.csv")
    _write(_aggregate_runtimes(all_runtimes), args.outdir / "aggregate_runtimes.csv")
    _write_config(args)
    write_subgroup_reports(args.outdir / "all_results.csv", args.outdir, target_coverage=0.90)
    write_final_benchmark_report(args.outdir)
    write_main_tables(args.outdir / "all_results.csv", args.outdir / "main_tables_by_dataset")


def run_task(
    task: str,
    family: str,
    split_seed: int,
    cache_root: Path = Path("results/npz"),
    score_override: str | None = None,
    raps_penalty: float = 0.01,
    raps_regularization_rank: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_args, data, probs, embeddings, score_name = load_task(task, family, split_seed, cache_root=cache_root)
    score_name = score_override or score_name
    score_matrix = score_matrix_from_probs(
        probs,
        score_name,
        random_state=task_args.seed,
        raps_penalty=raps_penalty,
        raps_regularization_rank=raps_regularization_rank,
    )
    uncertainty = observable_uncertainty(probs)
    base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), task_args.alpha)
    failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    inputs = image_experiment.DiscoveryInputs(
        embeddings[data.discovery_idx],
        uncertainty[data.discovery_idx],
        failures,
        probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx],
        probs[data.discovery_idx],
    )
    groups_by_method, timing_parts = _discover_consensus(task_args, data, probs, embeddings, failures, inputs)
    rows, runtimes = _evaluate(task, family, task_args, data, probs, embeddings, score_matrix, base_q, groups_by_method, timing_parts)
    return format_result_columns(pd.DataFrame(rows), task, family), pd.DataFrame(runtimes)


def _discover_consensus(task_args, data, probs, embeddings, failures, inputs):
    groups_by_method = {}
    timing_parts = {}
    for method, seeds in METHOD_SEEDS.items():
        discovered = {}
        start = perf_counter()
        for seed in seeds:
            seed_args = argparse.Namespace(**vars(task_args))
            seed_args.seed = seed
            discovered[(method, seed)] = image_experiment.discover_slice_method(method, seed_args, inputs)
        discovery_seconds = perf_counter() - start
        start = perf_counter()
        groups, _, _ = discover_consensus_groups(
            discovered,
            embeddings[data.discovery_idx],
            probs[data.discovery_idx],
            failures,
            embeddings[data.mvcal_idx],
            probs[data.mvcal_idx],
            task_args.alpha,
            effect_threshold=task_args.epsilon,
            max_groups=task_args.pool_groups,
        )
        groups_by_method[method] = groups
        timing_parts[method] = {"discovery_seconds": discovery_seconds, "aggregation_seconds": perf_counter() - start}
    return groups_by_method, timing_parts


def _evaluate(task, family, task_args, data, probs, embeddings, score_matrix, base_q, groups_by_method, timing_parts):
    test = data.test_idx
    known_test = data.known_group[test] if data.known_group is not None else None
    sensitive_groups = getattr(data, "sensitive_groups", None) or {}
    sensitive_test = {name: values[test] for name, values in sensitive_groups.items()}
    fixed_subgroup_name, fixed_subgroup_mask = _fixed_subgroup(task, family, sensitive_test)
    shared_groups = [group for groups in groups_by_method.values() for group in groups]
    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    rows, runtimes = [], []

    def record(method, discovery_seconds, aggregation_seconds, fit_fn, eval_fn, num_slices):
        fit_start = perf_counter()
        fitted = fit_fn()
        fit_seconds = perf_counter() - fit_start
        eval_start = perf_counter()
        row = eval_fn(fitted)
        eval_seconds = perf_counter() - eval_start
        row.update({"task": task, "family": family, "num_selected_slices": num_slices})
        rows.append(row)
        runtimes.append(
            {
                "task": task,
                "family": family,
                "method": method,
                "discovery_seconds": discovery_seconds,
                "aggregation_seconds": aggregation_seconds,
                "calibration_seconds": fit_seconds,
                "evaluation_seconds": eval_seconds,
                "total_seconds": discovery_seconds + aggregation_seconds + fit_seconds + eval_seconds,
            }
        )

    def threshold_eval(method, own_groups):
        return lambda model: evaluate_threshold_model(
            method,
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
            own_groups=own_groups,
            sensitive_groups=sensitive_test,
            fixed_subgroup_mask=fixed_subgroup_mask,
            fixed_subgroup_name=fixed_subgroup_name,
        )

    record("Standard CP", 0.0, 0.0, lambda: fit_standard(base_q), threshold_eval("Standard CP", None), 0)
    record(
        "Classwise CP",
        0.0,
        0.0,
        lambda: classwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], task_args.alpha),
        lambda q: evaluate_classwise_model(
            "Classwise CP",
            q,
            embeddings[test],
            score_matrix[test],
            data.labels[test],
            None,
            known_test,
            shared_groups,
            task_args.alpha,
            observed_scores=probs[test],
            eval_observed_scores=probs[test],
            sensitive_groups=sensitive_test,
            fixed_subgroup_mask=fixed_subgroup_mask,
            fixed_subgroup_name=fixed_subgroup_name,
        ),
        0,
    )
    if sensitive_groups:
        for attribute, raw_groups in sensitive_groups.items():
            encoded_groups = _encode_groups(raw_groups)
            method_name = f"Known-group CP ({attribute})"
            record(
                method_name,
                0.0,
                0.0,
                lambda groups=encoded_groups: groupwise_quantiles(
                    score_matrix[data.calib0_idx],
                    data.labels[data.calib0_idx],
                    groups[data.calib0_idx],
                    task_args.alpha,
                ),
                lambda q, groups=encoded_groups, name=method_name: evaluate_groupwise_model(
                    name,
                    q,
                    embeddings[test],
                    score_matrix[test],
                    data.labels[test],
                    groups[test],
                    None,
                    known_test,
                    shared_groups,
                    task_args.alpha,
                    observed_scores=probs[test],
                    eval_observed_scores=probs[test],
                    sensitive_groups=sensitive_test,
                    fixed_subgroup_mask=fixed_subgroup_mask,
                    fixed_subgroup_name=fixed_subgroup_name,
                ),
                0,
            )
    elif data.known_group is not None:
        record(
            "Known-group CP",
            0.0,
            0.0,
            lambda: groupwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], data.known_group[data.calib0_idx], task_args.alpha),
            lambda q: evaluate_groupwise_model(
                "Known-group CP",
                q,
                embeddings[test],
                score_matrix[test],
                data.labels[test],
                data.known_group[test],
                None,
                known_test,
                shared_groups,
                task_args.alpha,
                observed_scores=probs[test],
                eval_observed_scores=probs[test],
                fixed_subgroup_mask=fixed_subgroup_mask,
                fixed_subgroup_name=fixed_subgroup_name,
            ),
            0,
        )
    for method, groups in groups_by_method.items():
        timing = timing_parts[method]
        display_name = DISPLAY_NAMES[method]
        record(
            display_name,
            timing["discovery_seconds"],
            timing["aggregation_seconds"],
            lambda groups=groups: fit_batch_gcp(base_q, groups, embeddings[data.mvcal_idx], mv_scores, task_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=task_args.rectifier_min_size),
            threshold_eval(display_name, groups),
            len(groups),
        )
    return rows, runtimes



def _fixed_subgroup(task: str, family: str, sensitive_groups: dict[str, np.ndarray]) -> tuple[str, np.ndarray | None]:
    if not sensitive_groups:
        return "", None
    if task == "ham10000":
        return _mask_name("male & age_gt_70", sensitive_groups, sex="male", age="age_gt_70")
    if task == "isic":
        return _mask_name(
            "male & age_gt_70 & head/neck",
            sensitive_groups,
            sex="male",
            age_approx="age_gt_70",
            anatom_site_general="head/neck",
        )
    if task == "rsna_5k":
        return _mask_name("age_gt_70", sensitive_groups, age="age_gt_70")
    if family == "nih":
        return _mask_name("M & age_gt_60", sensitive_groups, gender="M", age="age_gt_60")
    if family == "mimic_cxr":
        return _mask_name("BLACK & acuity_1_2", sensitive_groups, race="BLACK", ed_acuity="acuity_1_2")
    if task == "mimic_care":
        return _mask_name("BLACK & acuity_1_2", sensitive_groups, race="BLACK", acuity_group="acuity_1_2")
    return "", None


def _mask_name(name: str, sensitive_groups: dict[str, np.ndarray], **criteria: str) -> tuple[str, np.ndarray | None]:
    mask = None
    for attribute, expected in criteria.items():
        values = sensitive_groups.get(attribute)
        if values is None:
            return name, None
        current = np.asarray(values).astype(str) == expected
        mask = current if mask is None else mask & current
    return name, mask

def _encode_groups(values):
    return pd.Series(pd.factorize(values, sort=True)[0]).to_numpy()


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in frame if column not in {"method", "task", "family"}]
    return frame.groupby("method", as_index=False)[metrics].mean(numeric_only=True)


def _family_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in frame if column not in {"method", "task", "family"}]
    return frame.groupby(["family", "method"], as_index=False)[metrics].mean(numeric_only=True)


def _aggregate_runtimes(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["discovery_seconds", "aggregation_seconds", "calibration_seconds", "evaluation_seconds", "total_seconds"]
    return frame.groupby("method", as_index=False)[metrics].mean(numeric_only=True)



def _write_config(args: argparse.Namespace) -> None:
    config = {
        "tasks": args.tasks,
        "nih_targets": args.nih_targets,
        "method_seeds": {method: list(seeds) for method, seeds in METHOD_SEEDS.items()},
        "split_seed": args.split_seed,
        "score_policy": "multiclass randomized APS; binary inverse probability",
        "cache_root": str(args.cache_root),
        "control": "fixed cached features; final method is Consensus-CPScore using CP-score-augmented discovery",
    }
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numbers(frame).to_csv(path, index=False, float_format="%.5f")


if __name__ == "__main__":
    main()

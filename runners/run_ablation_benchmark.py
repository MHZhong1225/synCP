
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from core.consensus import discover_consensus_groups
from core.conformal import conformal_quantile, covered, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_slice_quantile, fit_standard
from experiments import images as image_experiment
from reports.final_benchmark_report import DATASET_NAMES, SUMMARY_METRICS, _method_order
from reports.output import round_numbers
from runners.benchmark_tasks import NIH_TARGETS, expand_tasks, format_result_columns, load_task
from runners.run_final_consensus_benchmark import _fixed_subgroup


@dataclass(frozen=True)
class AblationVariant:
    name: str
    discovery_method: str | None
    discovery_seeds: tuple[int, ...]
    use_consensus: bool
    rectifier: str


VARIANTS = (
    AblationVariant("No local repair", None, tuple(), False, "standard"),
    AblationVariant("Single-seed CPScore", "cp_score_kmeans", (1,), False, "batch_gcp"),
    AblationVariant("Consensus K-means", "kmeans", tuple(range(1, 11)), True, "batch_gcp"),
    AblationVariant("Consensus CPScore + slice quantile", "cp_score_kmeans", tuple(range(1, 11)), True, "slice_quantile"),
    AblationVariant("Full Consensus-CPScore", "cp_score_kmeans", tuple(range(1, 11)), True, "batch_gcp"),
)
DISPLAY_ORDER = {variant.name: index for index, variant in enumerate(VARIANTS)}
ABLATION_METRICS = [
    "marginal_coverage",
    "avg_set_size",
    "worst_class_coverage",
    "worst_known_group_coverage",
    "known_group_mean_undercoverage_gap",
    "max_undercoverage_gap",
    "num_selected_slices",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final Consensus-CPScore component ablation from cached repeat npz files.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--tasks", nargs="+", default=["nih_all", "isic", "ham10000", "rsna_5k", "mimic_care", "mimic_cxr_all"])
    parser.add_argument("--nih-targets", nargs="+", default=NIH_TARGETS)
    parser.add_argument("--repeat-cache-root", type=Path, default=Path("results/npz/repeats"))
    parser.add_argument("--outdir", type=Path, default=Path("results/final_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path = args.outdir / "all_results_by_seed.csv"
    runtimes_path = args.outdir / "all_runtimes_by_seed.csv"
    temp_dir = args.outdir / "_seed_tmp"
    if temp_dir.exists():
        for path in temp_dir.glob("*.csv"):
            path.unlink()
    temp_dir.mkdir(parents=True, exist_ok=True)
    for path in (results_path, runtimes_path):
        if path.exists():
            path.unlink()
    tasks = expand_tasks(args.tasks, args.nih_targets)
    for seed in args.seeds:
        seed_cache = args.repeat_cache_root / f"seed_{seed:02d}"
        if not seed_cache.exists():
            raise FileNotFoundError(f"Missing cache directory: {seed_cache}")
        seed_results: list[pd.DataFrame] = []
        seed_runtimes: list[pd.DataFrame] = []
        for task, family in tasks:
            print(f"[seed {seed:02d}] ablation task={task}", flush=True)
            results, runtimes = run_task(task, family, seed, seed_cache)
            results.insert(0, "run_seed", seed)
            runtimes.insert(0, "run_seed", seed)
            seed_results.append(results)
            seed_runtimes.append(runtimes)
        write(pd.concat(seed_results, ignore_index=True, sort=False), temp_dir / f"seed_{seed:02d}_results.csv")
        write(pd.concat(seed_runtimes, ignore_index=True, sort=False), temp_dir / f"seed_{seed:02d}_runtimes.csv")
        refresh_outputs(args.outdir, temp_dir)
        print(f"[seed {seed:02d}] wrote incremental ablation outputs", flush=True)
    for path in temp_dir.glob("*.csv"):
        path.unlink()
    temp_dir.rmdir()
    print(f"wrote ablation results to {args.outdir}", flush=True)


def run_task(task: str, family: str, split_seed: int, cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_args, data, probs, embeddings, score_name = load_task(task, family, split_seed, cache_root=cache_root)
    score_matrix = score_matrix_from_probs(probs, score_name, random_state=task_args.seed)
    base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), task_args.alpha)
    failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    inputs = image_experiment.DiscoveryInputs(
        embeddings=embeddings[data.discovery_idx],
        uncertainty=observable_uncertainty(probs)[data.discovery_idx],
        coverage_failures=failures,
        classification_errors=probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx],
        cp_scores=probs[data.discovery_idx],
    )
    selected_groups, discovery_runtimes = discover_variant_groups(task_args, data, probs, embeddings, failures, inputs)
    shared_groups = [group for groups in selected_groups.values() for group in groups]
    rows, runtimes = evaluate_variants(task, family, task_args, data, probs, embeddings, score_matrix, base_q, selected_groups, shared_groups, discovery_runtimes)
    return format_result_columns(pd.DataFrame(rows), task, family), pd.DataFrame(runtimes)


def discover_variant_groups(task_args, data, probs, embeddings, failures, inputs) -> tuple[dict[str, list], dict[str, dict[str, float]]]:
    groups_by_variant: dict[str, list] = {}
    runtimes: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        if variant.discovery_method is None:
            groups_by_variant[variant.name] = []
            runtimes[variant.name] = {"discovery_seconds": 0.0, "aggregation_seconds": 0.0}
            continue
        start = perf_counter()
        discovered = {}
        for seed in variant.discovery_seeds:
            seed_args = argparse.Namespace(**vars(task_args))
            seed_args.seed = seed
            discovered[(variant.discovery_method, seed)] = image_experiment.discover_slice_method(variant.discovery_method, seed_args, inputs)
        discovery_seconds = perf_counter() - start
        start = perf_counter()
        if variant.use_consensus:
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
        else:
            groups = [group for seed_groups in discovered.values() for group in seed_groups]
        groups_by_variant[variant.name] = groups
        runtimes[variant.name] = {"discovery_seconds": discovery_seconds, "aggregation_seconds": perf_counter() - start}
    return groups_by_variant, runtimes


def evaluate_variants(task, family, task_args, data, probs, embeddings, score_matrix, base_q, groups_by_variant, shared_groups, discovery_runtimes):
    test = data.test_idx
    known_test = data.known_group[test] if data.known_group is not None else None
    sensitive_groups = getattr(data, "sensitive_groups", None) or {}
    sensitive_test = {name: values[test] for name, values in sensitive_groups.items()}
    fixed_subgroup_name, fixed_subgroup_mask = _fixed_subgroup(task, family, sensitive_test)
    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    rows, runtimes = [], []
    for variant in VARIANTS:
        groups = groups_by_variant[variant.name]
        timing = discovery_runtimes[variant.name]
        fit_start = perf_counter()
        model = fit_variant(variant, base_q, groups, embeddings[data.mvcal_idx], mv_scores, task_args, probs[data.mvcal_idx])
        calibration_seconds = perf_counter() - fit_start
        eval_start = perf_counter()
        row = evaluate_threshold_model(
            variant.name,
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
            own_groups=groups,
            sensitive_groups=sensitive_test,
            fixed_subgroup_mask=fixed_subgroup_mask,
            fixed_subgroup_name=fixed_subgroup_name,
        )
        evaluation_seconds = perf_counter() - eval_start
        row.update({"task": task, "family": family, "num_selected_slices": len(groups)})
        rows.append(row)
        runtimes.append(
            {
                "task": task,
                "family": family,
                "method": variant.name,
                "discovery_seconds": timing["discovery_seconds"],
                "aggregation_seconds": timing["aggregation_seconds"],
                "calibration_seconds": calibration_seconds,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": timing["discovery_seconds"] + timing["aggregation_seconds"] + calibration_seconds + evaluation_seconds,
            }
        )
    return rows, runtimes


def fit_variant(variant: AblationVariant, base_q: float, groups: list, z_mv: np.ndarray, mv_scores: np.ndarray, task_args, probs_mv: np.ndarray):
    if variant.rectifier == "standard":
        return fit_standard(base_q)
    if variant.rectifier == "slice_quantile":
        return fit_slice_quantile(base_q, groups, z_mv, mv_scores, task_args.alpha, observed_scores=probs_mv, min_size=task_args.rectifier_min_size)
    if variant.rectifier == "batch_gcp":
        return fit_batch_gcp(base_q, groups, z_mv, mv_scores, task_args.alpha, observed_scores=probs_mv, min_size=task_args.rectifier_min_size)
    raise ValueError(f"Unknown rectifier: {variant.rectifier}")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in ABLATION_METRICS if column in results]
    per_seed = results.groupby(["run_seed", "family", "method"], as_index=False)[metrics].mean(numeric_only=True)
    summary = mean_std(per_seed, ["family", "method"])
    summary.insert(0, "dataset", summary["family"].map(DATASET_NAMES).fillna(summary["family"]))
    summary.insert(3, "num_seeds", int(results["run_seed"].nunique()))
    return order_summary(summary.drop(columns="family"))


def summarize_runtimes(runtimes: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in runtimes if column.endswith("_seconds")]
    per_seed = runtimes.groupby(["run_seed", "family", "method"], as_index=False)[metrics].mean(numeric_only=True)
    summary = mean_std(per_seed, ["family", "method"])
    summary.insert(0, "dataset", summary["family"].map(DATASET_NAMES).fillna(summary["family"]))
    summary.insert(3, "num_seeds", int(runtimes["run_seed"].nunique()))
    return order_summary(summary.drop(columns="family"))


def mean_std(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metric_columns = [column for column in frame.select_dtypes(include="number").columns if column != "run_seed"]
    mean = frame.groupby(groups, as_index=False)[metric_columns].mean(numeric_only=True)
    std = frame.groupby(groups, as_index=False)[metric_columns].std(numeric_only=True).fillna(0.0)
    mean = mean.rename(columns={column: f"{column}_mean" for column in metric_columns})
    std = std.rename(columns={column: f"{column}_std" for column in metric_columns})
    return mean.merge(std, on=groups, how="left")


def order_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_dataset_order"] = ordered["dataset"].map({name: index for index, name in enumerate(DATASET_NAMES.values())}).fillna(99)
    ordered["_method_order"] = ordered["method"].map(DISPLAY_ORDER)
    return ordered.sort_values(["_dataset_order", "_method_order"]).drop(columns=["_dataset_order", "_method_order"]).reset_index(drop=True)


def write_analysis(summary: pd.DataFrame, runtimes: pd.DataFrame, path: Path) -> None:
    lines = ["# Consensus-CPScore Ablation", "", "All results are mean ± std over full-system seeds. NIH and MIMIC-CXR are averaged over target tasks within each seed.", ""]
    for dataset, part in summary.groupby("dataset", sort=False):
        display = part[["method", "marginal_coverage_mean", "avg_set_size_mean", "worst_class_coverage_mean", "worst_known_group_coverage_mean", "known_group_mean_undercoverage_gap_mean", "max_undercoverage_gap_mean", "num_selected_slices_mean"]].copy()
        lines.extend([f"## {dataset}", "", display.to_markdown(index=False, floatfmt=".4f"), ""])
    lines.extend(["## Runtime", "", runtimes[["dataset", "method", "total_seconds_mean", "total_seconds_std"]].to_markdown(index=False, floatfmt=".4f"), ""])
    lines.extend(write_takeaways(summary))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_takeaways(summary: pd.DataFrame) -> list[str]:
    lines = ["## Takeaways", ""]
    for dataset, part in summary.groupby("dataset", sort=False):
        full = row_for(part, "Full Consensus-CPScore")
        standard = row_for(part, "No local repair")
        single = row_for(part, "Single-seed CPScore")
        kmeans = row_for(part, "Consensus K-means")
        if full is None or standard is None:
            continue
        coverage_gain = full["worst_known_group_coverage_mean"] - standard["worst_known_group_coverage_mean"] if pd.notna(full.get("worst_known_group_coverage_mean")) and pd.notna(standard.get("worst_known_group_coverage_mean")) else np.nan
        size_cost = full["avg_set_size_mean"] - standard["avg_set_size_mean"]
        sentence = f"- {dataset}: full method changes average size by {size_cost:.4f}"
        if not np.isnan(coverage_gain):
            sentence += f" and worst known-group coverage by {coverage_gain:+.4f} versus no local repair"
        if single is not None and pd.notna(full.get("worst_known_group_coverage_mean")) and pd.notna(single.get("worst_known_group_coverage_mean")):
            sentence += f"; consensus adds {full['worst_known_group_coverage_mean'] - single['worst_known_group_coverage_mean']:+.4f} over single-seed CPScore"
        if kmeans is not None and pd.notna(full.get("worst_known_group_coverage_mean")) and pd.notna(kmeans.get("worst_known_group_coverage_mean")):
            sentence += f"; CP-score discovery adds {full['worst_known_group_coverage_mean'] - kmeans['worst_known_group_coverage_mean']:+.4f} over K-means consensus"
        lines.append(sentence + ".")
    return lines


def row_for(frame: pd.DataFrame, method: str) -> pd.Series | None:
    rows = frame[frame["method"].eq(method)]
    if rows.empty:
        return None
    return rows.iloc[0]



def refresh_outputs(outdir: Path, temp_dir: Path) -> None:
    result_files = sorted(temp_dir.glob("seed_*_results.csv"))
    runtime_files = sorted(temp_dir.glob("seed_*_runtimes.csv"))
    if not result_files or not runtime_files:
        return
    results = pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True, sort=False)
    runtimes = pd.concat([pd.read_csv(path) for path in runtime_files], ignore_index=True, sort=False)
    write(results, outdir / "all_results_by_seed.csv")
    write(runtimes, outdir / "all_runtimes_by_seed.csv")
    summary = summarize(results)
    runtime_summary = summarize_runtimes(runtimes)
    write(summary, outdir / "summary_mean_std.csv")
    write(runtime_summary, outdir / "runtime_mean_std.csv")
    write_analysis(summary, runtime_summary, outdir / "analysis.md")

def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numbers(frame).to_csv(path, index=False, float_format="%.5f")


if __name__ == "__main__":
    main()

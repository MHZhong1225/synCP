from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.conformal import (
    classwise_quantiles,
    conformal_quantile,
    covered,
    groupwise_quantiles,
    observable_uncertainty,
    score_matrix_from_probs,
    set_sizes,
    thresholds_from_groups,
    true_label_scores,
)
from core.metrics import evaluate_groupwise_model, evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_pool_mvp, fit_slice_quantile, fit_standard
from experiments import images as image_experiment
from reports.output import print_results, write_results
from slices.discovery import GroupFunction, select_group_pool


METHODS = ("random_halfspace", "kmeans", "gmm_unaware", "error_domino", "coverage_domino")
POOL_METHODS = ("gmm_unaware", "kmeans", "coverage_domino", "random_halfspace")
ALPHAS = (0.20, 0.15, 0.10, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation experiments for hidden undercoverage and DOCP repair.")
    parser.add_argument("--preset", default="ham10000", choices=sorted(image_experiment.DATASET_PRESETS))
    parser.add_argument("--outdir", type=Path, default=Path("results/validation/ham10000"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--known-group", default=None)
    parser.add_argument("--hidden-source", choices=["known_group", "class"], default="known_group")
    parser.add_argument("--hidden-value", default="auto")
    parser.add_argument("--shift-delta", type=float, default=0.15)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--pool-groups", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    image_args = _image_args(args, alpha=0.10)
    data = image_experiment.load_experiment_data(image_args)
    probs, embeddings = image_experiment.load_or_create_features(image_args, data, args.device)

    hidden = _choose_hidden_group(args, data, embeddings, probs, alpha=0.10)
    print(
        "validation: "
        f"dataset={args.preset}, hidden_source={hidden.source}, hidden_value={hidden.value}, "
        f"hidden_test_n={int(hidden.test_mask.sum())}, shift_delta={args.shift_delta}",
        file=sys.stderr,
        flush=True,
    )

    discovery_frame, ablation_frame = _hidden_recovery_and_repair(args, image_args, data, probs, embeddings, hidden)
    tradeoff_frame, setsize_frame = _efficiency_tradeoff(args, image_args, data, probs, embeddings)

    discovery_frame = _write(discovery_frame, args.outdir / "hidden_recovery.csv")
    ablation_frame = _write(ablation_frame, args.outdir / "rectifier_ablation.csv")
    tradeoff_frame = _write(tradeoff_frame, args.outdir / "alpha_tradeoff.csv")
    setsize_frame = _write(setsize_frame, args.outdir / "set_size_distribution.csv")
    _write_figures(tradeoff_frame, setsize_frame, args.outdir)

    print("\nhidden recovery")
    print_results(discovery_frame)
    print("\nrectifier ablation")
    print_results(ablation_frame)
    print("\nwrote:")
    for name in ["hidden_recovery.csv", "rectifier_ablation.csv", "alpha_tradeoff.csv", "set_size_distribution.csv"]:
        print(f"  {args.outdir / name}")


class HiddenGroup:
    def __init__(self, source: str, value: int, discovery_mask: np.ndarray, mvcal_mask: np.ndarray, test_mask: np.ndarray) -> None:
        self.source = source
        self.value = value
        self.discovery_mask = discovery_mask
        self.mvcal_mask = mvcal_mask
        self.test_mask = test_mask


def _image_args(args: argparse.Namespace, alpha: float) -> argparse.Namespace:
    image_args = argparse.Namespace(
        preset=args.preset,
        hyperparams=None,
        hyperparams_key=None,
        dataset_root=Path("../dataset/bracs"),
        dataset_type="imagefolder",
        split_mode="official_test",
        nih_target="Effusion",
        known_group="gender",
        max_samples=None,
        outdir=Path("results") / args.preset,
        device=args.device,
        cuda=None,
        seed=args.seed,
        seeds=None,
        alpha=alpha,
        epochs=8,
        lr=1e-4,
        batch_size=64,
        predict_batch_size=128,
        backbone="resnet18",
        backbone_checkpoint=None,
        cache_file=None,
        cache=True,
        components=20,
        min_size=20,
        rectifier_min_size=10,
        pool_groups=8,
        epsilon=image_experiment.COVERAGE_EPSILON,
        enrichment=image_experiment.COVERAGE_ENRICHMENT,
        pool_methods=list(POOL_METHODS),
        diagnostics=False,
    )
    image_args = image_experiment._resolve_experiment_args(image_args)
    if args.known_group is not None:
        image_args.known_group = args.known_group
    if args.components is not None:
        image_args.components = args.components
    if args.min_size is not None:
        image_args.min_size = args.min_size
        image_args.rectifier_min_size = max(5, args.min_size // 2)
    if args.pool_groups is not None:
        image_args.pool_groups = args.pool_groups
    image_args.alpha = alpha
    image_args.device = args.device
    image_args.outdir = args.outdir
    return image_args


def _choose_hidden_group(args: argparse.Namespace, data, embeddings: np.ndarray, probs: np.ndarray, alpha: float) -> HiddenGroup:
    labels = data.labels
    values = data.known_group if args.hidden_source == "known_group" else labels
    if values is None:
        raise ValueError(f"{args.preset} has no known_group; use --hidden-source class.")
    if args.hidden_value != "auto":
        value = int(args.hidden_value)
    else:
        value = _auto_hidden_value(values, data.discovery_idx, data.test_idx, min_size=20)
    return HiddenGroup(
        source=args.hidden_source,
        value=value,
        discovery_mask=values[data.discovery_idx] == value,
        mvcal_mask=values[data.mvcal_idx] == value,
        test_mask=values[data.test_idx] == value,
    )


def _auto_hidden_value(values: np.ndarray, discovery_idx: np.ndarray, test_idx: np.ndarray, min_size: int) -> int:
    candidates = []
    for value in np.unique(values):
        discovery_n = int((values[discovery_idx] == value).sum())
        test_n = int((values[test_idx] == value).sum())
        if discovery_n >= min_size and test_n >= min_size:
            candidates.append((test_n, int(value)))
    if not candidates:
        raise ValueError("No hidden group has enough discovery and test samples.")
    return min(candidates)[1]


def _hidden_recovery_and_repair(args, image_args, data, probs, embeddings, hidden: HiddenGroup) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_matrix = _shifted_score_matrix(probs, data, hidden, args.shift_delta)
    observed = observable_uncertainty(probs)
    base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), image_args.alpha)
    discovery_failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    classification_errors = probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx]
    discovery_outputs = _discover_all(image_args, embeddings[data.discovery_idx], observed[data.discovery_idx], discovery_failures, classification_errors)
    pool = select_group_pool(
        _pool_source_outputs(discovery_outputs, len(data.class_names)),
        embeddings[data.discovery_idx],
        probs[data.discovery_idx],
        discovery_failures,
        image_args.alpha,
        methods=tuple(image_args.pool_methods),
        min_size=image_args.min_size,
        epsilon=image_args.epsilon,
        enrichment=image_args.enrichment,
        max_groups=image_args.pool_groups,
    )
    discovery_outputs["docp_pool"] = pool
    eval_groups = _collect_groups(discovery_outputs)
    recovery = _discovery_recovery_table(discovery_outputs, embeddings, probs, score_matrix, data, hidden, discovery_failures, image_args.alpha, image_args.rectifier_min_size)
    ablation = _rectifier_ablation_table(image_args, data, embeddings, probs, score_matrix, observed, base_q, pool, eval_groups, hidden)
    return recovery, ablation


def _efficiency_tradeoff(args, base_image_args, data, probs, embeddings) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    setsize_rows = []
    for alpha in ALPHAS:
        image_args = copy.copy(base_image_args)
        image_args.alpha = alpha
        score_matrix = score_matrix_from_probs(probs, _score_name(len(data.class_names)))
        observed = observable_uncertainty(probs)
        base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), alpha)
        discovery_failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
        classification_errors = probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx]
        discovery_outputs = _discover_all(image_args, embeddings[data.discovery_idx], observed[data.discovery_idx], discovery_failures, classification_errors)
        pool = select_group_pool(
            _pool_source_outputs(discovery_outputs, len(data.class_names)),
            embeddings[data.discovery_idx],
            probs[data.discovery_idx],
            discovery_failures,
            alpha,
            methods=tuple(image_args.pool_methods),
            min_size=image_args.min_size,
            epsilon=image_args.epsilon,
            enrichment=image_args.enrichment,
            max_groups=image_args.pool_groups,
        )
        discovery_outputs["docp_pool"] = pool
        eval_groups = _collect_groups(discovery_outputs)
        rows.extend(_tradeoff_rows(image_args, data, embeddings, probs, score_matrix, observed, base_q, discovery_outputs, eval_groups))
        if alpha == 0.10:
            setsize_rows.extend(_setsize_rows(image_args, data, embeddings, probs, score_matrix, observed, base_q, pool))
    return pd.DataFrame(rows), pd.DataFrame(setsize_rows)


def _shifted_score_matrix(probs: np.ndarray, data, hidden: HiddenGroup, delta: float) -> np.ndarray:
    score_matrix = score_matrix_from_probs(probs, _score_name(len(data.class_names)))
    shifted = score_matrix.copy()
    for indices, mask in [(data.discovery_idx, hidden.discovery_mask), (data.mvcal_idx, hidden.mvcal_mask), (data.test_idx, hidden.test_mask)]:
        shifted[indices[mask]] = np.clip(shifted[indices[mask]] + delta, 0.0, 1.0)
    return shifted


def _discover_all(image_args, z, uncertainty, failures, classification_errors) -> dict[str, list[GroupFunction]]:
    inputs = image_experiment.DiscoveryInputs(z, uncertainty, failures, classification_errors)
    return {method: image_experiment.discover_slice_method(method, image_args, inputs) for method in METHODS}


def _discovery_recovery_table(
    outputs: dict[str, list[GroupFunction]],
    embeddings: np.ndarray,
    probs: np.ndarray,
    score_matrix: np.ndarray,
    data,
    hidden: HiddenGroup,
    discovery_failures: np.ndarray,
    alpha: float,
    min_size: int,
) -> pd.DataFrame:
    rows = []
    test_covered = covered(score_matrix[data.test_idx], data.labels[data.test_idx], conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), alpha))
    for method, groups in outputs.items():
        scores = _membership_scores(groups, embeddings[data.discovery_idx], probs[data.discovery_idx])
        union = scores > 0
        rows.append(
            {
                "method": _discovery_name(method),
                "num_slices": len(groups),
                "precision_at_50": _precision_at_k(scores, hidden.discovery_mask, 50),
                "recall": _safe_ratio(np.logical_and(union, hidden.discovery_mask).sum(), hidden.discovery_mask.sum()),
                "jaccard": _jaccard(union, hidden.discovery_mask),
                "membership_auroc": _safe_auroc(hidden.discovery_mask, scores),
                "discovery_slice_miscoverage": _mean_slice_miscoverage(groups, embeddings[data.discovery_idx], probs[data.discovery_idx], discovery_failures, min_size),
                "confirmed_slice_rate": _confirmed_slice_rate(groups, embeddings[data.test_idx], probs[data.test_idx], test_covered, alpha, min_size),
                "avg_undercoverage_gap": _avg_test_gap(groups, embeddings[data.test_idx], probs[data.test_idx], test_covered, alpha, min_size),
            }
        )
    return pd.DataFrame(rows)


def _rectifier_ablation_table(image_args, data, embeddings, probs, score_matrix, observed, base_q, pool, eval_groups, hidden: HiddenGroup) -> pd.DataFrame:
    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    hidden_test = hidden.test_mask[:, None]
    rows = []
    standard = fit_standard(base_q)
    rows.append(_eval_model("No rectification", standard, data, embeddings, probs, score_matrix, eval_groups, image_args.alpha, hidden_test))
    slice_q = fit_slice_quantile(base_q, pool, embeddings[data.mvcal_idx], mv_scores, image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size)
    rows.append(_eval_model("Slice-specific quantile", slice_q, data, embeddings, probs, score_matrix, eval_groups, image_args.alpha, hidden_test, own_groups=pool))
    group_max = fit_batch_gcp(base_q, pool, embeddings[data.mvcal_idx], mv_scores, image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size)
    rows.append(_eval_model("Group-max quantile", group_max, data, embeddings, probs, score_matrix, eval_groups, image_args.alpha, hidden_test, own_groups=pool))
    pool_mvp = fit_pool_mvp(base_q, pool, embeddings[data.mvcal_idx], mv_scores, image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size, max_active_groups=image_args.pool_groups)
    rows.append(_eval_model("Pool-MVP", pool_mvp, data, embeddings, probs, score_matrix, eval_groups, image_args.alpha, hidden_test, own_groups=pool))
    if data.known_group is not None:
        known_q = groupwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], data.known_group[data.calib0_idx], image_args.alpha)
        rows.append(
            evaluate_groupwise_model(
                "Known-group rectification",
                known_q,
                embeddings[data.test_idx],
                score_matrix[data.test_idx],
                data.labels[data.test_idx],
                data.known_group[data.test_idx],
                hidden_test,
                data.known_group[data.test_idx],
                eval_groups,
                image_args.alpha,
                observed_scores=probs[data.test_idx],
                eval_observed_scores=probs[data.test_idx],
            )
        )
    frame = pd.DataFrame(rows)
    keep = ["method", "marginal_coverage", "avg_set_size", "max_undercoverage_gap", "worst_hidden_group_coverage", "worst_discovered_slice_coverage", "worst_known_group_coverage", "worst_class_coverage"]
    return frame[[col for col in keep if col in frame.columns]]


def _tradeoff_rows(image_args, data, embeddings, probs, score_matrix, observed, base_q, discovery_outputs, eval_groups) -> list[dict[str, float | str]]:
    rows = []
    methods = {
        "Standard CP": fit_standard(base_q),
        "K-means": fit_slice_quantile(base_q, discovery_outputs["kmeans"], embeddings[data.mvcal_idx], true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx]), image_args.alpha, observed_scores=observed[data.mvcal_idx], min_size=image_args.rectifier_min_size),
        "Coverage-Domino": fit_slice_quantile(base_q, discovery_outputs["coverage_domino"], embeddings[data.mvcal_idx], true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx]), image_args.alpha, observed_scores=observed[data.mvcal_idx], min_size=image_args.rectifier_min_size),
        "DOCP": fit_batch_gcp(base_q, discovery_outputs["docp_pool"], embeddings[data.mvcal_idx], true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx]), image_args.alpha, observed_scores=probs[data.mvcal_idx], min_size=image_args.rectifier_min_size),
    }
    for name, model in methods.items():
        row = evaluate_threshold_model(
            name,
            model,
            embeddings[data.test_idx],
            score_matrix[data.test_idx],
            data.labels[data.test_idx],
            None,
            data.known_group[data.test_idx] if data.known_group is not None else None,
            eval_groups,
            image_args.alpha,
            observed_scores=probs[data.test_idx] if name == "DOCP" else observed[data.test_idx],
            eval_observed_scores=probs[data.test_idx],
            own_groups=discovery_outputs["docp_pool"] if name == "DOCP" else None,
        )
        row["alpha"] = image_args.alpha
        row["target_coverage"] = 1.0 - image_args.alpha
        rows.append(row)
    return rows


def _setsize_rows(image_args, data, embeddings, probs, score_matrix, observed, base_q, pool) -> list[dict[str, float | str | int]]:
    standard_q = np.full(len(data.test_idx), base_q, dtype=np.float64)
    docp = fit_batch_gcp(
        base_q,
        pool,
        embeddings[data.mvcal_idx],
        true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx]),
        image_args.alpha,
        observed_scores=probs[data.mvcal_idx],
        min_size=image_args.rectifier_min_size,
    )
    docp_q = docp.thresholds(embeddings[data.test_idx], observed_scores=probs[data.test_idx], score_matrix=score_matrix[data.test_idx])
    standard_sizes = set_sizes(score_matrix[data.test_idx], standard_q)
    docp_sizes = set_sizes(score_matrix[data.test_idx], docp_q)
    slice_union = _membership_scores(pool, embeddings[data.test_idx], probs[data.test_idx]) > 0
    rows = [
        _size_summary("all", "all", standard_sizes, docp_sizes, np.ones(len(data.test_idx), dtype=bool)),
        _size_summary("discovered_slice", "union", standard_sizes, docp_sizes, slice_union),
        _size_summary("non_slice", "complement", standard_sizes, docp_sizes, ~slice_union),
    ]
    if data.known_group is not None:
        for group in np.unique(data.known_group[data.test_idx]):
            rows.append(_size_summary("known_group", str(group), standard_sizes, docp_sizes, data.known_group[data.test_idx] == group))
    return [row for row in rows if row["n"] > 0]


def _eval_model(name, model, data, embeddings, probs, score_matrix, eval_groups, alpha, hidden_test, own_groups=None):
    return evaluate_threshold_model(
        name,
        model,
        embeddings[data.test_idx],
        score_matrix[data.test_idx],
        data.labels[data.test_idx],
        hidden_test,
        data.known_group[data.test_idx] if data.known_group is not None else None,
        eval_groups,
        alpha,
        observed_scores=probs[data.test_idx],
        eval_observed_scores=probs[data.test_idx],
        own_groups=own_groups,
    )


def _membership_scores(groups: list[GroupFunction], z: np.ndarray, observed_scores: np.ndarray) -> np.ndarray:
    if not groups:
        return np.zeros(len(z), dtype=np.float64)
    scores = np.zeros(len(z), dtype=np.float64)
    for group in groups:
        scores += group.membership(z, observed_scores).astype(np.float64)
    return scores


def _precision_at_k(scores: np.ndarray, target: np.ndarray, k: int) -> float:
    if len(scores) == 0 or target.sum() == 0:
        return float("nan")
    k = min(k, len(scores))
    order = np.lexsort((np.arange(len(scores)), -scores))
    top = order[:k]
    return float(target[top].mean())


def _safe_auroc(target: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(target)) < 2 or np.all(scores == scores[0]):
        return float("nan")
    return float(roc_auc_score(target, scores))


def _mean_slice_miscoverage(groups: list[GroupFunction], z: np.ndarray, observed_scores: np.ndarray, failures: np.ndarray, min_size: int) -> float:
    values = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        if mask.sum() >= min_size:
            values.append(float(failures[mask].mean()))
    return float(np.mean(values)) if values else float("nan")


def _confirmed_slice_rate(groups: list[GroupFunction], z: np.ndarray, observed_scores: np.ndarray, covered_mask: np.ndarray, alpha: float, min_size: int) -> float:
    confirmed = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        if mask.sum() >= min_size:
            confirmed.append(float((~covered_mask[mask]).mean()) > alpha)
    return float(np.mean(confirmed)) if confirmed else float("nan")


def _avg_test_gap(groups: list[GroupFunction], z: np.ndarray, observed_scores: np.ndarray, covered_mask: np.ndarray, alpha: float, min_size: int) -> float:
    gaps = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        if mask.sum() >= min_size:
            gaps.append(max(0.0, (1.0 - alpha) - float(covered_mask[mask].mean())))
    return float(np.mean(gaps)) if gaps else float("nan")


def _pool_source_outputs(outputs: dict[str, list[GroupFunction]], num_classes: int) -> dict[str, list[GroupFunction]]:
    return {method: image_experiment.class_conditional_groups(groups, num_classes) for method, groups in outputs.items()}


def _collect_groups(outputs: dict[str, list[GroupFunction]]) -> list[GroupFunction]:
    groups: list[GroupFunction] = []
    for method_groups in outputs.values():
        groups.extend(method_groups)
    return groups


def _score_name(num_classes: int) -> str:
    return "inverse_probability" if num_classes == 2 else "aps"


def _discovery_name(method: str) -> str:
    return "DOCP pool" if method == "docp_pool" else method.replace("_", " ").title()


def _safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom else float("nan")


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def _size_summary(subset: str, group: str, standard: np.ndarray, docp: np.ndarray, mask: np.ndarray) -> dict[str, float | str | int]:
    return {
        "subset": subset,
        "group": group,
        "n": int(mask.sum()),
        "standard_avg_set_size": float(standard[mask].mean()) if mask.any() else float("nan"),
        "docp_avg_set_size": float(docp[mask].mean()) if mask.any() else float("nan"),
        "delta_avg_set_size": float((docp[mask] - standard[mask]).mean()) if mask.any() else float("nan"),
        "standard_p90_set_size": float(np.quantile(standard[mask], 0.90)) if mask.any() else float("nan"),
        "docp_p90_set_size": float(np.quantile(docp[mask], 0.90)) if mask.any() else float("nan"),
    }


def _write(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    return write_results(frame, path)


def _write_figures(tradeoff: pd.DataFrame, setsize: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method, part in tradeoff.groupby("method", sort=False):
        ax.plot(part["avg_set_size"], part["worst_discovered_slice_coverage"], marker="o", label=method)
    ax.set_xlabel("Average set size")
    ax.set_ylabel("Worst discovered-slice coverage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "coverage_set_size_curve.png", dpi=200)
    plt.close(fig)

    compact = setsize[setsize["subset"].isin(["all", "discovered_slice", "non_slice"])].copy()
    x = np.arange(len(compact))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - 0.18, compact["standard_avg_set_size"], width=0.36, label="Standard CP")
    ax.bar(x + 0.18, compact["docp_avg_set_size"], width=0.36, label="DOCP")
    ax.set_xticks(x)
    ax.set_xticklabels(compact["subset"], rotation=15, ha="right")
    ax.set_ylabel("Average set size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "set_size_distribution.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

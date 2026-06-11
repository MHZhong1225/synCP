from __future__ import annotations

from dataclasses import asdict
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

from core.conformal import covered, covered_classwise, set_sizes, set_sizes_classwise, thresholds_from_groups, true_label_scores
from core.rectification import ThresholdModel
from slices.discovery import GroupFunction, SliceDiagnostics


def evaluate_prediction_sets(
    name: str,
    prediction_sets: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    hidden_groups: np.ndarray | None,
    known_group: np.ndarray | None,
    discovered_groups: list[GroupFunction],
    alpha: float,
    observed_scores: np.ndarray,
    own_groups: list[GroupFunction] | None = None,
    sensitive_groups: dict[str, np.ndarray] | None = None,
    fixed_subgroup_mask: np.ndarray | None = None,
    fixed_subgroup_name: str = "",
) -> dict[str, float | str | int]:
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    if prediction_sets.ndim != 2 or prediction_sets.shape[0] != len(y):
        raise ValueError("prediction_sets must have shape (n_examples, n_classes)")
    is_covered = prediction_sets[np.arange(len(y)), y]
    sizes = prediction_sets.sum(axis=1)
    row: dict[str, float | str | int] = {
        "method": name,
        "marginal_coverage": float(is_covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "max_undercoverage_gap": max(0.0, (1.0 - alpha) - float(is_covered.mean())),
    }
    _add_target_metrics(row, alpha)
    _add_group_coverage_metrics(row, discovered_groups, own_groups, z, observed_scores, is_covered)
    _add_fixed_subgroup_metrics(row, fixed_subgroup_mask, fixed_subgroup_name, is_covered, sizes, alpha)
    row["worst_hidden_group_coverage"] = _worst_binary_matrix_coverage(hidden_groups, is_covered)
    if not sensitive_groups:
        _add_known_group_metrics(row, known_group, is_covered, alpha)
    _add_sensitive_group_metrics(row, sensitive_groups, is_covered, alpha)
    row["worst_class_coverage"] = _worst_label_coverage(y, is_covered)
    finite = [
        float(row[key])
        for key in [
            "worst_shared_discovered_slice_coverage",
            "worst_hidden_group_coverage",
            "worst_known_group_coverage",
            "worst_class_coverage",
        ]
        if not np.isnan(float(row[key]))
    ]
    if finite:
        row["max_undercoverage_gap"] = max(0.0, (1.0 - alpha) - min(finite))
    return row


def evaluate_threshold_model(
    name: str,
    model: ThresholdModel,
    z: np.ndarray,
    score_matrix: np.ndarray,
    y: np.ndarray,
    hidden_groups: np.ndarray | None,
    known_group: np.ndarray | None,
    discovered_groups: list[GroupFunction],
    alpha: float,
    observed_scores: np.ndarray | None = None,
    eval_observed_scores: np.ndarray | None = None,
    own_groups: list[GroupFunction] | None = None,
    sensitive_groups: dict[str, np.ndarray] | None = None,
    fixed_subgroup_mask: np.ndarray | None = None,
    fixed_subgroup_name: str = "",
) -> dict[str, float | str | int]:
    true_scores = true_label_scores(score_matrix, y)
    if observed_scores is None:
        observed_scores = score_matrix.min(axis=1)
    thresholds = model.thresholds(z, observed_scores=observed_scores, score_matrix=score_matrix)
    is_covered = covered(score_matrix, y, thresholds)
    sizes = set_sizes(score_matrix, thresholds)
    row: dict[str, float | str | int] = {
        "method": name,
        "marginal_coverage": float(is_covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "max_undercoverage_gap": max(0.0, (1.0 - alpha) - float(is_covered.mean())),
    }
    _add_target_metrics(row, alpha)
    eval_scores = eval_observed_scores if eval_observed_scores is not None else observed_scores
    _add_group_coverage_metrics(row, discovered_groups, own_groups, z, eval_scores, is_covered)
    _add_fixed_subgroup_metrics(row, fixed_subgroup_mask, fixed_subgroup_name, is_covered, sizes, alpha)
    row["worst_hidden_group_coverage"] = _worst_binary_matrix_coverage(hidden_groups, is_covered)
    if not sensitive_groups:
        _add_known_group_metrics(row, known_group, is_covered, alpha)
    _add_sensitive_group_metrics(row, sensitive_groups, is_covered, alpha)
    row["worst_class_coverage"] = _worst_label_coverage(y, is_covered)
    worst_values = [
        row["worst_shared_discovered_slice_coverage"],
        row["worst_hidden_group_coverage"],
        row["worst_known_group_coverage"],
        row["worst_class_coverage"],
    ]
    finite = [v for v in worst_values if not np.isnan(float(v))]
    if finite:
        row["max_undercoverage_gap"] = max(0.0, (1.0 - alpha) - min(float(v) for v in finite))
    return row



def _add_fixed_subgroup_metrics(
    row: dict[str, float | str | int],
    mask: np.ndarray | None,
    name: str,
    covered_mask: np.ndarray,
    sizes: np.ndarray,
    alpha: float,
) -> None:
    row["fixed_subgroup"] = name
    if mask is None:
        row["fixed_subgroup_size"] = float("nan")
        row["fixed_subgroup_coverage"] = float("nan")
        row["fixed_subgroup_avg_set_size"] = float("nan")
        row["fixed_subgroup_undercoverage_gap"] = float("nan")
        return
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != covered_mask.shape[0]:
        raise ValueError("fixed_subgroup_mask must match the evaluation set length")
    count = int(mask.sum())
    row["fixed_subgroup_size"] = count
    if count == 0:
        row["fixed_subgroup_coverage"] = float("nan")
        row["fixed_subgroup_avg_set_size"] = float("nan")
        row["fixed_subgroup_undercoverage_gap"] = float("nan")
        return
    coverage = float(covered_mask[mask].mean())
    row["fixed_subgroup_coverage"] = coverage
    row["fixed_subgroup_avg_set_size"] = float(sizes[mask].mean())
    row["fixed_subgroup_undercoverage_gap"] = max(0.0, (1.0 - alpha) - coverage)

def diagnostics_to_frame(diags: list[SliceDiagnostics]) -> pd.DataFrame:
    if not diags:
        return pd.DataFrame(columns=["method", "name", "size", "miscoverage", "enrichment"])
    return pd.DataFrame([asdict(d) for d in diags])


def evaluate_classifier(
    probs: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    known_group: np.ndarray | None = None,
) -> pd.DataFrame:
    predicted = probs.argmax(axis=1)
    rows = [_classification_summary_row("overall", "all", probs, y, predicted)]
    rows.extend(_class_rows(y, predicted, class_names, probs.shape[1]))
    if known_group is not None:
        rows.extend(_group_rows(probs, y, predicted, known_group))
    return pd.DataFrame(rows)


def _classification_summary_row(
    subset: str,
    group: str,
    probs: np.ndarray,
    y: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    return {
        "subset": subset,
        "group": group,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y, predicted)),
        "macro_f1": _safe_metric(lambda: f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": _safe_metric(lambda: f1_score(y, predicted, average="weighted", zero_division=0)),
        "auroc": _safe_metric(lambda: _roc_auc(y, probs)),
        "auprc": _safe_metric(lambda: _average_precision(y, probs)),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
    }


def _class_rows(
    y: np.ndarray,
    predicted: np.ndarray,
    class_names: list[str] | None,
    num_classes: int,
) -> list[dict[str, float | int | str]]:
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=labels, zero_division=0)
    names = class_names or [str(label) for label in labels]
    return [
        {
            "subset": "class",
            "group": str(names[label]),
            "n": int(support[label]),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
        }
        for label in labels
    ]


def _group_rows(
    probs: np.ndarray,
    y: np.ndarray,
    predicted: np.ndarray,
    known_group: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows = []
    for group in np.unique(known_group):
        mask = known_group == group
        if not mask.any():
            continue
        rows.append(_classification_summary_row("known_group", str(group), probs[mask], y[mask], predicted[mask]))
    return rows


def _roc_auc(y: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if probs.shape[1] == 2:
        return float(roc_auc_score(y, probs[:, 1]))
    return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))


def _average_precision(y: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if probs.shape[1] == 2:
        return float(average_precision_score(y, probs[:, 1]))
    y_binary = label_binarize(y, classes=np.arange(probs.shape[1]))
    return float(average_precision_score(y_binary, probs, average="macro"))


def _safe_metric(metric) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(metric())
    except ValueError:
        return float("nan")


def evaluate_classwise_model(
    name: str,
    class_thresholds: np.ndarray,
    z: np.ndarray,
    score_matrix: np.ndarray,
    y: np.ndarray,
    hidden_groups: np.ndarray | None,
    known_group: np.ndarray | None,
    discovered_groups: list[GroupFunction],
    alpha: float,
    observed_scores: np.ndarray | None = None,
    eval_observed_scores: np.ndarray | None = None,
    own_groups: list[GroupFunction] | None = None,
    sensitive_groups: dict[str, np.ndarray] | None = None,
    fixed_subgroup_mask: np.ndarray | None = None,
    fixed_subgroup_name: str = "",
) -> dict[str, float | str | int]:
    if observed_scores is None:
        observed_scores = score_matrix.min(axis=1)
    is_covered = covered_classwise(score_matrix, y, class_thresholds)
    sizes = set_sizes_classwise(score_matrix, class_thresholds)
    row: dict[str, float | str | int] = {
        "method": name,
        "marginal_coverage": float(is_covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "max_undercoverage_gap": max(0.0, (1.0 - alpha) - float(is_covered.mean())),
    }
    _add_target_metrics(row, alpha)
    eval_scores = eval_observed_scores if eval_observed_scores is not None else observed_scores
    _add_group_coverage_metrics(row, discovered_groups, own_groups, z, eval_scores, is_covered)
    _add_fixed_subgroup_metrics(row, fixed_subgroup_mask, fixed_subgroup_name, is_covered, sizes, alpha)
    row["worst_hidden_group_coverage"] = _worst_binary_matrix_coverage(hidden_groups, is_covered)
    if not sensitive_groups:
        _add_known_group_metrics(row, known_group, is_covered, alpha)
    _add_sensitive_group_metrics(row, sensitive_groups, is_covered, alpha)
    row["worst_class_coverage"] = _worst_label_coverage(y, is_covered)
    finite = [
        float(row[key])
        for key in ["worst_shared_discovered_slice_coverage", "worst_hidden_group_coverage", "worst_known_group_coverage", "worst_class_coverage"]
        if not np.isnan(float(row[key]))
    ]
    if finite:
        row["max_undercoverage_gap"] = max(0.0, (1.0 - alpha) - min(finite))
    return row


def evaluate_groupwise_model(
    name: str,
    group_thresholds: dict[int, float],
    z: np.ndarray,
    score_matrix: np.ndarray,
    y: np.ndarray,
    eval_groups: np.ndarray,
    hidden_groups: np.ndarray | None,
    known_group: np.ndarray | None,
    discovered_groups: list[GroupFunction],
    alpha: float,
    observed_scores: np.ndarray | None = None,
    eval_observed_scores: np.ndarray | None = None,
    own_groups: list[GroupFunction] | None = None,
    sensitive_groups: dict[str, np.ndarray] | None = None,
    fixed_subgroup_mask: np.ndarray | None = None,
    fixed_subgroup_name: str = "",
) -> dict[str, float | str | int]:
    if observed_scores is None:
        observed_scores = score_matrix.min(axis=1)
    thresholds = thresholds_from_groups(eval_groups, group_thresholds)
    is_covered = covered(score_matrix, y, thresholds)
    sizes = set_sizes(score_matrix, thresholds)
    row: dict[str, float | str | int] = {
        "method": name,
        "marginal_coverage": float(is_covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "max_undercoverage_gap": max(0.0, (1.0 - alpha) - float(is_covered.mean())),
    }
    _add_target_metrics(row, alpha)
    eval_scores = eval_observed_scores if eval_observed_scores is not None else observed_scores
    _add_group_coverage_metrics(row, discovered_groups, own_groups, z, eval_scores, is_covered)
    _add_fixed_subgroup_metrics(row, fixed_subgroup_mask, fixed_subgroup_name, is_covered, sizes, alpha)
    row["worst_hidden_group_coverage"] = _worst_binary_matrix_coverage(hidden_groups, is_covered)
    if not sensitive_groups:
        _add_known_group_metrics(row, known_group, is_covered, alpha)
    _add_sensitive_group_metrics(row, sensitive_groups, is_covered, alpha)
    row["worst_class_coverage"] = _worst_label_coverage(y, is_covered)
    finite = [
        float(row[key])
        for key in ["worst_shared_discovered_slice_coverage", "worst_hidden_group_coverage", "worst_known_group_coverage", "worst_class_coverage"]
        if not np.isnan(float(row[key]))
    ]
    if finite:
        row["max_undercoverage_gap"] = max(0.0, (1.0 - alpha) - min(finite))
    return row


def discovered_slice_test_quality(
    method: str,
    groups: list[GroupFunction],
    z: np.ndarray,
    observed_scores: np.ndarray,
    covered_mask: np.ndarray,
    alpha: float,
    min_size: int = 20,
) -> pd.DataFrame:
    rows = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        size = int(mask.sum())
        if size < min_size:
            continue
        cov = float(covered_mask[mask].mean())
        failures = int((~covered_mask[mask]).sum())
        p_value = float(binom.sf(failures - 1, size, alpha))
        rows.append(
            {
                "method": method,
                "name": group.name,
                "test_size": size,
                "test_coverage": cov,
                "test_undercoverage_gap": max(0.0, (1.0 - alpha) - cov),
                "test_miscoverage": 1.0 - cov,
                "failure_enrichment": (1.0 - cov) / alpha if alpha > 0 else np.nan,
                "binom_p_undercoverage": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["bh_q_undercoverage"] = _benjamini_hochberg(frame["binom_p_undercoverage"].to_numpy())
        frame["confirmed_fdr_0.05"] = (frame["test_undercoverage_gap"] > 0) & (frame["bh_q_undercoverage"] <= 0.05)
    return frame


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adjusted[order[i]] = running
    return np.clip(adjusted, 0.0, 1.0)


def _add_target_metrics(row: dict[str, float | str | int], alpha: float) -> None:
    target = 1.0 - alpha
    marginal = float(row["marginal_coverage"])
    row["target_coverage"] = target
    row["marginal_target_error"] = abs(marginal - target)
    row["marginal_undercoverage"] = max(0.0, target - marginal)
    row["marginal_overcoverage"] = max(0.0, marginal - target)


def _add_group_coverage_metrics(
    row: dict[str, float | str | int],
    discovered_groups: list[GroupFunction],
    own_groups: list[GroupFunction] | None,
    z: np.ndarray,
    observed_scores: np.ndarray,
    covered_mask: np.ndarray,
) -> None:
    shared_coverage = _worst_group_coverage(discovered_groups, z, observed_scores, covered_mask)
    row["worst_shared_discovered_slice_coverage"] = shared_coverage
    row["worst_discovered_slice_coverage"] = shared_coverage
    if own_groups is None:
        row["worst_own_pool_coverage"] = float("nan")
    else:
        row["worst_own_pool_coverage"] = _worst_group_coverage(own_groups, z, observed_scores, covered_mask)


def _worst_group_coverage(groups: list[GroupFunction], z: np.ndarray, observed_scores: np.ndarray, covered_mask: np.ndarray, min_size: int = 20) -> float:
    values = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        if mask.sum() >= min_size:
            values.append(float(covered_mask[mask].mean()))
    return float(min(values)) if values else float("nan")


def _worst_binary_matrix_coverage(groups: np.ndarray | None, covered_mask: np.ndarray, min_size: int = 20) -> float:
    if groups is None:
        return float("nan")
    values = []
    for k in range(groups.shape[1]):
        mask = groups[:, k]
        if mask.sum() >= min_size:
            values.append(float(covered_mask[mask].mean()))
    return float(min(values)) if values else float("nan")


def _worst_label_coverage(labels: np.ndarray | None, covered_mask: np.ndarray, min_size: int = 20) -> float:
    if labels is None:
        return float("nan")
    values = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() >= min_size:
            values.append(float(covered_mask[mask].mean()))
    return float(min(values)) if values else float("nan")


def _add_sensitive_group_metrics(
    row: dict[str, float | str | int],
    sensitive_groups: dict[str, np.ndarray] | None,
    covered_mask: np.ndarray,
    alpha: float,
) -> None:
    if not sensitive_groups:
        return
    all_coverages = []
    attribute_macro_coverages = []
    attribute_undercoverage_gaps = []
    attribute_absolute_gaps = []
    for attribute, labels in sensitive_groups.items():
        attribute_token = _column_token(attribute)
        coverages, sizes = _label_coverage_and_sizes(labels, covered_mask)
        row.update({f"known_group_coverage_{attribute_token}_group_{group}": coverage for group, coverage in coverages.items()})
        row.update({f"known_group_size_{attribute_token}_group_{group}": size for group, size in sizes.items()})
        values = list(coverages.values())
        metrics = _coverage_fairness_metrics(values, prefix=f"known_group_{attribute_token}", target=1.0 - alpha)
        row.update(metrics)
        if values:
            attribute_macro_coverages.append(metrics[f"known_group_{attribute_token}_macro_coverage"])
            attribute_undercoverage_gaps.append(metrics[f"known_group_{attribute_token}_mean_undercoverage_gap"])
            attribute_absolute_gaps.append(metrics[f"known_group_{attribute_token}_mean_absolute_coverage_gap"])
        all_coverages.extend(values)
    row.update(_coverage_fairness_metrics(all_coverages, prefix="known_group", target=1.0 - alpha))
    row["known_group_attribute_macro_coverage"] = _safe_mean(attribute_macro_coverages)
    row["known_group_attribute_mean_undercoverage_gap"] = _safe_mean(attribute_undercoverage_gaps)
    row["known_group_attribute_mean_absolute_coverage_gap"] = _safe_mean(attribute_absolute_gaps)


def _add_known_group_metrics(
    row: dict[str, float | str | int],
    labels: np.ndarray | None,
    covered_mask: np.ndarray,
    alpha: float,
) -> None:
    coverages, sizes = _label_coverage_and_sizes(labels, covered_mask)
    row.update({f"known_group_coverage_group_{group}": coverage for group, coverage in coverages.items()})
    row.update({f"known_group_size_group_{group}": size for group, size in sizes.items()})
    row.update(_coverage_fairness_metrics(list(coverages.values()), prefix="known_group", target=1.0 - alpha))


def _label_coverages(prefix: str, labels: np.ndarray | None, covered_mask: np.ndarray, min_size: int = 20) -> dict[str, float]:
    coverages, _ = _label_coverage_and_sizes(labels, covered_mask, min_size)
    return {f"{prefix}_{group}": coverage for group, coverage in coverages.items()}


def _label_coverage_and_sizes(
    labels: np.ndarray | None,
    covered_mask: np.ndarray,
    min_size: int = 20,
) -> tuple[dict[str, float], dict[str, int]]:
    if labels is None:
        return {}, {}
    coverages = {}
    sizes = {}
    for label in np.unique(labels):
        mask = labels == label
        size = int(mask.sum())
        if size >= min_size:
            token = _column_token(label)
            coverages[token] = float(covered_mask[mask].mean())
            sizes[token] = size
    return coverages, sizes


def _coverage_fairness_metrics(values: list[float], prefix: str, target: float) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {
            f"worst_{prefix}_coverage": float("nan"),
            f"{prefix}_macro_coverage": float("nan"),
            f"{prefix}_mean_undercoverage_gap": float("nan"),
            f"{prefix}_mean_absolute_coverage_gap": float("nan"),
            f"{prefix}_coverage_disparity": float("nan"),
        }
    return {
        f"worst_{prefix}_coverage": float(finite.min()),
        f"{prefix}_macro_coverage": float(finite.mean()),
        f"{prefix}_mean_undercoverage_gap": float(np.maximum(0.0, target - finite).mean()),
        f"{prefix}_mean_absolute_coverage_gap": float(np.abs(finite - target).mean()),
        f"{prefix}_coverage_disparity": float(finite.max() - finite.min()),
    }


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _column_token(value: object) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return token or "unknown"

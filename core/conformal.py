from __future__ import annotations

import numpy as np


def inverse_probability_scores(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs


def aps_scores(probs: np.ndarray) -> np.ndarray:
    """Adaptive prediction set scores for every candidate label.

    For each example and label, the score is the cumulative probability mass of
    labels at least as likely as that label. Lower-ranked labels therefore have
    larger scores and enter the prediction set only at larger thresholds.
    """
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores


def score_matrix_from_probs(probs: np.ndarray, score: str) -> np.ndarray:
    if score == "inverse_probability":
        return inverse_probability_scores(probs)
    if score == "aps":
        return aps_scores(probs)
    raise ValueError(f"Unknown conformal score: {score}")


def observable_uncertainty(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs.max(axis=1)


def true_label_scores(score_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    return score_matrix[np.arange(len(y)), y]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.partition(scores, rank - 1)[rank - 1])


def prediction_sets(score_matrix: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    return score_matrix <= np.asarray(thresholds)[:, None] if np.ndim(thresholds) else score_matrix <= thresholds


def covered(score_matrix: np.ndarray, y: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    true_scores = true_label_scores(score_matrix, y)
    return true_scores <= thresholds


def set_sizes(score_matrix: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    return prediction_sets(score_matrix, thresholds).sum(axis=1)


def classwise_quantiles(score_matrix: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    pooled_scores = true_label_scores(score_matrix, y)
    fallback = conformal_quantile(pooled_scores, alpha)
    thresholds = np.full(score_matrix.shape[1], fallback, dtype=np.float64)
    for label in range(score_matrix.shape[1]):
        mask = y == label
        if mask.any():
            thresholds[label] = conformal_quantile(pooled_scores[mask], alpha)
    return thresholds


def covered_classwise(score_matrix: np.ndarray, y: np.ndarray, class_thresholds: np.ndarray) -> np.ndarray:
    true_scores = true_label_scores(score_matrix, y)
    return true_scores <= class_thresholds[y]


def set_sizes_classwise(score_matrix: np.ndarray, class_thresholds: np.ndarray) -> np.ndarray:
    return (score_matrix <= class_thresholds[None, :]).sum(axis=1)


def groupwise_quantiles(score_matrix: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> dict[int, float]:
    true_scores = true_label_scores(score_matrix, y)
    fallback = conformal_quantile(true_scores, alpha)
    thresholds: dict[int, float] = {-1: fallback}
    for group in np.unique(groups):
        mask = groups == group
        thresholds[int(group)] = conformal_quantile(true_scores[mask], alpha) if mask.any() else fallback
    return thresholds


def thresholds_from_groups(groups: np.ndarray, group_thresholds: dict[int, float]) -> np.ndarray:
    fallback = group_thresholds[-1]
    return np.asarray([group_thresholds.get(int(group), fallback) for group in groups], dtype=np.float64)

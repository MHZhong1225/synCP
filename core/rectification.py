from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.conformal import conformal_quantile
from slices.discovery import GroupFunction


@dataclass
class ThresholdModel:
    base_threshold: float
    groups: list[GroupFunction]
    group_thresholds: np.ndarray | None = None
    group_offsets: np.ndarray | None = None
    bin_edges: np.ndarray | None = None
    group_bin_thresholds: np.ndarray | None = None
    mode: str = "standard"

    def thresholds(self, z: np.ndarray, observed_scores: np.ndarray | None = None, score_matrix: np.ndarray | None = None) -> np.ndarray:
        n = len(z)
        q = np.full(n, self.base_threshold, dtype=np.float64)
        scores_for_membership = observed_scores
        if scores_for_membership is None and score_matrix is not None:
            scores_for_membership = score_matrix.min(axis=1)

        if self.mode == "slice_quantile" and self.group_thresholds is not None:
            for group, group_q in zip(self.groups, self.group_thresholds):
                mask = group.membership(z, scores_for_membership)
                q[mask] = np.maximum(q[mask], group_q)
        elif self.mode == "batch_gcp" and self.group_thresholds is not None:
            for group, group_q in zip(self.groups, self.group_thresholds):
                mask = group.membership(z, scores_for_membership)
                q[mask] = np.maximum(q[mask], group_q)
        elif self.mode == "batch_mvp" and self.group_bin_thresholds is not None and self.bin_edges is not None:
            if scores_for_membership is None:
                raise ValueError("BatchMVP thresholds require observable uncertainty scores.")
            bins = np.clip(np.digitize(scores_for_membership, self.bin_edges[1:-1], right=False), 0, self.group_bin_thresholds.shape[1] - 1)
            for group_idx, group in enumerate(self.groups):
                group_mask = group.membership(z, scores_for_membership)
                if not group_mask.any():
                    continue
                for bin_idx in range(self.group_bin_thresholds.shape[1]):
                    mask = group_mask & (bins == bin_idx)
                    if mask.any():
                        q[mask] = np.maximum(q[mask], self.group_bin_thresholds[group_idx, bin_idx])
        elif self.mode == "pool_mvp" and self.group_offsets is not None:
            for group, offset in zip(self.groups, self.group_offsets):
                if offset <= 0:
                    continue
                mask = group.membership(z, scores_for_membership)
                q[mask] = np.maximum(q[mask], self.base_threshold + offset)
            q = np.clip(q, 0.0, 1.0)
        return q


def fit_standard(base_threshold: float) -> ThresholdModel:
    return ThresholdModel(base_threshold=base_threshold, groups=[], mode="standard")


def fit_slice_quantile(
    base_threshold: float,
    groups: list[GroupFunction],
    z: np.ndarray,
    true_scores: np.ndarray,
    alpha: float,
    observed_scores: np.ndarray | None = None,
    min_size: int = 20,
) -> ThresholdModel:
    thresholds = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        if mask.sum() < min_size:
            thresholds.append(base_threshold)
        else:
            thresholds.append(max(base_threshold, conformal_quantile(true_scores[mask], alpha)))
    return ThresholdModel(base_threshold=base_threshold, groups=groups, group_thresholds=np.asarray(thresholds), mode="slice_quantile")


def fit_batch_gcp(
    base_threshold: float,
    groups: list[GroupFunction],
    z: np.ndarray,
    true_scores: np.ndarray,
    alpha: float,
    observed_scores: np.ndarray | None = None,
    eta: float = 0.01,
    max_iter: int = 200,
    min_size: int = 20,
) -> ThresholdModel:
    """BatchGCP-style iterative group coverage repair.

    This implements a practical batch multivalid update: repeatedly find a group
    whose current empirical miscoverage exceeds alpha + eta, then add a positive
    threshold correction on that group. Overlapping groups receive additive
    threshold correction on that group. Overlapping groups use the maximum
    learned threshold among active groups.
    """
    memberships = np.column_stack([group.membership(z, observed_scores) for group in groups]) if groups else np.zeros((len(z), 0), dtype=bool)
    q = np.full(len(z), base_threshold, dtype=np.float64)
    group_thresholds = np.full(len(groups), base_threshold, dtype=np.float64)
    for k in range(len(groups)):
        mask = memberships[:, k]
        if mask.sum() >= min_size:
            group_thresholds[k] = max(base_threshold, conformal_quantile(true_scores[mask], alpha))
            q[mask] = np.maximum(q[mask], group_thresholds[k])

    for _ in range(max_iter):
        violations = []
        for k in range(len(groups)):
            mask = memberships[:, k]
            if mask.sum() < min_size:
                continue
            misc = float((true_scores[mask] > q[mask]).mean())
            violations.append((misc - alpha, k, mask))
        if not violations:
            break
        gap, k, mask = max(violations, key=lambda item: item[0])
        if gap <= eta:
            break
        new_threshold = conformal_quantile(true_scores[mask], alpha)
        if new_threshold <= group_thresholds[k]:
            break
        group_thresholds[k] = new_threshold
        affected = memberships[:, k]
        q[affected] = np.maximum(q[affected], new_threshold)

    return ThresholdModel(base_threshold=base_threshold, groups=groups, group_thresholds=group_thresholds, mode="batch_gcp")


def fit_pool_mvp(
    base_threshold: float,
    groups: list[GroupFunction],
    z: np.ndarray,
    true_scores: np.ndarray,
    alpha: float,
    observed_scores: np.ndarray | None = None,
    step_size: float = 0.02,
    tolerance: float = 0.0,
    max_iter: int = 50,
    min_size: int = 20,
    max_active_groups: int = 8,
) -> ThresholdModel:
    """Iterative multivalid-style recalibration over a discovered slice pool."""
    if not groups:
        return ThresholdModel(base_threshold=base_threshold, groups=[], group_offsets=np.zeros(0), mode="pool_mvp")

    memberships = np.column_stack([group.membership(z, observed_scores) for group in groups])
    offsets = np.zeros(len(groups), dtype=np.float64)
    q = np.full(len(true_scores), base_threshold, dtype=np.float64)

    for _ in range(max_iter):
        gap, group_idx = _worst_violation(memberships, true_scores, q, alpha, min_size)
        if group_idx is None or gap <= tolerance:
            break
        if offsets[group_idx] <= 0 and np.count_nonzero(offsets > 0) >= max_active_groups:
            break
        mask = memberships[:, group_idx]
        target = max(base_threshold, conformal_quantile(true_scores[mask], alpha))
        step = min(step_size, max(0.0, target - (base_threshold + offsets[group_idx])))
        if step <= 0:
            break
        offsets[group_idx] += step
        q = _max_offset_thresholds(base_threshold, memberships, offsets)

    return ThresholdModel(base_threshold=base_threshold, groups=groups, group_offsets=offsets, mode="pool_mvp")


def _worst_violation(
    memberships: np.ndarray,
    true_scores: np.ndarray,
    thresholds: np.ndarray,
    alpha: float,
    min_size: int,
) -> tuple[float, int | None]:
    best_gap = 0.0
    best_idx = None
    for idx in range(memberships.shape[1]):
        mask = memberships[:, idx]
        if mask.sum() < min_size:
            continue
        miscoverage = float((true_scores[mask] > thresholds[mask]).mean())
        gap = miscoverage - alpha
        if gap > best_gap:
            best_gap = gap
            best_idx = idx
    return best_gap, best_idx


def _max_offset_thresholds(base_threshold: float, memberships: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    q = np.full(memberships.shape[0], base_threshold, dtype=np.float64)
    for idx, offset in enumerate(offsets):
        if offset <= 0:
            continue
        q[memberships[:, idx]] = np.maximum(q[memberships[:, idx]], base_threshold + offset)
    return q


def fit_batch_mvp(
    base_threshold: float,
    groups: list[GroupFunction],
    z: np.ndarray,
    true_scores: np.ndarray,
    alpha: float,
    observed_scores: np.ndarray | None = None,
    num_bins: int = 4,
    min_size: int = 20,
) -> ThresholdModel:
    """Practical BatchMVP-style repair over group x uncertainty-bin cells.

    This is a finite-grid multivalid variant: for each discovered group and each
    observable uncertainty bin, learn a threshold on the MV calibration split.
    """
    if observed_scores is None:
        raise ValueError("BatchMVP requires observed uncertainty scores.")
    if not groups:
        return ThresholdModel(base_threshold=base_threshold, groups=[], mode="batch_mvp")
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(observed_scores, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bins = np.clip(np.digitize(observed_scores, bin_edges[1:-1], right=False), 0, num_bins - 1)
    thresholds = np.full((len(groups), num_bins), base_threshold, dtype=np.float64)

    for group_idx, group in enumerate(groups):
        group_mask = group.membership(z, observed_scores)
        for bin_idx in range(num_bins):
            mask = group_mask & (bins == bin_idx)
            if mask.sum() >= min_size:
                thresholds[group_idx, bin_idx] = max(base_threshold, conformal_quantile(true_scores[mask], alpha))
    return ThresholdModel(
        base_threshold=base_threshold,
        groups=groups,
        bin_edges=bin_edges,
        group_bin_thresholds=thresholds,
        mode="batch_mvp",
    )

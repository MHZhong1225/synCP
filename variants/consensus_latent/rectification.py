from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.conformal import conformal_quantile
from variants.consensus_latent.consensus import ConsensusGroup


@dataclass
class SoftConsensusModel:
    base_threshold: float
    groups: list[ConsensusGroup]
    group_thresholds: np.ndarray

    def thresholds(
        self,
        z: np.ndarray,
        observed_scores: np.ndarray | None = None,
        score_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        q = np.full(len(z), self.base_threshold, dtype=np.float64)
        for group, group_threshold in zip(self.groups, self.group_thresholds):
            weight = group.soft_membership(z, observed_scores)
            interpolated = self.base_threshold + weight * (group_threshold - self.base_threshold)
            q = np.maximum(q, interpolated)
        return q


def fit_soft_consensus(
    base_threshold: float,
    groups: list[ConsensusGroup],
    z: np.ndarray,
    true_scores: np.ndarray,
    alpha: float,
    observed_scores: np.ndarray | None = None,
    min_size: int = 20,
) -> SoftConsensusModel:
    thresholds = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        threshold = (
            max(base_threshold, conformal_quantile(true_scores[mask], alpha))
            if mask.sum() >= min_size
            else base_threshold
        )
        thresholds.append(threshold)
    return SoftConsensusModel(
        base_threshold=base_threshold,
        groups=groups,
        group_thresholds=np.asarray(thresholds, dtype=np.float64),
    )

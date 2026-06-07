from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slices.discovery import GroupFunction


@dataclass(frozen=True)
class CandidateEvidence:
    group: GroupFunction
    source_seed: int
    source_index: int
    frequency: float
    median_gap: float
    discovery_size: int
    anchor_size: int
    ee_score: float
    anchor_membership: np.ndarray


def aggregate_ee_slices(
    groups_by_seed: dict[int, list[GroupFunction]],
    discovery_embeddings: np.ndarray,
    discovery_scores: np.ndarray,
    discovery_failures: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_scores: np.ndarray,
    alpha: float,
    jaccard_threshold: float = 0.50,
    frequency_threshold: float = 0.50,
    max_groups: int = 8,
    dedup_jaccard: float = 0.80,
) -> tuple[list[GroupFunction], list[GroupFunction], list[CandidateEvidence]]:
    """Aggregate repeated discovery runs using frequency and EE-style evidence.

    EE-Frequency ranks candidates by the fraction of runs that contain a
    matching slice. EE-Score additionally ranks by the median positive
    undercoverage gap among matched slices and candidate support.
    """
    memberships = {
        seed: _membership_matrix(groups, anchor_embeddings, anchor_scores)
        for seed, groups in groups_by_seed.items()
    }
    discovery_memberships = {
        seed: _membership_matrix(groups, discovery_embeddings, discovery_scores)
        for seed, groups in groups_by_seed.items()
    }
    gaps = {
        seed: _coverage_gaps(matrix, discovery_failures, alpha)
        for seed, matrix in discovery_memberships.items()
    }

    evidence = []
    for source_seed, groups in groups_by_seed.items():
        for source_index, group in enumerate(groups):
            anchor_mask = memberships[source_seed][:, source_index]
            discovery_mask = discovery_memberships[source_seed][:, source_index]
            matched_gaps = []
            matched_runs = 0
            for seed in groups_by_seed:
                best_index, best_similarity = _best_match(anchor_mask, memberships[seed])
                if best_index is None or best_similarity < jaccard_threshold:
                    continue
                matched_runs += 1
                matched_gaps.append(gaps[seed][best_index])
            frequency = matched_runs / max(len(groups_by_seed), 1)
            median_gap = float(np.median(matched_gaps)) if matched_gaps else 0.0
            discovery_size = int(discovery_mask.sum())
            anchor_size = int(anchor_mask.sum())
            score = frequency * median_gap * np.log1p(discovery_size)
            evidence.append(
                CandidateEvidence(
                    group=group,
                    source_seed=source_seed,
                    source_index=source_index,
                    frequency=frequency,
                    median_gap=median_gap,
                    discovery_size=discovery_size,
                    anchor_size=anchor_size,
                    ee_score=float(score),
                    anchor_membership=anchor_mask,
                )
            )

    eligible = [item for item in evidence if item.frequency >= frequency_threshold]
    frequency_ranked = sorted(
        eligible,
        key=lambda item: (item.frequency, item.median_gap, item.discovery_size),
        reverse=True,
    )
    score_ranked = sorted(
        eligible,
        key=lambda item: (item.ee_score, item.frequency, item.discovery_size),
        reverse=True,
    )
    frequency_groups = [item.group for item in _deduplicate(frequency_ranked, max_groups, dedup_jaccard)]
    score_groups = [item.group for item in _deduplicate(score_ranked, max_groups, dedup_jaccard)]
    return frequency_groups, score_groups, evidence


def _membership_matrix(groups: list[GroupFunction], embeddings: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if not groups:
        return np.zeros((len(embeddings), 0), dtype=bool)
    return np.column_stack([group.membership(embeddings, scores) for group in groups])


def _coverage_gaps(memberships: np.ndarray, failures: np.ndarray, alpha: float) -> np.ndarray:
    values = np.zeros(memberships.shape[1], dtype=np.float64)
    for index in range(memberships.shape[1]):
        mask = memberships[:, index]
        if mask.any():
            values[index] = max(0.0, float(failures[mask].mean()) - alpha)
    return values


def _best_match(candidate: np.ndarray, memberships: np.ndarray) -> tuple[int | None, float]:
    if memberships.shape[1] == 0:
        return None, 0.0
    similarities = np.asarray([_jaccard(candidate, memberships[:, index]) for index in range(memberships.shape[1])])
    best = int(np.argmax(similarities))
    return best, float(similarities[best])


def _deduplicate(
    ranked: list[CandidateEvidence],
    max_groups: int,
    dedup_jaccard: float,
) -> list[CandidateEvidence]:
    selected = []
    for candidate in ranked:
        if any(_jaccard(candidate.anchor_membership, old.anchor_membership) > dedup_jaccard for old in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_groups:
            break
    return selected


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering

from slices.discovery import GroupFunction


@dataclass
class ConsensusGroup:
    name: str
    members: tuple[GroupFunction, ...]
    member_run_ids: tuple[str, ...]
    member_threshold: float = 0.50

    def soft_membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        memberships = np.column_stack([group.membership(z, observed_scores) for group in self.members])
        run_votes = []
        for run_id in sorted(set(self.member_run_ids)):
            indices = [index for index, candidate_run in enumerate(self.member_run_ids) if candidate_run == run_id]
            run_votes.append(memberships[:, indices].max(axis=1))
        return np.column_stack(run_votes).mean(axis=1)

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        return self.soft_membership(z, observed_scores) >= self.member_threshold


@dataclass(frozen=True)
class CandidateNode:
    method: str
    seed: int
    group: GroupFunction
    anchor_membership: np.ndarray


@dataclass(frozen=True)
class ConsensusEvidence:
    group: ConsensusGroup
    cluster_id: int
    num_members: int
    num_methods: int
    num_method_runs: int
    within_method_stability: float
    cross_method_diversity: float
    discovery_size: int
    anchor_size: int
    observed_gap: float
    gap_standard_error: float
    undercoverage_probability: float
    reliability: float
    methods: str


def discover_consensus_groups(
    groups_by_method_seed: dict[tuple[str, int], list[GroupFunction]],
    discovery_embeddings: np.ndarray,
    discovery_scores: np.ndarray,
    discovery_failures: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_scores: np.ndarray,
    alpha: float,
    effect_threshold: float,
    similarity_threshold: float = 0.30,
    member_threshold: float = 0.50,
    min_cluster_members: int = 2,
    min_within_stability: float = 0.30,
    posterior_threshold: float = 0.50,
    method_bonus: float = 0.0,
    max_groups: int = 8,
    dedup_jaccard: float = 0.80,
) -> tuple[list[ConsensusGroup], list[ConsensusEvidence], list[CandidateNode]]:
    nodes = _candidate_nodes(groups_by_method_seed, anchor_embeddings, anchor_scores)
    if not nodes:
        return [], [], []
    labels = _cluster_memberships(nodes, similarity_threshold)
    methods = sorted({method for method, _ in groups_by_method_seed})
    seeds_by_method = {
        method: {seed for candidate_method, seed in groups_by_method_seed if candidate_method == method}
        for method in methods
    }

    evidence = []
    for cluster_id in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster_id)
        if len(indices) < min_cluster_members:
            continue
        cluster_nodes = [nodes[index] for index in indices]
        group = ConsensusGroup(
            name=f"consensus_latent_{cluster_id}",
            members=tuple(node.group for node in cluster_nodes),
            member_run_ids=tuple(f"{node.method}:{node.seed}" for node in cluster_nodes),
            member_threshold=member_threshold,
        )
        method_runs = {(node.method, node.seed) for node in cluster_nodes}
        cluster_methods = {node.method for node in cluster_nodes}
        within = max(
            len({seed for candidate_method, seed in method_runs if candidate_method == method})
            / max(len(seeds_by_method[method]), 1)
            for method in cluster_methods
        )
        diversity = len(cluster_methods) / len(methods) if len(methods) > 1 else 0.0
        discovery_mask = group.membership(discovery_embeddings, discovery_scores)
        anchor_mask = group.membership(anchor_embeddings, anchor_scores)
        gap, standard_error, probability = _undercoverage_evidence(
            discovery_mask,
            discovery_failures,
            alpha,
            effect_threshold,
        )
        reliability = probability * within * np.log1p(discovery_mask.sum()) * (1.0 + method_bonus * diversity)
        evidence.append(
            ConsensusEvidence(
                group=group,
                cluster_id=int(cluster_id),
                num_members=len(cluster_nodes),
                num_methods=len(cluster_methods),
                num_method_runs=len(method_runs),
                within_method_stability=float(within),
                cross_method_diversity=float(diversity),
                discovery_size=int(discovery_mask.sum()),
                anchor_size=int(anchor_mask.sum()),
                observed_gap=gap,
                gap_standard_error=standard_error,
                undercoverage_probability=probability,
                reliability=float(reliability),
                methods="|".join(sorted(cluster_methods)),
            )
        )

    ranked = sorted(
        [
            item
            for item in evidence
            if item.discovery_size > 0
            and item.within_method_stability >= min_within_stability
            and item.undercoverage_probability >= posterior_threshold
        ],
        key=lambda item: item.reliability,
        reverse=True,
    )
    selected = _deduplicate(ranked, anchor_embeddings, anchor_scores, max_groups, dedup_jaccard)
    return [item.group for item in selected], evidence, nodes


def _candidate_nodes(
    groups_by_method_seed: dict[tuple[str, int], list[GroupFunction]],
    anchor_embeddings: np.ndarray,
    anchor_scores: np.ndarray,
) -> list[CandidateNode]:
    nodes = []
    for (method, seed), groups in groups_by_method_seed.items():
        for group in groups:
            nodes.append(
                CandidateNode(
                    method=method,
                    seed=seed,
                    group=group,
                    anchor_membership=group.membership(anchor_embeddings, anchor_scores),
                )
            )
    return nodes


def _cluster_memberships(nodes: list[CandidateNode], similarity_threshold: float) -> np.ndarray:
    memberships = np.column_stack([node.anchor_membership for node in nodes])
    distance = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            similarity = _jaccard(memberships[:, left], memberships[:, right])
            distance[left, right] = distance[right, left] = 1.0 - similarity
    if len(nodes) == 1:
        return np.zeros(1, dtype=np.int64)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
    )
    return model.fit_predict(distance)


def _undercoverage_evidence(
    mask: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    effect_threshold: float,
) -> tuple[float, float, float]:
    n = int(mask.sum())
    if n == 0:
        return -alpha, float("inf"), 0.0
    failures_in_slice = int(failures[mask].sum())
    rate = failures_in_slice / n
    smoothed_rate = (failures_in_slice + 0.5) / (n + 1.0)
    standard_error = float(np.sqrt(smoothed_rate * (1.0 - smoothed_rate) / (n + 1.0)))
    gap = float(rate - alpha)
    probability = float(norm.sf((effect_threshold - gap) / max(standard_error, 1e-8)))
    return gap, standard_error, probability


def _deduplicate(
    ranked: list[ConsensusEvidence],
    anchor_embeddings: np.ndarray,
    anchor_scores: np.ndarray,
    max_groups: int,
    dedup_jaccard: float,
) -> list[ConsensusEvidence]:
    selected, selected_masks = [], []
    for candidate in ranked:
        mask = candidate.group.membership(anchor_embeddings, anchor_scores)
        if any(_jaccard(mask, old) > dedup_jaccard for old in selected_masks):
            continue
        selected.append(candidate)
        selected_masks.append(mask)
        if len(selected) >= max_groups:
            break
    return selected


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)

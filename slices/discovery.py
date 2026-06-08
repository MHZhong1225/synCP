from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

CONFIDENCE_Z = 1.96
MAX_DEDUP_JACCARD = 0.80


class GroupFunction(Protocol):
    name: str

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        ...


@dataclass
class GaussianComponentGroup:
    name: str
    scaler: StandardScaler
    gmm: GaussianMixture
    component: int
    threshold: float

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        transformed = self.scaler.transform(z)
        resp = self.gmm.predict_proba(transformed)[:, self.component]
        return resp >= self.threshold


@dataclass
class GaussianRadiusGroup:
    name: str
    scaler: StandardScaler
    center: np.ndarray
    radius: float

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        transformed = self.scaler.transform(z)
        distances = np.linalg.norm(transformed - self.center[None, :], axis=1)
        return distances <= self.radius


@dataclass
class KMeansGroup:
    name: str
    scaler: StandardScaler
    kmeans: KMeans
    cluster: int

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        transformed = self.scaler.transform(z)
        return self.kmeans.predict(transformed) == self.cluster


@dataclass
class UndercoverageRiskGroup:
    name: str
    scaler: StandardScaler
    model: LogisticRegression
    threshold: float

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        risk = self.model.predict_proba(self.scaler.transform(z))[:, 1]
        return risk >= self.threshold


@dataclass
class HighScoreGroup:
    name: str
    threshold: float

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        if observed_scores is None:
            raise ValueError("HighScoreGroup requires observable uncertainty scores.")
        return observed_scores >= self.threshold


@dataclass
class RandomHalfspaceGroup:
    name: str
    center: np.ndarray
    direction: np.ndarray
    threshold: float
    side: str

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        values = (z - self.center) @ self.direction
        if self.side == "high":
            return values >= self.threshold
        return values <= self.threshold


@dataclass
class PredictedClassGroup:
    base_group: GroupFunction
    predicted_class: int

    @property
    def name(self) -> str:
        return f"{self.base_group.name}_pred{self.predicted_class}"

    def membership(self, z: np.ndarray, observed_scores: np.ndarray | None = None) -> np.ndarray:
        if observed_scores is None or np.ndim(observed_scores) != 2:
            raise ValueError("PredictedClassGroup requires class probability scores.")
        uncertainty = 1.0 - observed_scores.max(axis=1)
        return self.base_group.membership(z, uncertainty) & (observed_scores.argmax(axis=1) == self.predicted_class)


@dataclass(frozen=True)
class SliceDiagnostics:
    method: str
    name: str
    size: int
    miscoverage: float
    enrichment: float


def _filter_groups(
    groups: list[GroupFunction],
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    method: str,
    min_size: int,
    epsilon: float,
    enrichment: float,
    max_groups: int,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    base_rate = max(float(failures.mean()), 1e-8)
    selected: list[tuple[float, GroupFunction, SliceDiagnostics, np.ndarray]] = []
    for group in groups:
        mask = group.membership(z, observed_scores)
        size = int(mask.sum())
        if size < min_size:
            continue
        rate = float(failures[mask].mean())
        enrich = rate / base_rate
        gap = rate - alpha
        if gap < epsilon or enrich < enrichment:
            continue
        diag = SliceDiagnostics(method=method, name=group.name, size=size, miscoverage=rate, enrichment=enrich)
        selected.append((gap, group, diag, mask))

    selected.sort(key=lambda item: item[0], reverse=True)
    kept: list[GroupFunction] = []
    diags: list[SliceDiagnostics] = []
    kept_masks: list[np.ndarray] = []
    for _, group, diag, mask in selected:
        if any(_jaccard(mask, old) > MAX_DEDUP_JACCARD for old in kept_masks):
            continue
        kept.append(group)
        diags.append(diag)
        kept_masks.append(mask)
        if len(kept) >= max_groups:
            break
    return kept, diags


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def select_group_pool(
    groups_by_method: dict[str, list[GroupFunction]],
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    methods: tuple[str, ...] = ("gmm_unaware", "kmeans", "coverage_domino", "random_halfspace"),
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
) -> list[GroupFunction]:
    """Select a deduplicated pool of coverage-risk slices across generators."""
    base_rate = max(float(failures.mean()), 1e-8)
    candidates_by_method: dict[str, list[tuple[float, GroupFunction, np.ndarray]]] = {}
    for method in methods:
        method_candidates: list[tuple[float, GroupFunction, np.ndarray]] = []
        for group in groups_by_method.get(method, []):
            mask = group.membership(z, observed_scores)
            size = int(mask.sum())
            if size < min_size:
                continue
            score = _pool_candidate_score(
                mask,
                failures,
                alpha,
                base_rate,
                epsilon,
                enrichment,
            )
            if score is None:
                continue
            method_candidates.append((score, group, mask))
        method_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates_by_method[method] = method_candidates

    selected: list[GroupFunction] = []
    selected_masks: list[np.ndarray] = []
    while len(selected) < max_groups:
        added_this_round = False
        for method in methods:
            if _select_next_candidate(candidates_by_method[method], selected, selected_masks):
                added_this_round = True
                if len(selected) >= max_groups:
                    break
        if not added_this_round:
            break
    return selected


def _pool_candidate_score(
    mask: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    base_rate: float,
    epsilon: float,
    enrichment: float,
) -> float | None:
    size = int(mask.sum())
    rate = float(failures[mask].mean())
    gap = rate - alpha
    enrich = rate / base_rate
    if gap < epsilon or enrich < enrichment:
        return None
    return _coverage_risk_score(rate, alpha, size)


def class_conditional_groups(groups: list[GroupFunction], num_classes: int) -> list[GroupFunction]:
    conditional: list[GroupFunction] = []
    for group in groups:
        conditional.append(group)
        for predicted_class in range(num_classes):
            conditional.append(PredictedClassGroup(group, predicted_class))
    return conditional


def _coverage_risk_score(miscoverage: float, alpha: float, size: int) -> float:
    standard_error = np.sqrt(max(miscoverage * (1.0 - miscoverage), 1e-12) / size)
    return (miscoverage - alpha) - CONFIDENCE_Z * standard_error


def _select_next_candidate(
    candidates: list[tuple[float, GroupFunction, np.ndarray]],
    selected: list[GroupFunction],
    selected_masks: list[np.ndarray],
) -> bool:
    while candidates:
        _, group, mask = candidates.pop(0)
        if any(_jaccard(mask, old) > MAX_DEDUP_JACCARD for old in selected_masks):
            continue
        selected.append(group)
        selected_masks.append(mask)
        return True
    return False


def discover_coverage_domino(
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    n_components: int = 30,
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
    responsibility_threshold: float = 0.35,
    failure_weight: int = 4,
    radius_quantile: float = 0.85,
    seed: int = 7,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    """Coverage-aware Domino-style discovery.

    The mixture uses embedding only, because the final slice membership must be
    evaluable for unlabeled future examples. Coverage failures are used only for
    selecting components after the mixture is fit.
    """
    scaler = StandardScaler().fit(z)
    transformed = scaler.transform(z)
    if failure_weight > 1 and failures.any():
        fit_transformed = np.concatenate([transformed, np.repeat(transformed[failures], failure_weight - 1, axis=0)], axis=0)
    else:
        fit_transformed = transformed
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=seed,
        max_iter=500,
    ).fit(fit_transformed)
    candidates: list[GroupFunction] = []
    responsibilities = gmm.predict_proba(transformed)
    for k in range(n_components):
        candidates.append(
            GaussianComponentGroup(
            name=f"coverage_domino_c{k}",
            scaler=scaler,
            gmm=gmm,
            component=k,
            threshold=responsibility_threshold,
            )
        )
        assigned = responsibilities[:, k] >= responsibility_threshold
        if assigned.sum() >= max(5, min_size // 2):
            center = gmm.means_[k]
            distances = np.linalg.norm(transformed[assigned] - center[None, :], axis=1)
            radius = float(np.quantile(distances, radius_quantile))
            candidates.append(GaussianRadiusGroup(name=f"coverage_domino_radius_c{k}", scaler=scaler, center=center, radius=radius))
    return _filter_groups(candidates, z, observed_scores, failures, alpha, "coverage_domino", min_size, epsilon, enrichment, max_groups)


def discover_error_domino(
    z: np.ndarray,
    observed_scores: np.ndarray,
    classification_errors: np.ndarray,
    n_components: int = 30,
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
    responsibility_threshold: float = 0.35,
    error_weight: int = 4,
    seed: int = 7,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    """Domino-style baseline driven by classification errors, not coverage failures."""
    scaler = StandardScaler().fit(z)
    transformed = scaler.transform(z)
    if error_weight > 1 and classification_errors.any():
        fit_transformed = np.concatenate([transformed, np.repeat(transformed[classification_errors], error_weight - 1, axis=0)], axis=0)
    else:
        fit_transformed = transformed
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=seed,
        max_iter=500,
    ).fit(fit_transformed)
    candidates = [
        GaussianComponentGroup(
            name=f"error_domino_c{k}",
            scaler=scaler,
            gmm=gmm,
            component=k,
            threshold=responsibility_threshold,
        )
        for k in range(n_components)
    ]
    base_error = float(classification_errors.mean())
    return _filter_groups(candidates, z, observed_scores, classification_errors, base_error, "error_domino", min_size, epsilon, enrichment, max_groups)


def discover_gmm_unaware(
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    n_components: int = 30,
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
    responsibility_threshold: float = 0.35,
    seed: int = 7,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    """GMM clustering baseline that does not use coverage failures when fitting."""
    scaler = StandardScaler().fit(z)
    transformed = scaler.transform(z)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=seed,
        max_iter=500,
    ).fit(transformed)
    candidates = [
        GaussianComponentGroup(
            name=f"gmm_unaware_c{k}",
            scaler=scaler,
            gmm=gmm,
            component=k,
            threshold=responsibility_threshold,
        )
        for k in range(n_components)
    ]
    return _filter_groups(candidates, z, observed_scores, failures, alpha, "gmm_unaware", min_size, epsilon, enrichment, max_groups)


def discover_kmeans(
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    n_clusters: int = 30,
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
    seed: int = 7,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    scaler = StandardScaler().fit(z)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(scaler.transform(z))
    candidates = [KMeansGroup(name=f"kmeans_c{k}", scaler=scaler, kmeans=kmeans, cluster=k) for k in range(n_clusters)]
    return _filter_groups(candidates, z, observed_scores, failures, alpha, "kmeans", min_size, epsilon, enrichment, max_groups)


def discover_undercoverage_risk(
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    quantiles: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
    seed: int = 7,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    """Directly score future examples by learned conformal-failure risk."""
    if np.unique(failures).size < 2:
        return [], []
    scaler = StandardScaler().fit(z)
    transformed = scaler.transform(z)
    model = _fit_undercoverage_risk_model(transformed, failures, seed)
    risk = model.predict_proba(transformed)[:, 1]
    candidates = [
        UndercoverageRiskGroup(
            name=f"undercoverage_risk_q{quantile:.2f}",
            scaler=scaler,
            model=model,
            threshold=float(np.quantile(risk, quantile)),
        )
        for quantile in quantiles
    ]
    return _filter_groups(candidates, z, observed_scores, failures, alpha, "undercoverage_risk", min_size, epsilon, enrichment, max_groups)


def _fit_undercoverage_risk_model(
    transformed: np.ndarray,
    failures: np.ndarray,
    seed: int,
) -> LogisticRegression:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(failures)
    negative = np.flatnonzero(~failures)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Undercoverage risk model requires both covered and uncovered examples.")
    pos_take = max(1, int(round(len(transformed) * len(positive) / len(transformed))))
    neg_take = max(1, len(transformed) - pos_take)
    bootstrap = np.concatenate(
        [
            rng.choice(positive, size=pos_take, replace=True),
            rng.choice(negative, size=neg_take, replace=True),
        ]
    )
    rng.shuffle(bootstrap)
    return LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=500,
        random_state=seed,
        solver="liblinear",
    ).fit(transformed[bootstrap], failures[bootstrap])


def _region_count(mask: np.ndarray, max_regions: int, min_size: int) -> int:
    high_risk_size = int(mask.sum())
    if high_risk_size < 2 * min_size:
        return 0
    return max(2, min(max_regions, high_risk_size // min_size))


def discover_high_score(
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    quantiles: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    candidates = [HighScoreGroup(name=f"high_score_q{q:.2f}", threshold=float(np.quantile(observed_scores, q))) for q in quantiles]
    return _filter_groups(candidates, z, observed_scores, failures, alpha, "high_score", min_size, epsilon, enrichment, max_groups)


def discover_random_halfspace(
    z: np.ndarray,
    observed_scores: np.ndarray,
    failures: np.ndarray,
    alpha: float,
    num_directions: int = 30,
    quantiles: tuple[float, ...] = (0.10, 0.20, 0.80, 0.90),
    min_size: int = 40,
    epsilon: float = 0.03,
    enrichment: float = 1.20,
    max_groups: int = 8,
    seed: int = 7,
) -> tuple[list[GroupFunction], list[SliceDiagnostics]]:
    """Discover undercoverage slices by sampling random embedding halfspaces."""
    rng = np.random.default_rng(seed)
    candidates: list[GroupFunction] = []
    center = z.mean(axis=0)
    z_centered = z - center
    for j in range(num_directions):
        direction = rng.normal(size=z.shape[1])
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        values = z_centered @ direction
        for q in quantiles:
            threshold = float(np.quantile(values, q))
            side = "low" if q < 0.5 else "high"
            candidates.append(
                RandomHalfspaceGroup(
                    name=f"random_halfspace_{j}_{side}_{q:.2f}",
                    center=center,
                    direction=direction,
                    threshold=threshold,
                    side=side,
                )
            )
    return _filter_groups(candidates, z, observed_scores, failures, alpha, "random_halfspace", min_size, epsilon, enrichment, max_groups)

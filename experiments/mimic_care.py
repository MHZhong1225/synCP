from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer

from core.conformal import classwise_quantiles, conformal_quantile, covered, groupwise_quantiles, observable_uncertainty, score_matrix_from_probs, set_sizes, thresholds_from_groups, true_label_scores
from core.metrics import evaluate_classwise_model, evaluate_classifier, evaluate_groupwise_model, evaluate_threshold_model
from core.rectification import fit_batch_gcp, fit_pool_mvp, fit_slice_quantile, fit_standard
from data.mimic_care import MimicCareData, load_mimic_care_experiment
from experiments.images import DiscoveryInputs, discover_slice_method
from reports.output import print_results, write_results
from slices.discovery import GroupFunction, class_conditional_groups, select_group_pool


POOL_METHODS = ("gmm_unaware", "kmeans", "coverage_domino", "random_halfspace")
DISCOVERY_METHODS = ("random_halfspace", "coverage_domino", "error_domino", "gmm_unaware", "kmeans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIMIC ED-to-ICU time-stratum text experiment.")
    parser.add_argument("--mimic-iv-root", type=Path, default=Path("../dataset/mimic-iv-3.1"))
    parser.add_argument("--mimic-ed-root", type=Path, default=Path("../dataset/mimic-iv-ed-2.2"))
    parser.add_argument("--outdir", type=Path, default=Path("results/mimic_care"))
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--feature-mode", choices=["text", "text_triage"], default="text_triage")
    parser.add_argument("--known-group", choices=["race", "gender", "age_group", "arrival_transport", "acuity_group", "insurance", "language", "marital_status", "ed_disposition", "race_acuity", "race_gender", "race_age", "gender_age"], default="race_acuity")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--components", type=int, default=24)
    parser.add_argument("--min-size", type=int, default=40)
    parser.add_argument("--rectifier-min-size", type=int, default=20)
    parser.add_argument("--pool-groups", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--enrichment", type=float, default=1.10)
    parser.add_argument("--cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_mimic_care_experiment(
        args.mimic_iv_root,
        args.mimic_ed_root,
        known_group=args.known_group,
        feature_mode=args.feature_mode,
        seed=args.seed,
    )
    _print_settings(args, data)
    probs, embeddings = _load_or_fit_features(args, data)
    write_results(evaluate_classifier(probs[data.test_idx], data.labels[data.test_idx], data.class_names, data.known_group[data.test_idx]), args.outdir / "classification_metrics.csv")
    write_results(_group_time_strata(data), args.outdir / "group_time_strata.csv")
    summary, group_coverage = run_conformal(args, data, probs, embeddings)
    cleaned = write_results(summary, args.outdir / "summary.csv")
    write_results(group_coverage, args.outdir / "group_coverage.csv")
    print_results(cleaned)
    print(f"\nwrote: {args.outdir / 'summary.csv'}")


def _load_or_fit_features(args: argparse.Namespace, data: MimicCareData) -> tuple[np.ndarray, np.ndarray]:
    cache_name = "mimic_care.npz" if args.feature_mode == "text_triage" else f"mimic_care_{args.feature_mode}.npz"
    path = Path("results/npz") / cache_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.cache and path.exists():
        cached = np.load(path)
        if "feature_mode" not in cached or str(cached["feature_mode"]) != args.feature_mode:
            raise ValueError("Cached MIMIC care features were created for a different feature mode.")
        return cached["probs"], cached["embeddings"]
    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=8000, sublinear_tf=True)),
        ]
    )
    train_matrix = vectorizer.fit_transform(data.documents[data.train_idx])
    matrix = vectorizer.transform(data.documents)
    dimensions = min(128, train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=dimensions, random_state=args.seed)
    reducer.fit(train_matrix)
    normalizer = Normalizer().fit(reducer.transform(train_matrix))
    embeddings = normalizer.transform(reducer.transform(matrix)).astype(np.float32)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    classifier.fit(train_matrix, data.labels[data.train_idx])
    probs = classifier.predict_proba(matrix)
    np.savez_compressed(path, probs=probs, embeddings=embeddings, feature_mode=np.asarray(args.feature_mode))
    return probs, embeddings


def run_conformal(args: argparse.Namespace, data: MimicCareData, probs: np.ndarray, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_matrix = score_matrix_from_probs(probs, "inverse_probability")
    uncertainty = observable_uncertainty(probs)
    base_q = conformal_quantile(true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx]), args.alpha)
    failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    errors = probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx]
    inputs = DiscoveryInputs(embeddings[data.discovery_idx], uncertainty[data.discovery_idx], failures, errors)
    outputs = {method: discover_slice_method(method, args, inputs) for method in DISCOVERY_METHODS}
    pool_sources = {method: class_conditional_groups(groups, len(data.class_names)) for method, groups in outputs.items()}
    pool = select_group_pool(
        pool_sources,
        embeddings[data.discovery_idx],
        probs[data.discovery_idx],
        failures,
        args.alpha,
        methods=POOL_METHODS,
        min_size=args.min_size,
        epsilon=args.epsilon,
        enrichment=args.enrichment,
        max_groups=args.pool_groups,
    )
    outputs["candidate_pool"] = pool
    eval_groups = [group for groups in outputs.values() for group in groups]

    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    standard = fit_standard(base_q)
    classwise = classwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], args.alpha)
    known_q = groupwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], data.known_group[data.calib0_idx], args.alpha)
    docp_mvp = fit_pool_mvp(base_q, pool, embeddings[data.mvcal_idx], mv_scores, args.alpha, observed_scores=probs[data.mvcal_idx], min_size=args.rectifier_min_size, max_active_groups=args.pool_groups)
    docp = fit_batch_gcp(base_q, pool, embeddings[data.mvcal_idx], mv_scores, args.alpha, observed_scores=probs[data.mvcal_idx], min_size=args.rectifier_min_size)

    test = data.test_idx
    rows = [
        evaluate_threshold_model("Standard CP", standard, embeddings[test], score_matrix[test], data.labels[test], None, data.known_group[test], eval_groups, args.alpha, observed_scores=uncertainty[test], eval_observed_scores=probs[test]),
        evaluate_classwise_model("Classwise CP", classwise, embeddings[test], score_matrix[test], data.labels[test], None, data.known_group[test], eval_groups, args.alpha, observed_scores=uncertainty[test], eval_observed_scores=probs[test]),
        evaluate_groupwise_model("Known-group CP", known_q, embeddings[test], score_matrix[test], data.labels[test], data.known_group[test], None, data.known_group[test], eval_groups, args.alpha, observed_scores=uncertainty[test], eval_observed_scores=probs[test]),
    ]
    for method, groups in outputs.items():
        if method == "candidate_pool":
            continue
        rectifier = fit_slice_quantile(base_q, groups, embeddings[data.mvcal_idx], mv_scores, args.alpha, observed_scores=uncertainty[data.mvcal_idx], min_size=args.rectifier_min_size)
        rows.append(evaluate_threshold_model(_method_name(method), rectifier, embeddings[test], score_matrix[test], data.labels[test], None, data.known_group[test], eval_groups, args.alpha, observed_scores=uncertainty[test], eval_observed_scores=probs[test], own_groups=groups))
    rows.append(evaluate_threshold_model("DOCP-MVP", docp_mvp, embeddings[test], score_matrix[test], data.labels[test], None, data.known_group[test], eval_groups, args.alpha, observed_scores=probs[test], eval_observed_scores=probs[test], own_groups=pool))
    rows.append(evaluate_threshold_model("DOCP", docp, embeddings[test], score_matrix[test], data.labels[test], None, data.known_group[test], eval_groups, args.alpha, observed_scores=probs[test], eval_observed_scores=probs[test], own_groups=pool))

    group_rows = []
    group_rows.extend(_group_rows("Standard CP", standard.thresholds(embeddings[test]), data, test, score_matrix))
    group_rows.extend(_group_rows("Known-group CP", thresholds_from_groups(data.known_group[test], known_q), data, test, score_matrix))
    group_rows.extend(_group_rows("DOCP", docp.thresholds(embeddings[test], observed_scores=probs[test]), data, test, score_matrix))
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def _group_rows(method: str, thresholds: np.ndarray, data: MimicCareData, test: np.ndarray, score_matrix: np.ndarray) -> list[dict[str, object]]:
    is_covered = covered(score_matrix[test], data.labels[test], thresholds)
    sizes = set_sizes(score_matrix[test], thresholds)
    rows = []
    for dimension in data.groups.columns:
        values = data.groups.iloc[test][dimension].to_numpy()
        for value in np.unique(values):
            mask = values == value
            if mask.sum() < 20:
                continue
            rows.append({"method": method, "group_dimension": dimension, "group": value, "n": int(mask.sum()), "coverage": float(is_covered[mask].mean()), "avg_set_size": float(sizes[mask].mean())})
    return rows


def _group_time_strata(data: MimicCareData) -> pd.DataFrame:
    rows = []
    for dimension in data.groups.columns:
        values = data.groups[dimension].to_numpy()
        for value in np.unique(values):
            mask = values == value
            counts = np.bincount(data.labels[mask], minlength=len(data.class_names))
            row: dict[str, object] = {"group_dimension": dimension, "group": value, "n": int(mask.sum())}
            for class_name, count in zip(data.class_names, counts):
                row[f"rate_{class_name}"] = float(count / mask.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _method_name(method: str) -> str:
    return {
        "random_halfspace": "Random halfspace",
        "coverage_domino": "Coverage-Domino",
        "error_domino": "Error-slice",
        "gmm_unaware": "GMM-unaware",
        "kmeans": "K-means",
    }[method]


def _print_settings(args: argparse.Namespace, data: MimicCareData) -> None:
    counts = np.bincount(data.labels, minlength=len(data.class_names)).tolist()
    splits = [len(data.train_idx), len(data.calib0_idx), len(data.discovery_idx), len(data.mvcal_idx), len(data.test_idx)]
    print(f"settings: task=ED-to-ICU time strata, feature_mode={args.feature_mode}, known_group={args.known_group}, seed={args.seed}, alpha={args.alpha}, classes={counts}, splits={splits}")


if __name__ == "__main__":
    main()

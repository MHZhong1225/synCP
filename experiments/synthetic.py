from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.conformal import classwise_quantiles, conformal_quantile, covered, groupwise_quantiles, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_classwise_model, evaluate_groupwise_model, evaluate_threshold_model
from core.progress import progress
from core.rectification import fit_batch_gcp, fit_pool_mvp, fit_slice_quantile, fit_standard
from data.synthetic import make_splits, make_synthetic_medical_data, predict_and_embed, train_classifier
from reports.diagnostics import slice_repair_diagnostics
from reports.output import print_results, write_results
from slices.discovery import GroupFunction, class_conditional_groups, discover_coverage_domino, discover_error_domino, discover_gmm_unaware, discover_kmeans, discover_random_halfspace, select_group_pool


DISCOVERY_METHODS = ("random_halfspace", "coverage_domino", "error_domino", "gmm_unaware", "kmeans")
POOL_METHODS = ("gmm_unaware", "kmeans", "coverage_domino", "random_halfspace")
RECTIFIERS = ("standard", "pool_mvp", "slice_quantile", "batch_gcp")
POOL_STEP = 0.02
POOL_TOLERANCE = 0.0
POOL_MAX_ITER = 50
COVERAGE_EPSILON = 0.025
COVERAGE_ENRICHMENT = 1.15
DOMINO_RESPONSIBILITY_THRESHOLD = 0.20
DOMINO_RADIUS_QUANTILE = 0.90
SYNTHETIC_DIM = 32
SYNTHETIC_CLASSES = 4
SYNTHETIC_SCORE = "inverse_probability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage-driven slice discovery + conformal recalibration.")
    parser.add_argument("--outdir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda", action="append", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--components", type=int, default=30)
    parser.add_argument("--min-size", type=int, default=35)
    parser.add_argument("--rectifier-min-size", type=int, default=20)
    parser.add_argument("--pool-groups", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=COVERAGE_EPSILON)
    parser.add_argument("--enrichment", type=float, default=COVERAGE_ENRICHMENT)
    parser.add_argument("--pool-methods", nargs="+", default=list(POOL_METHODS))
    parser.add_argument("--diagnostics", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, args.cuda)
    _print_run_settings(args, device)

    data = make_synthetic_medical_data(n=args.n, dim=SYNTHETIC_DIM, num_classes=SYNTHETIC_CLASSES, seed=args.seed)
    splits = make_splits(len(data.y), seed=args.seed)
    model = train_classifier(data.x, data.y, splits.train, SYNTHETIC_CLASSES, device=device, seed=args.seed)
    probs, embeddings = predict_and_embed(model, data.x, device=device)
    score_matrix = score_matrix_from_probs(probs, SYNTHETIC_SCORE)
    observed_uncertainty = observable_uncertainty(probs)

    calib_scores = true_label_scores(score_matrix[splits.calib0], data.y[splits.calib0])
    base_q = conformal_quantile(calib_scores, args.alpha)

    disc_observed_scores = observed_uncertainty[splits.discovery]
    disc_failures = ~covered(score_matrix[splits.discovery], data.y[splits.discovery], base_q)
    disc_classification_errors = probs[splits.discovery].argmax(axis=1) != data.y[splits.discovery]

    discovery_outputs = {}
    for method in progress(DISCOVERY_METHODS, desc="discover slices", leave=False):
        if method == "coverage_domino":
            groups, _ = discover_coverage_domino(
                embeddings[splits.discovery],
                disc_observed_scores,
                disc_failures,
                alpha=args.alpha,
                n_components=args.components,
                min_size=args.min_size,
                epsilon=args.epsilon,
                enrichment=args.enrichment,
                responsibility_threshold=DOMINO_RESPONSIBILITY_THRESHOLD,
                radius_quantile=DOMINO_RADIUS_QUANTILE,
                seed=args.seed,
            )
        elif method == "kmeans":
            groups, _ = discover_kmeans(
                embeddings[splits.discovery],
                disc_observed_scores,
                disc_failures,
                alpha=args.alpha,
                n_clusters=args.components,
                min_size=args.min_size,
                epsilon=args.epsilon,
                enrichment=args.enrichment,
                seed=args.seed,
            )
        elif method == "gmm_unaware":
            groups, _ = discover_gmm_unaware(
                embeddings[splits.discovery],
                disc_observed_scores,
                disc_failures,
                alpha=args.alpha,
                n_components=args.components,
                min_size=args.min_size,
                epsilon=args.epsilon,
                enrichment=args.enrichment,
                seed=args.seed,
            )
        elif method == "error_domino":
            groups, _ = discover_error_domino(
                embeddings[splits.discovery],
                disc_observed_scores,
                disc_classification_errors,
                n_components=args.components,
                min_size=args.min_size,
                epsilon=args.epsilon,
                enrichment=args.enrichment,
                seed=args.seed,
            )
        elif method == "random_halfspace":
            groups, _ = discover_random_halfspace(
                embeddings[splits.discovery],
                disc_observed_scores,
                disc_failures,
                alpha=args.alpha,
                num_directions=args.components,
                min_size=args.min_size,
                epsilon=args.epsilon,
                enrichment=args.enrichment,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unknown discovery method: {method}")
        discovery_outputs[method] = groups
    pool_source_outputs = _pool_source_outputs(discovery_outputs, SYNTHETIC_CLASSES)
    pool_observed_scores = probs[splits.discovery]
    discovery_outputs["candidate_pool"] = select_group_pool(
        pool_source_outputs,
        embeddings[splits.discovery],
        pool_observed_scores,
        disc_failures,
        args.alpha,
        methods=tuple(args.pool_methods),
        min_size=args.min_size,
        epsilon=args.epsilon,
        enrichment=args.enrichment,
        max_groups=args.pool_groups,
    )
    eval_groups = _collect_eval_groups(discovery_outputs)

    mv_scores = true_label_scores(score_matrix[splits.mvcal], data.y[splits.mvcal])
    mv_observed_scores = observed_uncertainty[splits.mvcal]
    mv_pool_scores = probs[splits.mvcal]
    test_idx = splits.test
    test_observed_scores = observed_uncertainty[test_idx]
    test_pool_scores = probs[test_idx]
    eval_observed_scores = test_pool_scores

    rows = []
    diagnostic_frames = []
    standard_model = fit_standard(base_q)
    rows.append(
        evaluate_threshold_model(
            "Standard CP",
            standard_model,
            embeddings[test_idx],
            score_matrix[test_idx],
            data.y[test_idx],
            data.hidden_groups[test_idx],
            data.known_group[test_idx],
            eval_groups,
            args.alpha,
            observed_scores=test_observed_scores,
            eval_observed_scores=eval_observed_scores,
        )
    )
    class_q = classwise_quantiles(score_matrix[splits.calib0], data.y[splits.calib0], args.alpha)
    rows.append(
        evaluate_classwise_model(
            "Classwise CP",
            class_q,
            embeddings[test_idx],
            score_matrix[test_idx],
            data.y[test_idx],
            data.hidden_groups[test_idx],
            data.known_group[test_idx],
            eval_groups,
            args.alpha,
            observed_scores=test_observed_scores,
            eval_observed_scores=eval_observed_scores,
        )
    )
    known_group_q = groupwise_quantiles(score_matrix[splits.calib0], data.y[splits.calib0], data.known_group[splits.calib0], args.alpha)
    rows.append(
        evaluate_groupwise_model(
            "Known-group CP",
            known_group_q,
            embeddings[test_idx],
            score_matrix[test_idx],
            data.y[test_idx],
            data.known_group[test_idx],
            data.hidden_groups[test_idx],
            data.known_group[test_idx],
            eval_groups,
            args.alpha,
            observed_scores=test_observed_scores,
            eval_observed_scores=eval_observed_scores,
        )
    )

    for method, groups in progress(discovery_outputs.items(), desc="evaluate methods", total=len(discovery_outputs), leave=False):
        if "pool_mvp" in RECTIFIERS:
            if method == "candidate_pool":
                rect = fit_pool_mvp(
                    base_q,
                    groups,
                    embeddings[splits.mvcal],
                    mv_scores,
                    args.alpha,
                    observed_scores=mv_pool_scores,
                    step_size=POOL_STEP,
                    tolerance=POOL_TOLERANCE,
                    max_iter=POOL_MAX_ITER,
                    min_size=args.rectifier_min_size,
                    max_active_groups=args.pool_groups,
                )
                rows.append(_eval(_method_display_name(method, "pool_mvp"), rect, embeddings, score_matrix, data, test_idx, eval_groups, args.alpha, observed_scores=test_pool_scores, eval_observed_scores=eval_observed_scores))
        if "slice_quantile" in RECTIFIERS and method != "candidate_pool":
            rect = fit_slice_quantile(base_q, groups, embeddings[splits.mvcal], mv_scores, args.alpha, observed_scores=mv_observed_scores, min_size=args.rectifier_min_size)
            name = _method_display_name(method, "slice_quantile")
            rows.append(_eval(name, rect, embeddings, score_matrix, data, test_idx, eval_groups, args.alpha, observed_scores=test_observed_scores, eval_observed_scores=eval_observed_scores))
            if args.diagnostics:
                diagnostic_frames.append(
                    slice_repair_diagnostics(
                        name,
                        groups,
                        rect,
                        embeddings[splits.mvcal],
                        mv_scores,
                        mv_observed_scores,
                        embeddings[test_idx],
                        score_matrix[test_idx],
                        data.y[test_idx],
                        test_observed_scores,
                        base_q,
                    )
                )
        if "batch_gcp" in RECTIFIERS:
            observed_for_method = mv_pool_scores if method == "candidate_pool" else mv_observed_scores
            test_scores_for_method = test_pool_scores if method == "candidate_pool" else test_observed_scores
            rect = fit_batch_gcp(
                base_q,
                groups,
                embeddings[splits.mvcal],
                mv_scores,
                args.alpha,
                observed_scores=observed_for_method,
                min_size=args.rectifier_min_size,
            )
            rows.append(_eval(_method_display_name(method, "batch_gcp"), rect, embeddings, score_matrix, data, test_idx, eval_groups, args.alpha, observed_scores=test_scores_for_method, eval_observed_scores=eval_observed_scores))
    summary = pd.DataFrame(rows)
    cleaned = write_results(summary, args.outdir / "summary.csv")
    if args.diagnostics:
        _write_diagnostics(diagnostic_frames, args.outdir / "slice_diagnostics.csv")
    print_results(cleaned)
    print(f"\nwrote: {args.outdir / 'summary.csv'}")


def _print_run_settings(args: argparse.Namespace, device: str) -> None:
    print(
        "settings: "
        f"dataset=synthetic, "
        f"n={args.n}, "
        f"seed={args.seed}, "
        f"device={device}, "
        f"alpha={args.alpha}, "
        f"components={args.components}, "
        f"min_size={args.min_size}, "
        f"rectifier_min_size={args.rectifier_min_size}, "
        f"pool_groups={args.pool_groups}, "
        f"epsilon={args.epsilon}, "
        f"enrichment={args.enrichment}",
        file=sys.stderr,
        flush=True,
    )


def _eval(name, rect, embeddings, score_matrix, data, test_idx, groups, alpha, observed_scores=None, eval_observed_scores=None):
    return evaluate_threshold_model(
        name,
        rect,
        embeddings[test_idx],
        score_matrix[test_idx],
        data.y[test_idx],
        data.hidden_groups[test_idx],
        data.known_group[test_idx],
        groups,
        alpha,
        observed_scores=observed_scores,
        eval_observed_scores=eval_observed_scores,
    )


def _write_diagnostics(frames: list[pd.DataFrame], path: Path) -> None:
    non_empty = [frame for frame in frames if not frame.empty]
    if non_empty:
        write_results(pd.concat(non_empty, ignore_index=True), path)


def _pool_source_outputs(discovery_outputs: dict[str, list[GroupFunction]], num_classes: int) -> dict[str, list[GroupFunction]]:
    return {method: class_conditional_groups(groups, num_classes) for method, groups in discovery_outputs.items()}


def _collect_eval_groups(discovery_outputs: dict[str, list[GroupFunction]]) -> list[GroupFunction]:
    groups: list[GroupFunction] = []
    for method_groups in discovery_outputs.values():
        groups.extend(method_groups)
    return groups


def _resolve_device(device: str, cuda_args: list[list[str]] | None) -> str:
    cuda_ids = [item for group in cuda_args or [] for item in group]
    if not cuda_ids:
        return device
    return f"cuda:{cuda_ids[0]}"


def _method_display_name(discovery_method: str, rectifier: str) -> str:
    names = {
        ("candidate_pool", "pool_mvp"): "DOCP-MVP",
        ("candidate_pool", "batch_gcp"): "DOCP",
        ("random_halfspace", "slice_quantile"): "Random halfspace",
        ("random_halfspace", "batch_gcp"): "Random halfspace GCP",
        ("coverage_domino", "batch_gcp"): "Coverage-Domino GCP",
        ("coverage_domino", "slice_quantile"): "Coverage-Domino",
        ("kmeans", "batch_gcp"): "K-means GCP",
        ("kmeans", "slice_quantile"): "K-means",
        ("gmm_unaware", "batch_gcp"): "GMM-unaware GCP",
        ("gmm_unaware", "slice_quantile"): "GMM-unaware",
        ("error_domino", "batch_gcp"): "Error-slice GCP",
        ("error_domino", "slice_quantile"): "Error-slice",
    }
    return names.get((discovery_method, rectifier), f"{discovery_method} {rectifier}")


if __name__ == "__main__":
    main()

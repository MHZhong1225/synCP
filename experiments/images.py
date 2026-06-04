from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.conformal import classwise_quantiles, conformal_quantile, covered, groupwise_quantiles, observable_uncertainty, score_matrix_from_probs, true_label_scores
from core.metrics import evaluate_classwise_model, evaluate_groupwise_model, evaluate_threshold_model
from core.progress import progress
from core.rectification import fit_batch_gcp, fit_pool_mvp, fit_slice_quantile, fit_standard
from data.images import load_imagefolder_experiment, load_nih_experiment, load_rsna_breast_experiment, predict_image_dataset, train_image_classifier
from reports.diagnostics import slice_repair_diagnostics
from reports.output import print_results, write_results
from slices.discovery import GroupFunction, class_conditional_groups, discover_coverage_domino, discover_error_domino, discover_gmm_unaware, discover_kmeans, discover_random_halfspace, select_group_pool


DISCOVERY_METHODS = ("random_halfspace", "coverage_domino", "error_domino", "gmm_unaware", "kmeans")
POOL_METHODS = ("gmm_unaware", "kmeans", "coverage_domino", "random_halfspace")
RECTIFIERS = ("standard", "pool_mvp", "slice_quantile", "batch_gcp")
POOL_STEP = 0.02
POOL_TOLERANCE = 0.0
POOL_MAX_ITER = 50
COVERAGE_EPSILON = 0.02
COVERAGE_ENRICHMENT = 1.10
DOMINO_RESPONSIBILITY_THRESHOLD = 0.20
DOMINO_RADIUS_QUANTILE = 0.90
DEFAULT_HYPERPARAMS = Path("configs/best_docp_current.json")
FIXED_HYPERPARAM_FIELDS = (
    "components",
    "min_size",
    "rectifier_min_size",
    "pool_groups",
    "epsilon",
    "enrichment",
)
DATASET_PRESETS = {
    "bracs": {
        "dataset_root": Path("../dataset/bracs"),
        "cache_file": Path("results/bracs/features.npz"),
    },
    "chest_xray": {
        "dataset_root": Path("../dataset/chest_xray"),
        "split_mode": "random_all",
        "cache_file": Path("results/chest_xray/features.npz"),
    },
    "nih": {
        "dataset_type": "nih",
        "dataset_root": Path("../dataset/NIH_ChestXrays"),
        "nih_target": "Effusion",
        "known_group": "gender",
        "max_samples": 5000,
        "cache_file": Path("results/nih/features.npz"),
    },
    "rsna_5k": {
        "dataset_type": "rsna_5k",
        "dataset_root": Path("../dataset/rsna-breast"),
        "max_samples": 5000,
        "cache_file": Path("results/rsna_5k/features.npz"),
    },
}


@dataclass(frozen=True)
class DiscoveryInputs:
    embeddings: np.ndarray
    uncertainty: np.ndarray
    coverage_failures: np.ndarray
    classification_errors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real image experiment for coverage-driven DOCP.")
    parser.add_argument("--preset", choices=sorted(DATASET_PRESETS), default=None)
    parser.add_argument("--hyperparams", type=Path, default=None)
    parser.add_argument("--hyperparams-key", default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset/bracs"))
    parser.add_argument("--dataset-type", choices=["imagefolder", "nih", "rsna_5k"], default="imagefolder")
    parser.add_argument("--split-mode", choices=["official_test", "random_all"], default="official_test")
    parser.add_argument("--nih-target", default="Effusion")
    parser.add_argument("--known-group", choices=["gender", "age", "view", "gender_age", "gender_view"], default="gender")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("results/bracs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda", action="append", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "densenet121", "chexnet", "radimagenet-resnet50", "radimagenet-densenet121"])
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--rectifier-min-size", type=int, default=10)
    parser.add_argument("--pool-groups", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=COVERAGE_EPSILON)
    parser.add_argument("--enrichment", type=float, default=COVERAGE_ENRICHMENT)
    parser.add_argument("--pool-methods", nargs="+", default=list(POOL_METHODS))
    parser.add_argument("--diagnostics", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = _resolve_experiment_args(args)
    if args.seeds:
        _run_multiseed(args)
        return

    summary = run_experiment(args, show_progress=True)
    cleaned = write_results(summary, args.outdir / "summary.csv")
    print_results(cleaned)
    print(f"\nwrote: {args.outdir / 'summary.csv'}")


def run_experiment(args: argparse.Namespace, show_progress: bool = True) -> pd.DataFrame:
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, args.cuda)
    if show_progress:
        _print_run_settings(args, device)

    data = load_experiment_data(args)
    probs, embeddings = load_or_create_features(args, data, device)
    score = _resolve_score(len(data.class_names))
    score_matrix = score_matrix_from_probs(probs, score)
    observed_uncertainty = observable_uncertainty(probs)

    calib_scores = true_label_scores(score_matrix[data.calib0_idx], data.labels[data.calib0_idx])
    base_q = conformal_quantile(calib_scores, args.alpha)
    disc_observed_scores = observed_uncertainty[data.discovery_idx]
    disc_failures = ~covered(score_matrix[data.discovery_idx], data.labels[data.discovery_idx], base_q)
    disc_classification_errors = probs[data.discovery_idx].argmax(axis=1) != data.labels[data.discovery_idx]

    discovery_inputs = DiscoveryInputs(
        embeddings=embeddings[data.discovery_idx],
        uncertainty=disc_observed_scores,
        coverage_failures=disc_failures,
        classification_errors=disc_classification_errors,
    )
    discovery_outputs = discover_slices(args, discovery_inputs, show_progress)
    pool_source_outputs = _pool_source_outputs(discovery_outputs, len(data.class_names))
    pool_observed_scores = probs[data.discovery_idx]
    discovery_outputs["candidate_pool"] = select_group_pool(
        pool_source_outputs,
        embeddings[data.discovery_idx],
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

    mv_scores = true_label_scores(score_matrix[data.mvcal_idx], data.labels[data.mvcal_idx])
    mv_observed_scores = observed_uncertainty[data.mvcal_idx]
    mv_pool_scores = probs[data.mvcal_idx]
    test_idx = data.test_idx
    test_observed_scores = observed_uncertainty[test_idx]
    test_pool_scores = probs[test_idx]
    eval_observed_scores = test_pool_scores

    rows = []
    diagnostic_frames = []
    standard = fit_standard(base_q)
    rows.append(
        evaluate_threshold_model(
            "Standard CP",
            standard,
            embeddings[test_idx],
            score_matrix[test_idx],
            data.labels[test_idx],
            None,
            data.known_group[test_idx] if data.known_group is not None else None,
            eval_groups,
            args.alpha,
            observed_scores=test_observed_scores,
            eval_observed_scores=eval_observed_scores,
        )
    )
    class_q = classwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], args.alpha)
    rows.append(
        evaluate_classwise_model(
            "Classwise CP",
            class_q,
            embeddings[test_idx],
            score_matrix[test_idx],
            data.labels[test_idx],
            None,
            data.known_group[test_idx] if data.known_group is not None else None,
            eval_groups,
            args.alpha,
            observed_scores=test_observed_scores,
            eval_observed_scores=eval_observed_scores,
        )
    )
    if data.known_group is not None:
        known_group_q = groupwise_quantiles(score_matrix[data.calib0_idx], data.labels[data.calib0_idx], data.known_group[data.calib0_idx], args.alpha)
        rows.append(
            evaluate_groupwise_model(
                "Known-group CP",
                known_group_q,
                embeddings[test_idx],
                score_matrix[test_idx],
                data.labels[test_idx],
                data.known_group[test_idx],
                None,
                data.known_group[test_idx],
                eval_groups,
                args.alpha,
                observed_scores=test_observed_scores,
                eval_observed_scores=eval_observed_scores,
            )
        )

    method_items = discovery_outputs.items()
    if show_progress:
        method_items = progress(method_items, desc="evaluate methods", total=len(discovery_outputs), leave=False)
    for method, groups in method_items:
        if "pool_mvp" in RECTIFIERS:
            if method == "candidate_pool":
                rect = fit_pool_mvp(
                    base_q,
                    groups,
                    embeddings[data.mvcal_idx],
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
            rect = fit_slice_quantile(base_q, groups, embeddings[data.mvcal_idx], mv_scores, args.alpha, observed_scores=mv_observed_scores, min_size=args.rectifier_min_size)
            name = _method_display_name(method, "slice_quantile")
            rows.append(_eval(name, rect, embeddings, score_matrix, data, test_idx, eval_groups, args.alpha, observed_scores=test_observed_scores, eval_observed_scores=eval_observed_scores))
            if args.diagnostics:
                diagnostic_frames.append(
                    slice_repair_diagnostics(
                        name,
                        groups,
                        rect,
                        embeddings[data.mvcal_idx],
                        mv_scores,
                        mv_observed_scores,
                        embeddings[test_idx],
                        score_matrix[test_idx],
                        data.labels[test_idx],
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
                embeddings[data.mvcal_idx],
                mv_scores,
                args.alpha,
                observed_scores=observed_for_method,
                min_size=args.rectifier_min_size,
            )
            rows.append(_eval(_method_display_name(method, "batch_gcp"), rect, embeddings, score_matrix, data, test_idx, eval_groups, args.alpha, observed_scores=test_scores_for_method, eval_observed_scores=eval_observed_scores))
    if not args.diagnostics:
        return pd.DataFrame(rows)

    summary = pd.DataFrame(rows)
    if args.diagnostics:
        _write_diagnostics(diagnostic_frames, args.outdir / "slice_diagnostics.csv")
    return summary


def load_experiment_data(args: argparse.Namespace):
    if args.dataset_type == "nih":
        return load_nih_experiment(
            args.dataset_root,
            target=args.nih_target,
            known_group=args.known_group,
            seed=args.seed,
            max_samples=args.max_samples,
            split_mode=args.split_mode,
        )
    if args.dataset_type == "rsna_5k":
        return load_rsna_breast_experiment(
            args.dataset_root,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    return load_imagefolder_experiment(args.dataset_root, seed=args.seed, split_mode=args.split_mode)


def load_or_create_features(args: argparse.Namespace, data, device: str) -> tuple[np.ndarray, np.ndarray]:
    cache_file = args.cache_file or (args.outdir / "features.npz")
    if args.cache and cache_file.exists():
        return _load_feature_cache(cache_file, len(data.class_names), args.backbone)

    model = train_image_classifier(
        data.dataset,
        data.train_idx,
        num_classes=len(data.class_names),
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        pretrained=True,
        backbone=args.backbone,
        checkpoint=args.backbone_checkpoint,
    )
    probs, embeddings = predict_image_dataset(model, data.dataset, device=device, batch_size=args.predict_batch_size)
    _save_feature_cache(cache_file, probs, embeddings, data.class_names, args.backbone)
    return probs, embeddings


def discover_slices(args: argparse.Namespace, inputs: DiscoveryInputs, show_progress: bool) -> dict[str, list[GroupFunction]]:
    methods = progress(DISCOVERY_METHODS, desc="discover slices", leave=False) if show_progress else DISCOVERY_METHODS
    return {method: discover_slice_method(method, args, inputs) for method in methods}


def discover_slice_method(method: str, args: argparse.Namespace, inputs: DiscoveryInputs) -> list[GroupFunction]:
    common = dict(
        min_size=args.min_size,
        epsilon=args.epsilon,
        enrichment=args.enrichment,
        seed=args.seed,
    )
    if method == "coverage_domino":
        groups, _ = discover_coverage_domino(
            inputs.embeddings,
            inputs.uncertainty,
            inputs.coverage_failures,
            alpha=args.alpha,
            n_components=args.components,
            responsibility_threshold=DOMINO_RESPONSIBILITY_THRESHOLD,
            radius_quantile=DOMINO_RADIUS_QUANTILE,
            **common,
        )
        return groups
    if method == "kmeans":
        groups, _ = discover_kmeans(
            inputs.embeddings,
            inputs.uncertainty,
            inputs.coverage_failures,
            alpha=args.alpha,
            n_clusters=args.components,
            **common,
        )
        return groups
    if method == "gmm_unaware":
        groups, _ = discover_gmm_unaware(
            inputs.embeddings,
            inputs.uncertainty,
            inputs.coverage_failures,
            alpha=args.alpha,
            n_components=args.components,
            **common,
        )
        return groups
    if method == "error_domino":
        groups, _ = discover_error_domino(
            inputs.embeddings,
            inputs.uncertainty,
            inputs.classification_errors,
            n_components=args.components,
            **common,
        )
        return groups
    if method == "random_halfspace":
        groups, _ = discover_random_halfspace(
            inputs.embeddings,
            inputs.uncertainty,
            inputs.coverage_failures,
            alpha=args.alpha,
            num_directions=args.components,
            **common,
        )
        return groups
    raise ValueError(f"Unknown discovery method: {method}")


def _run_multiseed(args: argparse.Namespace) -> None:
    frames = []
    devices = _resolve_devices(args.device, args.cuda)
    for index, seed in enumerate(progress(args.seeds, desc="multi-seed")):
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        device = devices[index % len(devices)]
        seed_args.device = device
        seed_args.cuda = None
        print(_seed_settings(seed_args, device), file=sys.stderr, flush=True)
        frames.append(run_experiment(seed_args, show_progress=False))

    summary = average_runs(frames)
    cleaned = write_results(summary, args.outdir / "summary.csv")
    print_results(cleaned)
    print(f"\nwrote: {args.outdir / 'summary.csv'}")


def _resolve_devices(device: str, cuda_args: list[list[str]] | None) -> list[str]:
    cuda_ids = [item for group in cuda_args or [] for item in group]
    if cuda_ids:
        return [f"cuda:{cuda_id}" for cuda_id in cuda_ids]
    return [device]


def _seed_settings(args: argparse.Namespace, device: str) -> str:
    return (
        "running: "
        f"dataset_type={args.dataset_type}, "
        f"dataset_root={args.dataset_root}, "
        f"seed={args.seed}, "
        f"device={device}, "
        f"components={args.components}, "
        f"min_size={args.min_size}, "
        f"rectifier_min_size={args.rectifier_min_size}, "
        f"pool_groups={args.pool_groups}"
    )


def average_runs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No seed results to average.")
    metric_cols = [col for col in frames[0].columns if col != "method"]
    all_results = pd.concat(frames, ignore_index=True)
    grouped = all_results.groupby("method", sort=False)[metric_cols]
    means = grouped.mean(numeric_only=True)
    stds = grouped.std(ddof=1, numeric_only=True)

    summary = pd.DataFrame(index=means.index)
    for col in metric_cols:
        if col not in means:
            continue
        summary[col] = means[col]
        summary[f"{col}_std"] = stds[col].where(means[col].isna(), stds[col].fillna(0.0))
    return summary.reset_index()


def _print_run_settings(args: argparse.Namespace, device: str) -> None:
    print(
        "settings: "
        f"dataset_type={args.dataset_type}, "
        f"dataset_root={args.dataset_root}, "
        f"split={args.split_mode}, "
        f"backbone={args.backbone}, "
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


def _resolve_experiment_args(args: argparse.Namespace) -> argparse.Namespace:
    _apply_dataset_preset(args)
    _apply_fixed_hyperparams(args)
    return args


def _apply_dataset_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    preset = DATASET_PRESETS[args.preset]
    for key, value in preset.items():
        if key == "cache_file" and args.cache_file is not None:
            continue
        setattr(args, key, value)
    args.cache = True
    if args.hyperparams is None:
        args.hyperparams = DEFAULT_HYPERPARAMS
    if args.hyperparams_key is None:
        args.hyperparams_key = args.preset


def _apply_fixed_hyperparams(args: argparse.Namespace) -> None:
    if args.hyperparams is None:
        return
    if args.hyperparams_key is None:
        raise ValueError("--hyperparams-key is required when --hyperparams is used without --preset.")
    config = _load_fixed_hyperparams(args.hyperparams, args.hyperparams_key)
    for field in FIXED_HYPERPARAM_FIELDS:
        setattr(args, field, config[field])


def _load_fixed_hyperparams(path: Path, key: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key not in payload:
        raise KeyError(f"No fixed hyperparameters for {key!r} in {path}.")
    config = payload[key]
    missing_fields = [field for field in FIXED_HYPERPARAM_FIELDS if field not in config]
    if missing_fields:
        raise KeyError(f"Fixed hyperparameters for {key!r} are missing: {', '.join(missing_fields)}.")
    return config


def _eval(name, rect, embeddings, score_matrix, data, test_idx, groups, alpha, observed_scores=None, eval_observed_scores=None):
    return evaluate_threshold_model(
        name,
        rect,
        embeddings[test_idx],
        score_matrix[test_idx],
        data.labels[test_idx],
        None,
        data.known_group[test_idx] if data.known_group is not None else None,
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


def _resolve_score(num_classes: int) -> str:
    return "aps" if num_classes > 2 else "inverse_probability"


def _load_feature_cache(path: Path, num_classes: int, backbone: str) -> tuple[np.ndarray, np.ndarray]:
    cached = np.load(path)
    probs = cached["probs"]
    embeddings = cached["embeddings"]
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(
            f"Cached probabilities have {probs.shape[1] if probs.ndim == 2 else 'invalid'} classes, "
            f"but this dataset has {num_classes}. Use a task-specific cache or rerun without --reuse-cache."
        )
    if "num_classes" in cached and int(cached["num_classes"]) != num_classes:
        raise ValueError("Cached feature file was created for a different number of classes.")
    if "backbone" in cached:
        if str(cached["backbone"]) != backbone:
            raise ValueError("Cached feature file was created for a different backbone.")
    elif backbone != "resnet18":
        raise ValueError("Cached feature file has no backbone metadata. Rerun without --reuse-cache for this backbone.")
    return probs, embeddings


def _save_feature_cache(path: Path, probs: np.ndarray, embeddings: np.ndarray, class_names: list[str], backbone: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probs=probs,
        embeddings=embeddings,
        num_classes=np.asarray(len(class_names), dtype=np.int64),
        class_names=np.asarray(class_names),
        backbone=np.asarray(backbone),
    )


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

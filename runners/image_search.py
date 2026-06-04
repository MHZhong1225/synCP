from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from core.progress import progress
from reports.output import print_results, write_results


DATASET_ROOTS = {
    "bach": Path("../dataset/bach"),
    "bracs": Path("../dataset/bracs"),
    "chest_xray": Path("../dataset/chest_xray"),
}

ARG_KEYS = {
    "dataset.batch_size": "--batch-size",
    "dataset.predict_batch_size": "--predict-batch-size",
    "dataset.image_backbone": "--backbone",
    "docp.components": "--components",
    "docp.min_size": "--min-size",
    "conformal.alpha": "--alpha",
    "docp.rectifier_min_size": "--rectifier-min-size",
    "docp.pool_groups": "--pool-groups",
    "backbone_train.epochs": "--epochs",
    "backbone_train.lr": "--lr",
}

CONFIG_COLUMNS = {
    "dataset.dataset_mode": "dataset",
    "dataset.image_backbone": "backbone",
    "backbone_train.epochs": "epochs",
    "backbone_train.lr": "lr",
    "docp.components": "components",
    "docp.pool_groups": "pool_groups",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact image backbone hyperparameter search.")
    parser.add_argument("--config", type=Path, default=Path("configs/image_backbone_search.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results/image_backbone_search"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda", action="append", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    args.outdir.mkdir(parents=True, exist_ok=True)
    run_root = args.outdir / "_runs"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir()

    try:
        rows = []
        combinations = list(_parameter_combinations(config))
        seeds = [int(seed) for seed in config["seeds"]]
        devices = _resolve_devices(args.device, args.cuda)
        search_items = list(enumerate(combinations, start=1))
        for config_index, params in progress(search_items, desc="image search", total=len(search_items)):
            rows.append(_run_combination(config_index, len(combinations), params, seeds, devices, run_root))
        summary = pd.concat(rows, ignore_index=True)
        cleaned = write_results(summary, args.outdir / "summary.csv")
        print_results(cleaned)
        print(f"\nwrote: {args.outdir / 'summary.csv'}")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def _load_config(path: Path) -> dict[str, list]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if "seeds" not in config:
        raise ValueError("Search config requires a seeds field.")
    allowed = set(ARG_KEYS) | {"dataset.dataset_mode", "seeds"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown search config keys: {unknown}")
    for key, values in config.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a non-empty list.")
    return config


def _parameter_combinations(config: dict[str, list]) -> list[dict[str, object]]:
    keys = [key for key in config if key != "seeds"]
    values = [config[key] for key in keys]
    return [dict(zip(keys, item)) for item in itertools.product(*values)]


def _resolve_devices(device: str, cuda_args: list[list[str]] | None) -> list[str]:
    cuda_ids = [item for group in cuda_args or [] for item in group]
    if cuda_ids:
        return [f"cuda:{cuda_id}" for cuda_id in cuda_ids]
    return [device]


def _run_combination(
    config_index: int,
    total_configs: int,
    params: dict[str, object],
    seeds: list[int],
    devices: list[str],
    run_root: Path,
) -> pd.DataFrame:
    seed_frames = []
    for seed_index, seed in progress(list(enumerate(seeds)), desc=f"config {config_index}/{total_configs}", leave=False):
        device = devices[seed_index % len(devices)]
        run_dir = run_root / f"config_{config_index}" / f"seed_{seed}"
        _print_search_settings(config_index, total_configs, params, seed, device)
        _run_one(params, seed, device, run_dir)
        seed_frames.append(pd.read_csv(run_dir / "summary.csv"))

    frame = pd.concat(seed_frames, ignore_index=True)
    metric_cols = [col for col in frame.columns if col != "method"]
    averaged = frame.groupby("method", as_index=False)[metric_cols].mean(numeric_only=True)
    averaged.insert(1, "num_runs", len(seeds))
    for key, column in CONFIG_COLUMNS.items():
        if key in params:
            averaged.insert(1, column, params[key])
    return averaged


def _print_search_settings(config_index: int, total_configs: int, params: dict[str, object], seed: int, device: str) -> None:
    shown = {
        "dataset": params.get("dataset.dataset_mode", "bach"),
        "backbone": params.get("dataset.image_backbone", "resnet18"),
        "seed": seed,
        "device": device,
        "batch_size": params.get("dataset.batch_size"),
        "predict_batch_size": params.get("dataset.predict_batch_size"),
        "components": params.get("docp.components"),
        "min_size": params.get("docp.min_size"),
        "rectifier_min_size": params.get("docp.rectifier_min_size"),
        "pool_groups": params.get("docp.pool_groups"),
        "epochs": params.get("backbone_train.epochs"),
        "lr": params.get("backbone_train.lr"),
    }
    compact = ", ".join(f"{key}={value}" for key, value in shown.items() if value is not None)
    print(f"running config {config_index}/{total_configs}: {compact}", file=sys.stderr, flush=True)


def _run_one(params: dict[str, object], seed: int, device: str, outdir: Path) -> None:
    dataset_mode = str(params.get("dataset.dataset_mode", "bach"))
    if dataset_mode not in DATASET_ROOTS:
        raise ValueError(f"Unknown dataset.dataset_mode: {dataset_mode}")

    command = [
        sys.executable,
        "-m",
        "experiments.images",
        "--dataset-root",
        str(DATASET_ROOTS[dataset_mode]),
        "--seed",
        str(seed),
        "--device",
        device,
        "--outdir",
        str(outdir),
    ]
    for key, flag in ARG_KEYS.items():
        if key in params:
            command.extend([flag, str(params[key])])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()

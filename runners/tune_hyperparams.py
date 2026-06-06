from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.progress import progress
from reports.output import print_results, write_results


@dataclass(frozen=True)
class TuningConfig:
    name: str
    components: int
    min_size: int
    rectifier_min_size: int
    pool_groups: int
    epsilon: float
    enrichment: float


DEFAULT_CONFIGS = [
    TuningConfig("efficient", 8, 30, 15, 3, 0.02, 1.10),
    TuningConfig("balanced", 12, 20, 10, 4, 0.01, 1.05),
    TuningConfig("conservative", 8, 40, 25, 4, 0.03, 1.15),
    TuningConfig("larger_pool", 16, 20, 10, 8, 0.01, 1.05),
    TuningConfig("small_slice", 16, 10, 6, 8, 0.00, 1.00),
    TuningConfig("strong", 12, 15, 8, 6, 0.00, 1.00),
]

CONSERVATIVE_CONFIGS = [
    TuningConfig("conservative_c6", 6, 40, 25, 4, 0.03, 1.15),
    TuningConfig("conservative_c8", 8, 40, 25, 4, 0.03, 1.15),
    TuningConfig("conservative_c10", 10, 40, 25, 4, 0.03, 1.15),
    TuningConfig("conservative_s30", 8, 30, 25, 4, 0.03, 1.15),
    TuningConfig("conservative_s50", 8, 50, 25, 4, 0.03, 1.15),
    TuningConfig("conservative_r20", 8, 40, 20, 4, 0.03, 1.15),
    TuningConfig("conservative_r30", 8, 40, 30, 4, 0.03, 1.15),
    TuningConfig("conservative_p2", 8, 40, 25, 2, 0.03, 1.15),
    TuningConfig("conservative_p6", 8, 40, 25, 6, 0.03, 1.15),
]

CONFIG_PROFILES = {
    "default": DEFAULT_CONFIGS,
    "conservative": CONSERVATIVE_CONFIGS,
}

DATASETS = {
    "bracs": {
        "module": "experiments.images",
        "args": ["--dataset-root", "../dataset/bracs", "--cache-file", "results/bracs/features.npz", "--cache"],
    },
    "nih": {
        "module": "experiments.images",
        "args": [
            "--dataset-type",
            "nih",
            "--dataset-root",
            "../dataset/NIH_ChestXrays",
            "--nih-target",
            "Effusion",
            "--known-group",
            "gender",
            "--max-samples",
            "5000",
            "--cache-file",
            "results/nih/features.npz",
            "--cache",
        ],
    },
    "rsna_5k": {
        "module": "experiments.images",
        "args": [
            "--dataset-type",
            "rsna_5k",
            "--dataset-root",
            "../dataset/rsna-breast",
            "--max-samples",
            "5000",
            "--cache-file",
            "results/rsna_5k/features.npz",
            "--cache",
        ],
    },
    "synthetic": {"module": "experiments.synthetic", "args": []},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune compact DOCP hyperparameters on all datasets.")
    parser.add_argument("--outdir", type=Path, default=Path("results/current_tuning"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--profile", choices=sorted(CONFIG_PROFILES), default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    run_root = args.outdir / "_runs"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)

    rows = []
    failures = []
    configs = [(dataset_name, config) for dataset_name in args.datasets for config in CONFIG_PROFILES[args.profile]]
    for dataset_name, config in progress(configs, desc="tune DOCP", total=len(configs)):
        dataset = DATASETS[dataset_name]
        config_name = config.name
        _print_tuning_settings(dataset_name, config, args.device, args.seed)
        run_dir = run_root / dataset_name / config_name
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            dataset["module"],
            "--outdir",
            str(run_dir),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--components",
            str(config.components),
            "--min-size",
            str(config.min_size),
            "--rectifier-min-size",
            str(config.rectifier_min_size),
            "--pool-groups",
            str(config.pool_groups),
            "--epsilon",
            str(config.epsilon),
            "--enrichment",
            str(config.enrichment),
            *dataset["args"],
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            failures.append({"dataset": dataset_name, "config": config_name, "stderr": result.stderr[-2000:]})
            continue
        frame = pd.read_csv(run_dir / "summary.csv")
        frame.insert(0, "dataset", dataset_name)
        frame.insert(1, "config", config_name)
        frame["components"] = config.components
        frame["min_size"] = config.min_size
        frame["rectifier_min_size"] = config.rectifier_min_size
        frame["pool_groups"] = config.pool_groups
        frame["epsilon"] = config.epsilon
        frame["enrichment"] = config.enrichment
        rows.append(frame)

    if not rows:
        raise RuntimeError("No tuning run completed successfully.")

    sweep = pd.concat(rows, ignore_index=True)
    write_results(sweep, args.outdir / "sweep_summary.csv")
    best = select_best_docp(sweep)
    cleaned_best = write_results(best, args.outdir / "best_docp_by_dataset.csv")
    write_fixed_config(best, args.outdir / "fixed_hyperparams.json")
    print_results(cleaned_best)
    print(f"\nwrote: {args.outdir / 'sweep_summary.csv'}")
    print(f"wrote: {args.outdir / 'best_docp_by_dataset.csv'}")
    print(f"wrote: {args.outdir / 'fixed_hyperparams.json'}")

    if failures:
        pd.DataFrame(failures).to_csv(args.outdir / "failures.csv", index=False)
        print(f"wrote: {args.outdir / 'failures.csv'}")
    if not args.keep_runs:
        shutil.rmtree(run_root, ignore_errors=True)


def select_best_docp(sweep: pd.DataFrame) -> pd.DataFrame:
    docp = sweep[sweep["method"].eq("DOCP")].copy()
    slice_col = "worst_shared_discovered_slice_coverage" if "worst_shared_discovered_slice_coverage" in docp else "worst_discovered_slice_coverage"
    docp["_slice_coverage"] = docp[slice_col].fillna(docp["marginal_coverage"])
    docp["_known_coverage"] = docp["worst_known_group_coverage"].fillna(-1.0)
    docp["_class_coverage"] = docp["worst_class_coverage"].fillna(-1.0)
    docp["_coverage_ok"] = docp["marginal_coverage"] >= 0.9
    best_rows = []
    for _, group in docp.groupby("dataset"):
        if group["_coverage_ok"].any():
            ranked = group[group["_coverage_ok"]].sort_values(
                ["max_undercoverage_gap", "_slice_coverage", "avg_set_size", "marginal_coverage", "_class_coverage", "_known_coverage"],
                ascending=[True, False, True, False, False, False],
            )
        else:
            ranked = group.sort_values(
                ["max_undercoverage_gap", "_slice_coverage", "marginal_coverage", "_class_coverage", "_known_coverage", "avg_set_size"],
                ascending=[True, False, False, False, False, True],
            )
        best_rows.append(ranked.iloc[0])
    docp = pd.DataFrame(best_rows)
    keep = [
        "dataset",
        "config",
        "components",
        "min_size",
        "rectifier_min_size",
        "pool_groups",
        "epsilon",
        "enrichment",
        "marginal_coverage",
        "avg_set_size",
        "max_undercoverage_gap",
        "worst_shared_discovered_slice_coverage",
        "worst_own_pool_coverage",
        "worst_discovered_slice_coverage",
        "worst_class_coverage",
        "worst_known_group_coverage",
    ]
    return docp.groupby("dataset", as_index=False).head(1)[keep].reset_index(drop=True)


def _print_tuning_settings(dataset_name: str, config: TuningConfig, device: str, seed: int) -> None:
    print(
        "running: "
        f"dataset={dataset_name}, "
        f"config={config.name}, "
        f"seed={seed}, "
        f"device={device}, "
        f"components={config.components}, "
        f"min_size={config.min_size}, "
        f"rectifier_min_size={config.rectifier_min_size}, "
        f"pool_groups={config.pool_groups}, "
        f"epsilon={config.epsilon}, "
        f"enrichment={config.enrichment}",
        file=sys.stderr,
        flush=True,
    )


def write_fixed_config(best: pd.DataFrame, path: Path) -> None:
    payload = {}
    for row in best.to_dict(orient="records"):
        payload[row["dataset"]] = {
            "config": row["config"],
            "components": int(row["components"]),
            "min_size": int(row["min_size"]),
            "rectifier_min_size": int(row["rectifier_min_size"]),
            "pool_groups": int(row["pool_groups"]),
            "epsilon": float(row["epsilon"]),
            "enrichment": float(row["enrichment"]),
            "selection_metrics": {
                "marginal_coverage": float(row["marginal_coverage"]),
                "avg_set_size": float(row["avg_set_size"]),
                "max_undercoverage_gap": float(row["max_undercoverage_gap"]),
                "worst_shared_discovered_slice_coverage": _optional_float(row.get("worst_shared_discovered_slice_coverage")),
                "worst_own_pool_coverage": _optional_float(row.get("worst_own_pool_coverage")),
                "worst_discovered_slice_coverage": _optional_float(row["worst_discovered_slice_coverage"]),
                "worst_class_coverage": _optional_float(row["worst_class_coverage"]),
                "worst_known_group_coverage": _optional_float(row["worst_known_group_coverage"]),
            },
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


if __name__ == "__main__":
    main()

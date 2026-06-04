from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from core.progress import progress


NIH_TARGETS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NIH ChestXray14 one-vs-rest tasks.")
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset/NIH_ChestXrays"))
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--targets", nargs="+", default=NIH_TARGETS)
    parser.add_argument("--known-group", default="gender", choices=["gender", "age", "view", "gender_age", "gender_view"])
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda", action="append", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--cache", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_targets(args.targets)
    for target in progress(args.targets, desc="NIH one-vs-rest"):
        command = build_command(args, target)
        print("running: " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def validate_targets(targets: list[str]) -> None:
    unknown = sorted(set(targets) - set(NIH_TARGETS))
    if unknown:
        raise ValueError(f"Unknown NIH target(s): {', '.join(unknown)}")


def build_command(args: argparse.Namespace, target: str) -> list[str]:
    suffix = target_suffix(target)
    outdir = args.out_root / f"nih_{suffix}"
    command = [
        sys.executable,
        "-m",
        "experiments.images",
        "--dataset-type",
        "nih",
        "--dataset-root",
        str(args.dataset_root),
        "--nih-target",
        target,
        "--known-group",
        args.known_group,
        "--max-samples",
        str(args.max_samples),
        "--outdir",
        str(outdir),
        "--cache-file",
        str(outdir / "features.npz"),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.cache:
        command.append("--cache")
    for cuda_group in args.cuda or []:
        command.extend(["--cuda", *cuda_group])
    if args.seeds:
        command.extend(["--seeds", *map(str, args.seeds)])
    return command


def target_suffix(target: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")


if __name__ == "__main__":
    main()

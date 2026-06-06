from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


TARGETS = {
    "Pleural Effusion": "effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Pulmonary Edema": "edema",
    "Cardiomegaly": "cardiomegaly",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all MIMIC-CXR report-label tasks with MIMIC-IV/ED subgroups.")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--known-group", default="race", choices=["gender", "age", "race", "arrival", "race_age", "gender_age", "ed_acuity", "icu_history"])
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for target in args.targets:
        suffix = TARGETS[target]
        outdir = args.out_root / f"mimic_{suffix}"
        command = [
            sys.executable,
            "-m",
            "experiments.images",
            "--dataset-type",
            "mimic",
            "--dataset-root",
            "../dataset/MIMIC-CXR",
            "--mimic-target",
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
            "--cache",
        ]
        print(f"running MIMIC target: {target}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

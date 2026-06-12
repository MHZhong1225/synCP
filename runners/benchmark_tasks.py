from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.mimic_care import load_mimic_care_experiment
from experiments import images as image_experiment


IMAGE_TASKS = ("ham10000", "isic", "rsna_5k")
MIMIC_CXR_TARGETS = {
    "effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "edema": "Pulmonary Edema",
    "cardiomegaly": "Cardiomegaly",
}
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
CORE_RESULT_COLUMNS = (
    "method",
    "marginal_coverage",
    "marginal_undercoverage",
    "avg_set_size",
    "worst_class_coverage",
    "worst_known_group_coverage",
    "known_group_macro_coverage",
    "known_group_mean_undercoverage_gap",
    "known_group_mean_absolute_coverage_gap",
    "known_group_coverage_disparity",
    "fixed_subgroup",
    "fixed_subgroup_size",
    "fixed_subgroup_coverage",
    "fixed_subgroup_avg_set_size",
    "fixed_subgroup_undercoverage_gap",
    "known_group_attribute_macro_coverage",
    "known_group_attribute_mean_undercoverage_gap",
    "known_group_attribute_mean_absolute_coverage_gap",
    "worst_shared_discovered_slice_coverage",
    "worst_own_pool_coverage",
    "max_common_undercoverage_gap",
    "max_undercoverage_gap",
    "worst_hidden_group_coverage",
    "task",
    "family",
    "num_selected_slices",
)


def target_suffix(target: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")


def nih_target_from_suffix(suffix: str) -> str:
    for target in NIH_TARGETS:
        if target_suffix(target) == suffix:
            return target
    raise ValueError(f"Unknown NIH target suffix: {suffix}")


def make_image_args(dataset: str, split_seed: int) -> argparse.Namespace:
    if dataset not in image_experiment.DATASET_PRESETS:
        raise KeyError(f"Unknown dataset preset: {dataset}")
    preset = image_experiment.DATASET_PRESETS[dataset]
    hyperparams = image_experiment._load_fixed_hyperparams(image_experiment.DEFAULT_HYPERPARAMS, dataset)
    return argparse.Namespace(
        preset=dataset,
        dataset_root=preset["dataset_root"],
        dataset_type=preset.get("dataset_type", "imagefolder"),
        split_mode="official_test",
        nih_target=preset.get("nih_target", "Effusion"),
        mimic_target=preset.get("mimic_target", "Pleural Effusion"),
        mimic_iv_root=preset.get("mimic_iv_root", Path("../dataset/mimic-iv-3.1")),
        mimic_ed_root=preset.get("mimic_ed_root", Path("../dataset/mimic-iv-ed-2.2")),
        known_group=preset.get("known_group", "gender"),
        max_samples=preset.get("max_samples"),
        cache_file=preset["cache_file"],
        backbone="resnet18",
        seed=split_seed,
        alpha=0.10,
        components=int(hyperparams["components"]),
        min_size=int(hyperparams["min_size"]),
        rectifier_min_size=int(hyperparams["rectifier_min_size"]),
        pool_groups=int(hyperparams["pool_groups"]),
        epsilon=float(hyperparams["epsilon"]),
        enrichment=float(hyperparams["enrichment"]),
    )


def expand_tasks(tasks: list[str], nih_targets: list[str]) -> list[tuple[str, str]]:
    expanded = []
    for task in tasks:
        if task == "nih_all":
            expanded.extend((f"nih_{target_suffix(target)}", "nih") for target in nih_targets)
        elif task == "mimic_cxr_all":
            expanded.extend((f"mimic_cxr_{suffix}", "mimic_cxr") for suffix in MIMIC_CXR_TARGETS)
        elif task == "mimic_care":
            expanded.append((task, "mimic_care"))
        elif task in IMAGE_TASKS:
            expanded.append((task, task))
        else:
            raise ValueError(f"Unknown task: {task}")
    return expanded


def load_task(task_name: str, family: str, split_seed: int, cache_root: Path = Path("results/npz")):
    if family == "mimic_care":
        args = argparse.Namespace(
            alpha=0.10,
            components=24,
            min_size=40,
            rectifier_min_size=20,
            pool_groups=8,
            epsilon=0.02,
            enrichment=1.10,
            seed=split_seed,
        )
        data = load_mimic_care_experiment(
            Path("../dataset/mimic-iv-3.1"),
            Path("../dataset/mimic-iv-ed-2.2"),
            known_group="race_acuity",
            feature_mode="text_triage",
            seed=split_seed,
        )
        cache = np.load(cache_root / "mimic_care.npz")
        normalized = SimpleNamespace(
            labels=data.labels,
            class_names=data.class_names,
            calib0_idx=data.calib0_idx,
            discovery_idx=data.discovery_idx,
            mvcal_idx=data.mvcal_idx,
            test_idx=data.test_idx,
            known_group=data.known_group,
            sensitive_groups=data.sensitive_groups,
        )
        return args, normalized, cache["probs"], cache["embeddings"], "inverse_probability"

    preset = "nih" if family == "nih" else ("mimic_effusion" if family == "mimic_cxr" else task_name)
    task_args = make_image_args(preset, split_seed)
    if family == "nih":
        target = nih_target_from_suffix(task_name.removeprefix("nih_"))
        task_args.nih_target = target
        task_args.cache_file = cache_root / f"{task_name}_strict.npz"
    elif family == "mimic_cxr":
        suffix = task_name.removeprefix("mimic_cxr_")
        task_args.mimic_target = MIMIC_CXR_TARGETS[suffix]
        task_args.cache_file = cache_root / f"mimic_{suffix}.npz"
    elif family in IMAGE_TASKS:
        task_args.cache_file = cache_root / f"{task_name}.npz"
    data = image_experiment.load_experiment_data(task_args)
    probs, embeddings = image_experiment._load_feature_cache(task_args.cache_file, len(data.class_names), task_args.backbone)
    return task_args, data, probs, embeddings, image_experiment._resolve_score(len(data.class_names), task_name)


def format_result_columns(frame: pd.DataFrame, task_name: str, family: str) -> pd.DataFrame:
    attribute = _known_group_attribute(task_name, family)
    generic_prefix = "known_group_coverage_group_"
    if attribute is not None:
        frame = frame.rename(
            columns={
                column: column.replace(generic_prefix, f"known_group_coverage_{attribute}_group_", 1)
                for column in frame.columns
                if column.startswith(generic_prefix)
            }
        )
    return order_result_columns(frame)


def order_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    core_columns = [column for column in CORE_RESULT_COLUMNS if column in frame]
    known_group_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("known_group_") and column not in core_columns
    )
    columns = [column for column in CORE_RESULT_COLUMNS[:5] if column in frame]
    columns.extend(known_group_columns)
    columns.extend(column for column in CORE_RESULT_COLUMNS[5:] if column in frame)
    return frame.loc[:, columns]


def _known_group_attribute(task_name: str, family: str) -> str | None:
    if family == "nih":
        return "gender"
    if family == "mimic_cxr":
        return "race"
    return {
        "ham10000": "sex",
        "isic": "anatom_site_general",
        "mimic_care": "race_acuity",
        "rsna_5k": "site",
    }.get(task_name)

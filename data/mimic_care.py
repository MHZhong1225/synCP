from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CARE_CLASSES = ["early_icu_le_6h", "intermediate_icu_6_to_24h", "late_icu_gt_24h"]
GROUP_COLUMNS = ["race", "gender", "age_group", "arrival_transport", "acuity_group", "race_acuity"]


@dataclass(frozen=True)
class MimicCareData:
    documents: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    train_idx: np.ndarray
    calib0_idx: np.ndarray
    discovery_idx: np.ndarray
    mvcal_idx: np.ndarray
    test_idx: np.ndarray
    known_group: np.ndarray
    groups: pd.DataFrame


def load_mimic_care_experiment(
    mimic_iv_root: Path,
    mimic_ed_root: Path,
    known_group: str = "race_acuity",
    feature_mode: str = "text_triage",
    seed: int = 7,
    max_hours: float = 168.0,
) -> MimicCareData:
    cohort = _load_care_cohort(mimic_iv_root, mimic_ed_root, max_hours)
    cohort["document"] = _build_documents(cohort, feature_mode)
    cohort = cohort.reset_index(drop=True)

    labels = cohort["label"].to_numpy(dtype=np.int64)
    splits = _patient_level_splits(cohort, seed)
    group_frame = cohort[GROUP_COLUMNS].copy()
    if known_group not in group_frame:
        raise ValueError(f"Unknown MIMIC care group: {known_group}")
    _, encoded_group = np.unique(group_frame[known_group].to_numpy(), return_inverse=True)
    return MimicCareData(
        documents=cohort["document"].to_numpy(dtype=str),
        labels=labels,
        class_names=CARE_CLASSES,
        train_idx=splits[0],
        calib0_idx=splits[1],
        discovery_idx=splits[2],
        mvcal_idx=splits[3],
        test_idx=splits[4],
        known_group=encoded_group.astype(np.int64),
        groups=group_frame,
    )


def _load_care_cohort(mimic_iv_root: Path, mimic_ed_root: Path, max_hours: float) -> pd.DataFrame:
    ed = pd.read_csv(
        mimic_ed_root / "ed" / "edstays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "gender", "race", "arrival_transport"],
        parse_dates=["intime"],
    )
    triage = pd.read_csv(
        mimic_ed_root / "ed" / "triage.csv.gz",
        usecols=["stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"],
    )
    patients = pd.read_csv(mimic_iv_root / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    first_icu = (
        pd.read_csv(mimic_iv_root / "icu" / "icustays.csv.gz", usecols=["hadm_id", "intime"], parse_dates=["intime"])
        .groupby("hadm_id", as_index=False)["intime"]
        .min()
        .rename(columns={"intime": "icu_intime"})
    )

    cohort = ed.merge(first_icu, on="hadm_id", how="inner").merge(triage, on="stay_id", how="left")
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort["hours_to_icu"] = (cohort["icu_intime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort = cohort[
        cohort["chiefcomplaint"].notna()
        & cohort["hours_to_icu"].between(0.0, max_hours, inclusive="both")
    ].copy()
    cohort = cohort.sort_values(["hadm_id", "hours_to_icu"]).drop_duplicates("hadm_id", keep="first")
    cohort["label"] = pd.cut(
        cohort["hours_to_icu"],
        bins=[-np.inf, 6.0, 24.0, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype(np.int64)
    _add_group_columns(cohort)
    return cohort


def _add_group_columns(cohort: pd.DataFrame) -> None:
    cohort["race"] = cohort["race"].map(_normalize_race)
    cohort["gender"] = cohort["gender"].fillna("UNKNOWN").astype(str).str.upper()
    cohort["arrival_transport"] = cohort["arrival_transport"].fillna("UNKNOWN").astype(str).str.upper()
    cohort["age_group"] = pd.cut(
        cohort["anchor_age"].fillna(-1),
        bins=[-2, 40, 65, 120],
        labels=["age_le_40", "age_41_65", "age_gt_65"],
    ).astype(str)
    cohort["acuity_group"] = pd.cut(
        cohort["acuity"].fillna(9),
        bins=[0, 2, 3, 9],
        labels=["acuity_1_2", "acuity_3", "acuity_4_5_or_unknown"],
    ).astype(str)
    cohort["race_acuity"] = cohort["race"] + "_" + cohort["acuity_group"]


def _build_documents(cohort: pd.DataFrame, feature_mode: str) -> pd.Series:
    complaint = cohort["chiefcomplaint"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    if feature_mode == "text":
        return complaint
    if feature_mode != "text_triage":
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    tokens = pd.DataFrame(index=cohort.index)
    tokens["transport"] = cohort["arrival_transport"].map(lambda value: f"transport_{_token(value)}")
    tokens["acuity"] = cohort["acuity_group"]
    for column, bins in {
        "heartrate": [-np.inf, 60, 100, np.inf],
        "resprate": [-np.inf, 12, 20, np.inf],
        "o2sat": [-np.inf, 92, 96, np.inf],
        "sbp": [-np.inf, 90, 140, np.inf],
        "temperature": [-np.inf, 96.8, 100.4, np.inf],
    }.items():
        values = pd.cut(cohort[column], bins=bins, labels=["low", "normal", "high"]).astype(str)
        tokens[column] = [f"{column}_{value}" if value != "nan" else f"{column}_unknown" for value in values]
    return complaint + " " + tokens.agg(" ".join, axis=1)


def _patient_level_splits(cohort: pd.DataFrame, seed: int) -> tuple[np.ndarray, ...]:
    patient_labels = cohort.groupby("subject_id")["label"].agg(lambda values: int(values.mode().iloc[0]))
    rng = np.random.default_rng(seed)
    split_subjects = {name: [] for name in ["train", "calib0", "discovery", "mvcal", "test"]}
    for label in range(len(CARE_CLASSES)):
        subjects = rng.permutation(patient_labels.index[patient_labels == label].to_numpy())
        cuts = (np.asarray([0.50, 0.65, 0.80, 0.90]) * len(subjects)).astype(int)
        pieces = np.split(subjects, cuts)
        for name, piece in zip(split_subjects, pieces):
            split_subjects[name].extend(piece.tolist())
    return tuple(
        cohort.index[cohort["subject_id"].isin(split_subjects[name])].to_numpy(dtype=np.int64)
        for name in split_subjects
    )


def _normalize_race(value: object) -> str:
    text = str(value).upper()
    for group in ["WHITE", "BLACK", "HISPANIC", "ASIAN"]:
        if group in text:
            return group
    return "OTHER" if text not in {"NAN", "NONE", ""} else "UNKNOWN"


def _token(value: object) -> str:
    return str(value).lower().replace(" ", "_").replace("/", "_")

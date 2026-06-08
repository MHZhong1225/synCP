from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import re
import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
import pandas as pd

from core.progress import progress

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class ImageExperimentData:
    dataset: Dataset
    labels: np.ndarray
    class_names: list[str]
    train_idx: np.ndarray
    calib0_idx: np.ndarray
    discovery_idx: np.ndarray
    mvcal_idx: np.ndarray
    test_idx: np.ndarray
    known_group: np.ndarray | None = None
    sensitive_groups: dict[str, np.ndarray] | None = None


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


class ImageRecordsDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


class DicomRecordsDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = _read_dicom_image(self.paths[index])
        return self.transform(image), int(self.labels[index])


class Hdf5ImageDataset(Dataset):
    def __init__(self, path: Path, image_ids: list[str], labels: np.ndarray, transform) -> None:
        self.path = path
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform
        self._file = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        if self._file is None:
            import h5py

            self._file = h5py.File(self.path, "r")
        image = Image.open(BytesIO(self._file[self.image_ids[index]][()])).convert("RGB")
        return self.transform(image), int(self.labels[index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state


def load_imagefolder_experiment(
    root: Path,
    seed: int = 7,
    image_size: int = 224,
    split_mode: str = "official_test",
    train_fraction: float = 0.50,
    calib0_fraction: float = 0.15,
    discovery_fraction: float = 0.15,
    mvcal_fraction: float = 0.10,
) -> ImageExperimentData:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected ImageFolder train/test directories under {root}")

    train_base = datasets.ImageFolder(train_dir, transform=transform)
    class_names = train_base.classes
    parts: list[Dataset] = [train_base]
    train_labels = [np.asarray(train_base.targets, dtype=np.int64)]
    if val_dir.exists():
        val_base = datasets.ImageFolder(val_dir, transform=transform)
        if val_base.classes != class_names:
            raise ValueError("train and val class names differ")
        parts.append(val_base)
        train_labels.append(np.asarray(val_base.targets, dtype=np.int64))
    test_base = datasets.ImageFolder(test_dir, transform=transform)
    if test_base.classes != class_names:
        raise ValueError("train and test class names differ")

    trainval = ConcatDataset(parts) if len(parts) > 1 else train_base
    labels_trainval = np.concatenate(train_labels)
    labels_test = np.asarray(test_base.targets, dtype=np.int64)
    dataset = ConcatDataset([trainval, test_base])
    labels = np.concatenate([labels_trainval, labels_test])

    rng = np.random.default_rng(seed)
    if split_mode == "official_test":
        split_pool = rng.permutation(len(labels_trainval))
        test_idx = np.arange(len(labels_trainval), len(labels), dtype=np.int64)
    elif split_mode == "random_all":
        all_idx = rng.permutation(len(labels))
        n_test = max(1, int(0.10 * len(all_idx)))
        test_idx = all_idx[-n_test:]
        split_pool = all_idx[:-n_test]
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    n_trainval = len(split_pool)
    a = int(train_fraction * n_trainval)
    b = int((train_fraction + calib0_fraction) * n_trainval)
    c = int((train_fraction + calib0_fraction + discovery_fraction) * n_trainval)
    d = int((train_fraction + calib0_fraction + discovery_fraction + mvcal_fraction) * n_trainval)

    return ImageExperimentData(
        dataset=dataset,
        labels=labels,
        class_names=class_names,
        train_idx=split_pool[:a],
        calib0_idx=split_pool[a:b],
        discovery_idx=split_pool[b:c],
        mvcal_idx=split_pool[c:d],
        test_idx=test_idx,
        known_group=None,
    )


def load_rsna_breast_experiment(
    root: Path,
    known_group: str = "site",
    seed: int = 7,
    image_size: int = 224,
    max_samples: int | None = None,
    train_fraction: float = 0.50,
    calib0_fraction: float = 0.15,
    discovery_fraction: float = 0.15,
    mvcal_fraction: float = 0.10,
) -> ImageExperimentData:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metadata = pd.read_csv(root / "train.csv")
    metadata["path"] = [
        root / "train_images" / str(patient_id) / f"{image_id}.dcm"
        for patient_id, image_id in zip(metadata["patient_id"], metadata["image_id"])
    ]
    metadata = metadata[metadata["path"].map(Path.exists)].copy()
    metadata = metadata[metadata["path"].map(_dicom_header_is_readable)].copy()
    if max_samples is not None and len(metadata) > max_samples:
        metadata = _balanced_image_sample(metadata, "cancer", max_samples, seed)
    metadata = metadata.reset_index(drop=True)

    labels = metadata["cancer"].to_numpy(dtype=np.int64)
    paths = metadata["path"].tolist()
    dataset = DicomRecordsDataset(paths, labels, transform)
    train_idx, calib0_idx, discovery_idx, mvcal_idx, test_idx = _stratified_patient_splits(
        metadata,
        seed=seed,
        train_fraction=train_fraction,
        calib0_fraction=calib0_fraction,
        discovery_fraction=discovery_fraction,
        mvcal_fraction=mvcal_fraction,
    )
    sensitive_groups = _rsna_sensitive_groups(metadata)
    encoded_group = _encode_group(sensitive_groups[known_group])
    return ImageExperimentData(
        dataset=dataset,
        labels=labels,
        class_names=["no_cancer", "cancer"],
        train_idx=train_idx,
        calib0_idx=calib0_idx,
        discovery_idx=discovery_idx,
        mvcal_idx=mvcal_idx,
        test_idx=test_idx,
        known_group=encoded_group,
        sensitive_groups=sensitive_groups,
    )


def load_nih_experiment(
    root: Path,
    target: str = "Effusion",
    known_group: str = "gender",
    seed: int = 7,
    image_size: int = 224,
    max_samples: int | None = None,
    split_mode: str = "official_test",
    train_fraction: float = 0.50,
    calib0_fraction: float = 0.15,
    discovery_fraction: float = 0.15,
    mvcal_fraction: float = 0.10,
) -> ImageExperimentData:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metadata = pd.read_csv(root / "Data_Entry_2017.csv")
    path_map = {path.name: path for path in root.glob("images*/**/*.png")}
    metadata = metadata[metadata["Image Index"].isin(path_map)].copy()
    finding_sets = metadata["Finding Labels"].str.split("|").apply(set)
    metadata = metadata[
        finding_sets.apply(lambda findings: target in findings or findings == {"No Finding"})
    ].copy()
    metadata["label"] = metadata["Finding Labels"].str.split("|").apply(lambda findings: int(target in findings))
    metadata["split"] = "unused"
    train_names = set((root / "train_val_list.txt").read_text(encoding="utf-8").splitlines())
    test_names = set((root / "test_list.txt").read_text(encoding="utf-8").splitlines())
    metadata.loc[metadata["Image Index"].isin(train_names), "split"] = "trainval"
    metadata.loc[metadata["Image Index"].isin(test_names), "split"] = "test"
    metadata = metadata[metadata["split"].isin(["trainval", "test"])].copy()

    if max_samples is not None and len(metadata) > max_samples:
        metadata = _balanced_metadata_sample(metadata, max_samples, seed)

    metadata = metadata.reset_index(drop=True)
    paths = [path_map[name] for name in metadata["Image Index"]]
    labels = metadata["label"].to_numpy(dtype=np.int64)
    sensitive_groups = _nih_sensitive_groups(metadata)
    encoded_groups = _encode_group(_nih_group_values(metadata, known_group))

    dataset = ImageRecordsDataset(paths, labels, transform)
    rng = np.random.default_rng(seed)
    if split_mode == "official_test":
        split_pool = rng.permutation(metadata.index[metadata["split"] == "trainval"].to_numpy(dtype=np.int64))
        test_idx = metadata.index[metadata["split"] == "test"].to_numpy(dtype=np.int64)
    elif split_mode == "random_all":
        all_idx = rng.permutation(len(metadata))
        n_test = max(1, int(0.10 * len(all_idx)))
        test_idx = all_idx[-n_test:]
        split_pool = all_idx[:-n_test]
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    n_trainval = len(split_pool)
    a = int(train_fraction * n_trainval)
    b = int((train_fraction + calib0_fraction) * n_trainval)
    c = int((train_fraction + calib0_fraction + discovery_fraction) * n_trainval)
    d = int((train_fraction + calib0_fraction + discovery_fraction + mvcal_fraction) * n_trainval)

    return ImageExperimentData(
        dataset=dataset,
        labels=labels,
        class_names=[f"not_{target}", target],
        train_idx=split_pool[:a],
        calib0_idx=split_pool[a:b],
        discovery_idx=split_pool[b:c],
        mvcal_idx=split_pool[c:d],
        test_idx=test_idx,
        known_group=encoded_groups,
        sensitive_groups=sensitive_groups,
    )


def load_ham10000_experiment(
    root: Path,
    known_group: str = "sex",
    seed: int = 7,
    image_size: int = 224,
    max_samples: int | None = None,
    train_fraction: float = 0.50,
    calib0_fraction: float = 0.15,
    discovery_fraction: float = 0.15,
    mvcal_fraction: float = 0.10,
) -> ImageExperimentData:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metadata = pd.read_csv(root / "HAM10000_metadata.csv")
    path_map = {path.stem: path for path in root.glob("HAM10000_images_part_*/*.jpg")}
    metadata = metadata[metadata["image_id"].isin(path_map)].copy()
    if max_samples is not None and len(metadata) > max_samples:
        metadata = _balanced_image_sample(metadata, "dx", max_samples, seed)
    metadata = metadata.reset_index(drop=True)

    class_names = sorted(metadata["dx"].dropna().unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = metadata["dx"].map(class_to_index).to_numpy(dtype=np.int64)
    paths = [path_map[image_id] for image_id in metadata["image_id"]]
    sensitive_groups = _ham10000_sensitive_groups(metadata)
    encoded_groups = _encode_group(_ham10000_group_values(metadata, known_group))

    dataset = ImageRecordsDataset(paths, labels, transform)
    train_idx, calib0_idx, discovery_idx, mvcal_idx, test_idx = _stratified_lesion_splits(
        metadata,
        labels,
        seed=seed,
        train_fraction=train_fraction,
        calib0_fraction=calib0_fraction,
        discovery_fraction=discovery_fraction,
        mvcal_fraction=mvcal_fraction,
    )
    return ImageExperimentData(
        dataset=dataset,
        labels=labels,
        class_names=class_names,
        train_idx=train_idx,
        calib0_idx=calib0_idx,
        discovery_idx=discovery_idx,
        mvcal_idx=mvcal_idx,
        test_idx=test_idx,
        known_group=encoded_groups.astype(np.int64),
        sensitive_groups=sensitive_groups,
    )



def _exclude_ham10000_images(metadata: pd.DataFrame, ham_root: Path) -> pd.DataFrame:
    ham_metadata = ham_root / HAM10000_metadata.csv
    if not ham_metadata.exists():
        return metadata
    ham_ids = pd.read_csv(ham_metadata, usecols=[image_id])[image_id].astype(str)
    return metadata[~metadata[isic_id].astype(str).isin(set(ham_ids))].copy()


def load_isic_experiment(
    root: Path,
    known_group: str = "anatom_site_general",
    seed: int = 7,
    image_size: int = 224,
    max_samples: int | None = 20000,
    train_fraction: float = 0.50,
    calib0_fraction: float = 0.15,
    discovery_fraction: float = 0.15,
    mvcal_fraction: float = 0.10,
) -> ImageExperimentData:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metadata = pd.read_csv(root / "metadata.csv", low_memory=False)
    image_store = root / "image_256sq.hdf5"
    if not image_store.exists():
        raise FileNotFoundError(f"Expected ISIC image store: {image_store}")
    metadata = metadata[metadata["image_type"].eq("dermoscopic") & metadata["diagnosis"].notna()].copy()
    common_diagnoses = metadata["diagnosis"].value_counts()
    metadata = metadata[metadata["diagnosis"].isin(common_diagnoses[common_diagnoses >= 300].index)].copy()
    if max_samples is not None and len(metadata) > max_samples:
        metadata = _balanced_image_sample(metadata, "diagnosis", max_samples, seed)
    metadata = metadata.reset_index(drop=True)

    class_names = sorted(metadata["diagnosis"].unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = metadata["diagnosis"].map(class_to_index).to_numpy(dtype=np.int64)
    metadata["label_index"] = labels
    metadata["entity_id"] = _isic_entity_ids(metadata)
    sensitive_groups = _isic_sensitive_groups(metadata)
    encoded_groups = _encode_group(_isic_group_values(metadata, known_group))
    splits = _stratified_entity_splits(
        metadata,
        entity_column="entity_id",
        label_column="label_index",
        seed=seed,
        train_fraction=train_fraction,
        calib0_fraction=calib0_fraction,
        discovery_fraction=discovery_fraction,
        mvcal_fraction=mvcal_fraction,
    )
    return ImageExperimentData(
        dataset=Hdf5ImageDataset(image_store, metadata["isic_id"].astype(str).tolist(), labels, transform),
        labels=labels,
        class_names=class_names,
        train_idx=splits[0],
        calib0_idx=splits[1],
        discovery_idx=splits[2],
        mvcal_idx=splits[3],
        test_idx=splits[4],
        known_group=encoded_groups.astype(np.int64),
        sensitive_groups=sensitive_groups,
    )


def load_mimic_joint_experiment(
    root: Path,
    mimic_iv_root: Path,
    mimic_ed_root: Path,
    target: str = "Pleural Effusion",
    known_group: str = "race",
    seed: int = 7,
    image_size: int = 224,
    max_samples: int | None = 5000,
    train_fraction: float = 0.50,
    calib0_fraction: float = 0.15,
    discovery_fraction: float = 0.15,
    mvcal_fraction: float = 0.10,
) -> ImageExperimentData:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metadata = _load_mimic_cxr_records(root, target)
    if max_samples is not None and len(metadata) > max_samples:
        metadata = _balanced_mimic_sample(metadata, max_samples, seed)
    metadata = _join_mimic_clinical_metadata(metadata, mimic_iv_root, mimic_ed_root)
    metadata = metadata.reset_index(drop=True)

    labels = metadata["label"].to_numpy(dtype=np.int64)
    dataset = ImageRecordsDataset(metadata["path"].tolist(), labels, transform)
    train_idx, calib0_idx, discovery_idx, mvcal_idx, test_idx = _mimic_patient_splits(
        metadata,
        seed=seed,
        train_fraction=train_fraction,
        calib0_fraction=calib0_fraction,
        discovery_fraction=discovery_fraction,
        mvcal_fraction=mvcal_fraction,
    )
    sensitive_groups = _mimic_sensitive_groups(metadata)
    encoded_groups = _encode_group(_mimic_group_values(metadata, known_group))
    return ImageExperimentData(
        dataset=dataset,
        labels=labels,
        class_names=[f"not_{target}", target],
        train_idx=train_idx,
        calib0_idx=calib0_idx,
        discovery_idx=discovery_idx,
        mvcal_idx=mvcal_idx,
        test_idx=test_idx,
        known_group=encoded_groups.astype(np.int64),
        sensitive_groups=sensitive_groups,
    )


MIMIC_TARGET_TERMS = {
    "Pleural Effusion": (r"pleural effusion", r"pleural fluid"),
    "Pneumonia": (r"pneumonia",),
    "Pneumothorax": (r"pneumothorax",),
    "Pulmonary Edema": (r"pulmonary edema", r"interstitial edema"),
    "Cardiomegaly": (r"cardiomegaly", r"cardiac enlargement", r"enlarged cardiac silhouette"),
}


def _load_mimic_cxr_records(root: Path, target: str) -> pd.DataFrame:
    if target not in MIMIC_TARGET_TERMS:
        raise ValueError(f"Unknown MIMIC target: {target}")
    records = pd.read_csv(root / "cxr-record-list.csv.gz", usecols=["subject_id", "study_id", "path"])
    records["path"] = records["path"].map(lambda path: root / Path(path).with_suffix(".jpg"))
    records = records[records["path"].map(Path.exists)].copy()

    study_labels = _load_mimic_study_labels(root, target)
    records["label"] = records["study_id"].map(study_labels)
    return records.dropna(subset=["label"]).astype({"label": np.int64})


def _load_mimic_study_labels(root: Path, target: str) -> dict[int, int]:
    return _load_mimic_all_study_labels(root)[target]


@lru_cache(maxsize=2)
def _load_mimic_all_study_labels(root: Path) -> dict[str, dict[int, int]]:
    labels = {target: {} for target in MIMIC_TARGET_TERMS}
    with ZipFile(root / "mimic-cxr-reports.zip") as reports:
        for name in reports.namelist():
            if not name.endswith(".txt"):
                continue
            study_id = int(Path(name).stem.removeprefix("s"))
            report = reports.read(name).decode("utf-8", errors="ignore")
            for target in labels:
                labels[target][study_id] = _mimic_report_label(report, target)
    return labels


def _mimic_report_label(report: str, target: str) -> int:
    text = report.lower()
    clauses = re.split(r"[.;]|\bbut\b|\bhowever\b", text)
    terms = MIMIC_TARGET_TERMS[target]
    negative = re.compile(r"(?:no|without|absence of|free of|negative for|no evidence of|rule out)[^.;]{0,100}$")
    for clause in clauses:
        for term in terms:
            for match in re.finditer(term, clause):
                prefix = clause[max(0, match.start() - 80):match.start()]
                if not negative.search(prefix):
                    return 1
    return 0


def _join_mimic_clinical_metadata(metadata: pd.DataFrame, mimic_iv_root: Path, mimic_ed_root: Path) -> pd.DataFrame:
    patients = pd.read_csv(mimic_iv_root / "hosp" / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(
        mimic_iv_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "race", "insurance", "hospital_expire_flag"],
    )
    admissions["race"] = admissions["race"].map(_normalize_mimic_race)
    admission_summary = admissions.groupby("subject_id", as_index=False).agg(
        race=("race", _mode_or_unknown),
        insurance=("insurance", _mode_or_unknown),
        hospital_expire_flag=("hospital_expire_flag", "max"),
    )
    edstays = pd.read_csv(
        mimic_ed_root / "ed" / "edstays.csv.gz",
        usecols=["subject_id", "stay_id", "arrival_transport", "disposition"],
    )
    ed_summary = edstays.groupby("subject_id", as_index=False).agg(
        arrival_transport=("arrival_transport", _mode_or_unknown),
        ed_disposition=("disposition", _mode_or_unknown),
    )
    triage = pd.read_csv(mimic_ed_root / "ed" / "triage.csv.gz", usecols=["stay_id", "acuity"])
    acuity = edstays[["subject_id", "stay_id"]].merge(triage, on="stay_id", how="left")
    acuity = acuity.groupby("subject_id", as_index=False)["acuity"].median().rename(columns={"acuity": "ed_acuity"})
    icu_subjects = set(pd.read_csv(mimic_iv_root / "icu" / "icustays.csv.gz", usecols=["subject_id"])["subject_id"].unique())

    joined = metadata.merge(patients, on="subject_id", how="left")
    joined = joined.merge(admission_summary, on="subject_id", how="left")
    joined = joined.merge(ed_summary, on="subject_id", how="left")
    joined = joined.merge(acuity, on="subject_id", how="left")
    joined["icu_history"] = joined["subject_id"].isin(icu_subjects).astype(np.int64)
    return joined


def _mode_or_unknown(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "UNKNOWN"
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(values.iloc[0])


def _normalize_mimic_race(value: object) -> str:
    text = str(value).upper()
    for group in ["WHITE", "BLACK", "HISPANIC", "ASIAN"]:
        if group in text:
            return group
    return "OTHER"


def _balanced_mimic_sample(metadata: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = []
    per_label = max(1, max_samples // 2)
    for label in [0, 1]:
        indices = metadata.index[metadata["label"] == label].to_numpy()
        take = min(len(indices), per_label)
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) < max_samples:
        remaining = np.setdiff1d(metadata.index.to_numpy(), np.asarray(selected, dtype=np.int64))
        take = min(len(remaining), max_samples - len(selected))
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return metadata.loc[selected].sample(frac=1.0, random_state=seed)


def _mimic_patient_splits(
    metadata: pd.DataFrame,
    seed: int,
    train_fraction: float,
    calib0_fraction: float,
    discovery_fraction: float,
    mvcal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patient_labels = metadata.groupby("subject_id")["label"].max()
    rng = np.random.default_rng(seed)
    split_subjects = {"train": [], "calib0": [], "discovery": [], "mvcal": [], "test": []}
    for label in [0, 1]:
        subject_ids = rng.permutation(patient_labels.index[patient_labels == label].to_numpy())
        n = len(subject_ids)
        a = int(train_fraction * n)
        b = int((train_fraction + calib0_fraction) * n)
        c = int((train_fraction + calib0_fraction + discovery_fraction) * n)
        d = int((train_fraction + calib0_fraction + discovery_fraction + mvcal_fraction) * n)
        split_subjects["train"].extend(subject_ids[:a].tolist())
        split_subjects["calib0"].extend(subject_ids[a:b].tolist())
        split_subjects["discovery"].extend(subject_ids[b:c].tolist())
        split_subjects["mvcal"].extend(subject_ids[c:d].tolist())
        split_subjects["test"].extend(subject_ids[d:].tolist())
    return tuple(
        metadata.index[metadata["subject_id"].isin(split_subjects[name])].to_numpy(dtype=np.int64)
        for name in ["train", "calib0", "discovery", "mvcal", "test"]
    )


def _mimic_group_values(metadata: pd.DataFrame, group: str) -> np.ndarray:
    age = pd.cut(metadata["anchor_age"].fillna(-1), bins=[-2, 40, 65, 120], labels=["age_le_40", "age_41_65", "age_gt_65"]).astype(str)
    gender = metadata["gender"].fillna("UNKNOWN").astype(str)
    race = metadata["race"].fillna("UNKNOWN").astype(str)
    arrival = metadata["arrival_transport"].fillna("UNKNOWN").astype(str)
    acuity = pd.cut(metadata["ed_acuity"].fillna(9), bins=[0, 2, 3, 9], labels=["acuity_1_2", "acuity_3", "acuity_4_5_or_unknown"]).astype(str)
    values = {
        "gender": gender,
        "age": age,
        "race": race,
        "insurance": metadata["insurance"].fillna("UNKNOWN").astype(str),
        "arrival": arrival,
        "race_age": race + "_" + age,
        "gender_age": gender + "_" + age,
        "ed_acuity": acuity,
        "icu_history": metadata["icu_history"].astype(str),
    }
    if group not in values:
        raise ValueError(f"Unknown MIMIC known group: {group}")
    return values[group].to_numpy()


def _mimic_sensitive_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: _mimic_group_values(metadata, name)
        for name in ["gender", "age", "race", "insurance", "arrival", "ed_acuity", "icu_history"]
    }


def _read_dicom_image(path: Path) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("RSNA breast DICOM loading requires pydicom.") from exc

    dicom = pydicom.dcmread(path)
    if not hasattr(dicom, "PixelData"):
        raise ValueError(f"DICOM file has no pixel data: {path}")
    array = dicom.pixel_array.astype(np.float32)
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = array.max() - array
    low, high = np.percentile(array, [1.0, 99.0])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        normalized = np.zeros_like(array, dtype=np.uint8)
    else:
        normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
        normalized = (normalized * 255.0).astype(np.uint8)
    return Image.fromarray(normalized).convert("RGB")


def _dicom_header_is_readable(path: Path) -> bool:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("RSNA breast DICOM loading requires pydicom.") from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return False
    return True


def _balanced_image_sample(metadata: pd.DataFrame, label_column: str, max_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_label = max(1, max_samples // metadata[label_column].nunique())
    selected: list[int] = []
    for label in sorted(metadata[label_column].dropna().unique()):
        label_idx = metadata.index[metadata[label_column] == label].to_numpy()
        take = min(len(label_idx), per_label)
        selected.extend(rng.choice(label_idx, size=take, replace=False).tolist())
    if len(selected) < max_samples:
        remaining = np.setdiff1d(metadata.index.to_numpy(), np.asarray(selected, dtype=np.int64))
        take = min(len(remaining), max_samples - len(selected))
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return metadata.loc[selected].sample(frac=1.0, random_state=seed)


def _stratified_patient_splits(
    metadata: pd.DataFrame,
    seed: int,
    train_fraction: float,
    calib0_fraction: float,
    discovery_fraction: float,
    mvcal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patient_labels = metadata.groupby("patient_id")["cancer"].max()
    split_patient_ids = {"train": [], "calib0": [], "discovery": [], "mvcal": [], "test": []}
    rng = np.random.default_rng(seed)
    for label in [0, 1]:
        patient_ids = patient_labels.index[patient_labels == label].to_numpy()
        patient_ids = rng.permutation(patient_ids)
        n = len(patient_ids)
        a = int(train_fraction * n)
        b = int((train_fraction + calib0_fraction) * n)
        c = int((train_fraction + calib0_fraction + discovery_fraction) * n)
        d = int((train_fraction + calib0_fraction + discovery_fraction + mvcal_fraction) * n)
        split_patient_ids["train"].extend(patient_ids[:a].tolist())
        split_patient_ids["calib0"].extend(patient_ids[a:b].tolist())
        split_patient_ids["discovery"].extend(patient_ids[b:c].tolist())
        split_patient_ids["mvcal"].extend(patient_ids[c:d].tolist())
        split_patient_ids["test"].extend(patient_ids[d:].tolist())

    return tuple(
        metadata.index[metadata["patient_id"].isin(split_patient_ids[name])].to_numpy(dtype=np.int64)
        for name in ["train", "calib0", "discovery", "mvcal", "test"]
    )


def _rsna_sensitive_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    age = pd.cut(
        metadata["age"].fillna(-1),
        bins=[-2, 40, 50, 60, 70, 120],
        labels=["age_unknown_or_le_40", "age_41_50", "age_51_60", "age_61_70", "age_gt_70"],
    ).astype(str)
    return {
        "age": age.to_numpy(),
        "site": metadata["site_id"].fillna("UNKNOWN").astype(str).to_numpy(),
    }


def _stratified_lesion_splits(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
    train_fraction: float,
    calib0_fraction: float,
    discovery_fraction: float,
    mvcal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    metadata = metadata.copy()
    metadata["label_index"] = labels
    lesion_labels = metadata.groupby("lesion_id")["label_index"].first()
    split_lesions = {"train": [], "calib0": [], "discovery": [], "mvcal": [], "test": []}
    for label in sorted(lesion_labels.unique()):
        lesion_ids = lesion_labels.index[lesion_labels == label].to_numpy()
        lesion_ids = rng.permutation(lesion_ids)
        n = len(lesion_ids)
        a = int(train_fraction * n)
        b = int((train_fraction + calib0_fraction) * n)
        c = int((train_fraction + calib0_fraction + discovery_fraction) * n)
        d = int((train_fraction + calib0_fraction + discovery_fraction + mvcal_fraction) * n)
        split_lesions["train"].extend(lesion_ids[:a].tolist())
        split_lesions["calib0"].extend(lesion_ids[a:b].tolist())
        split_lesions["discovery"].extend(lesion_ids[b:c].tolist())
        split_lesions["mvcal"].extend(lesion_ids[c:d].tolist())
        split_lesions["test"].extend(lesion_ids[d:].tolist())

    return tuple(
        metadata.index[metadata["lesion_id"].isin(split_lesions[name])].to_numpy(dtype=np.int64)
        for name in ["train", "calib0", "discovery", "mvcal", "test"]
    )


def _stratified_entity_splits(
    metadata: pd.DataFrame,
    entity_column: str,
    label_column: str,
    seed: int,
    train_fraction: float,
    calib0_fraction: float,
    discovery_fraction: float,
    mvcal_fraction: float,
) -> tuple[np.ndarray, ...]:
    entity_labels = metadata.groupby(entity_column)[label_column].agg(lambda values: int(values.mode().iloc[0]))
    rng = np.random.default_rng(seed)
    split_entities = {name: [] for name in ["train", "calib0", "discovery", "mvcal", "test"]}
    for label in sorted(entity_labels.unique()):
        entities = rng.permutation(entity_labels.index[entity_labels == label].to_numpy())
        cuts = (
            np.asarray(
                [
                    train_fraction,
                    train_fraction + calib0_fraction,
                    train_fraction + calib0_fraction + discovery_fraction,
                    train_fraction + calib0_fraction + discovery_fraction + mvcal_fraction,
                ]
            )
            * len(entities)
        ).astype(int)
        for name, values in zip(split_entities, np.split(entities, cuts)):
            split_entities[name].extend(values.tolist())
    return tuple(
        metadata.index[metadata[entity_column].isin(split_entities[name])].to_numpy(dtype=np.int64)
        for name in split_entities
    )


def _isic_entity_ids(metadata: pd.DataFrame) -> np.ndarray:
    patient = metadata["patient_id"].fillna("").astype(str)
    lesion = metadata["lesion_id"].fillna("").astype(str)
    image = metadata["isic_id"].astype(str)
    return np.where(patient.ne(""), "patient:" + patient, np.where(lesion.ne(""), "lesion:" + lesion, "image:" + image))


def _isic_group_values(metadata: pd.DataFrame, group: str) -> np.ndarray:
    values = {
        "source": metadata["attribution"].fillna("UNKNOWN").astype(str),
        "sex": metadata["sex"].fillna("UNKNOWN").astype(str),
        "age_group": pd.cut(metadata["age_approx"].fillna(-1), bins=[-2, 30, 50, 70, 120], labels=["age_unknown_or_le_30", "age_31_50", "age_51_70", "age_gt_70"]).astype(str),
        "anatom_site": metadata["anatom_site_general"].fillna("UNKNOWN").astype(str),
        "confirmation": metadata["diagnosis_confirm_type"].fillna("UNKNOWN").astype(str),
        "skin_type": metadata["fitzpatrick_skin_type"].fillna("UNKNOWN").astype(str),
    }
    values["age_approx"] = values["age_group"]
    values["anatom_site_general"] = values["anatom_site"]
    if group not in values:
        raise ValueError(f"Unknown ISIC known group: {group}")
    return values[group].to_numpy()


def _isic_sensitive_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: _isic_group_values(metadata, name)
        for name in ["anatom_site_general", "age_approx", "sex"]
    }


def _balanced_metadata_sample(metadata: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    per_split = max_samples // 2
    for split in ["trainval", "test"]:
        part = metadata[metadata["split"] == split]
        split_target = min(len(part), per_split)
        per_class = max(1, split_target // 2)
        selected = []
        for label in [0, 1]:
            label_idx = part.index[part["label"] == label].to_numpy()
            take = min(len(label_idx), per_class)
            if take:
                selected.extend(rng.choice(label_idx, size=take, replace=False).tolist())
        if len(selected) < split_target:
            remaining = np.setdiff1d(part.index.to_numpy(), np.asarray(selected, dtype=np.int64))
            take = min(len(remaining), split_target - len(selected))
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
        pieces.append(metadata.loc[selected])
    return pd.concat(pieces, ignore_index=True)


def _nih_group_values(metadata: pd.DataFrame, group: str) -> np.ndarray:
    age = pd.cut(metadata["Patient Age"].clip(0, 100), bins=[-1, 40, 60, 120], labels=["age_le_40", "age_41_60", "age_gt_60"]).astype(str)
    gender = metadata["Patient Gender"].astype(str)
    view = metadata["View Position"].astype(str)
    if group == "gender":
        return gender.to_numpy()
    if group == "age":
        return age.to_numpy()
    if group == "view":
        return view.to_numpy()
    if group == "gender_age":
        return (gender + "_" + age).to_numpy()
    if group == "gender_view":
        return (gender + "_" + view).to_numpy()
    raise ValueError(f"Unknown NIH known group: {group}")


def _nih_sensitive_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: _nih_group_values(metadata, name) for name in ["gender", "age"]}


def _ham10000_group_values(metadata: pd.DataFrame, group: str) -> np.ndarray:
    if group == "sex":
        return metadata["sex"].fillna("unknown").astype(str).to_numpy()
    if group == "localization":
        return metadata["localization"].fillna("unknown").astype(str).to_numpy()
    if group == "dx_type":
        return metadata["dx_type"].fillna("unknown").astype(str).to_numpy()
    if group == "age":
        age = metadata["age"].fillna(-1).clip(-1, 100)
        return pd.cut(age, bins=[-2, 30, 50, 70, 120], labels=["age_unknown_or_le_30", "age_31_50", "age_51_70", "age_gt_70"]).astype(str).to_numpy()
    raise ValueError(f"Unknown HAM10000 known group: {group}")


def _ham10000_sensitive_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: _ham10000_group_values(metadata, name) for name in ["sex", "age", "localization"]}


def _encode_group(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone, in_features = _build_backbone(backbone, pretrained, checkpoint)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def train_image_classifier(
    dataset: Dataset,
    train_idx: np.ndarray,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 7,
    pretrained: bool = True,
    backbone: str = "resnet18",
    checkpoint: Path | None = None,
) -> ImageClassifier:
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = device.startswith("cuda")
    model = ImageClassifier(num_classes=num_classes, backbone=backbone, pretrained=pretrained, checkpoint=checkpoint).to(device)
    loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        **_loader_kwargs(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in progress(range(epochs), desc="train backbone", leave=False):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_image_dataset(
    model: ImageClassifier,
    dataset: Dataset,
    device: str = "cuda",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    indexed = IndexedDataset(dataset)
    loader = DataLoader(indexed, batch_size=batch_size, shuffle=False, **_loader_kwargs(device))
    model.eval()
    prob_batches: list[np.ndarray] = []
    embedding_batches: list[np.ndarray] = []
    for xb, _, indices in progress(loader, desc="extract features", total=len(loader), leave=False):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        prob_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        embedding_batches.append(model.embed(xb).cpu().numpy())
    return np.concatenate(prob_batches, axis=0), np.concatenate(embedding_batches, axis=0).astype(np.float32)


def _loader_kwargs(device: str) -> dict[str, object]:
    workers = 20
    return {
        "num_workers": workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": workers > 0,
        "prefetch_factor": 4,
    }


def _build_backbone(backbone: str, pretrained: bool, checkpoint: Path | None) -> tuple[nn.Module, int]:
    name = backbone.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name in {"densenet121", "chexnet"}:
        if name == "chexnet" and checkpoint is None:
            raise ValueError("chexnet requires --backbone-checkpoint pointing to a DenseNet-121 CheXNet checkpoint.")
        model = _DenseNetBackbone(pretrained=pretrained)
        in_features = model.out_features
    elif name in {"radimagenet-resnet50", "radimagenet-densenet121"}:
        if checkpoint is None:
            raise ValueError(f"{backbone} requires --backbone-checkpoint pointing to RadImageNet weights.")
        model, in_features = _build_backbone(name.removeprefix("radimagenet-"), pretrained=False, checkpoint=None)
    else:
        raise ValueError(f"Unknown image backbone: {backbone}")

    if checkpoint is not None:
        _load_checkpoint(model, checkpoint)
    return model, in_features


class _DenseNetBackbone(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features
        self.out_features = densenet.classifier.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.features(x), inplace=True)
        pooled = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        return torch.flatten(pooled, 1)


def _load_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    cleaned = {_clean_checkpoint_key(key): value for key, value in state.items()}
    current = model.state_dict()
    matched = {key: value for key, value in cleaned.items() if key in current and current[key].shape == value.shape}
    if not matched:
        raise ValueError(f"No backbone parameters in {checkpoint} matched the selected architecture.")
    model.load_state_dict(matched, strict=False)


def _clean_checkpoint_key(key: str) -> str:
    key = re.sub(r"^(module\.|model\.|backbone\.)+", "", key)
    if key.startswith("features."):
        return key
    if key.startswith("backbone.features."):
        return key.removeprefix("backbone.")
    return key

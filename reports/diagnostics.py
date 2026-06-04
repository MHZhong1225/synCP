from __future__ import annotations

import numpy as np
import pandas as pd

from core.conformal import covered
from core.rectification import ThresholdModel
from slices.discovery import GroupFunction


def slice_repair_diagnostics(
    method: str,
    groups: list[GroupFunction],
    model: ThresholdModel,
    z_mv: np.ndarray,
    true_scores_mv: np.ndarray,
    observed_mv: np.ndarray,
    z_test: np.ndarray,
    score_matrix_test: np.ndarray,
    y_test: np.ndarray,
    observed_test: np.ndarray,
    base_threshold: float,
) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame(columns=_columns())

    base_test_covered = covered(score_matrix_test, y_test, base_threshold)
    repaired_thresholds = model.thresholds(z_test, observed_scores=observed_test, score_matrix=score_matrix_test)
    repaired_test_covered = covered(score_matrix_test, y_test, repaired_thresholds)
    group_thresholds = _group_thresholds(model, len(groups), base_threshold)

    rows = []
    for index, group in enumerate(groups):
        mv_mask = group.membership(z_mv, observed_mv)
        test_mask = group.membership(z_test, observed_test)
        group_threshold = float(group_thresholds[index])
        rows.append(
            {
                "method": method,
                "group": group.name,
                "mvcal_size": int(mv_mask.sum()),
                "test_size": int(test_mask.sum()),
                "threshold": group_threshold,
                "threshold_increase": max(0.0, group_threshold - base_threshold),
                "test_coverage_before": _coverage(base_test_covered, test_mask),
                "test_coverage_after": _coverage(repaired_test_covered, test_mask),
            }
        )
    return pd.DataFrame(rows, columns=_columns())


def _group_thresholds(model: ThresholdModel, size: int, base_threshold: float) -> np.ndarray:
    if model.group_thresholds is not None:
        return model.group_thresholds
    if model.group_offsets is not None:
        return base_threshold + model.group_offsets
    return np.full(size, base_threshold, dtype=np.float64)


def _coverage(covered_mask: np.ndarray, group_mask: np.ndarray) -> float:
    if not group_mask.any():
        return float("nan")
    return float(covered_mask[group_mask].mean())


def _columns() -> list[str]:
    return [
        "method",
        "group",
        "mvcal_size",
        "test_size",
        "threshold",
        "threshold_increase",
        "test_coverage_before",
        "test_coverage_after",
    ]

from __future__ import annotations

from pathlib import Path

import pandas as pd


METHOD_ORDER = {
    "Standard CP": 0,
    "Classwise CP": 1,
    "Known-group CP": 2,
    "Error-slice": 10,
    "Error-slice GCP": 11,
    "K-means": 30,
    "K-means GCP": 31,
    "Random halfspace": 70,
    "GMM-unaware": 71,
    "Coverage-Domino": 72,
    "Random halfspace GCP": 73,
    "GMM-unaware GCP": 74,
    "Coverage-Domino GCP": 75,
    "DOCP-MVP": 90,
    "DOCP": 99,
}


def order_methods(frame: pd.DataFrame) -> pd.DataFrame:
    if "method" not in frame.columns:
        return frame.copy()
    ordered = frame.copy()
    ordered["_method_order"] = ordered["method"].map(METHOD_ORDER).fillna(80)
    ordered = ordered.sort_values(["_method_order", "method"]).drop(columns="_method_order")
    return ordered.reset_index(drop=True)


def round_numbers(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    numeric = rounded.select_dtypes(include="number").columns
    rounded[numeric] = rounded[numeric].round(5)
    return rounded


def clean_results(frame: pd.DataFrame) -> pd.DataFrame:
    return round_numbers(order_methods(frame))


def write_results(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    cleaned = clean_results(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index=False, float_format="%.5f")
    return cleaned


def print_results(frame: pd.DataFrame) -> None:
    print(clean_results(frame).to_string(index=False, float_format=lambda value: f"{value:.5f}"))

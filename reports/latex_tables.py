from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENTS = {
    "synthetic": "Synthetic",
    "bach": "BACH",
    "bracs": "BRACS",
    "nih": "NIH Effusion",
    "rsna_5k": "RSNA Breast",
}

METHOD_LABELS = {
    "standard_cp": "Standard CP",
    "classwise_cp": "Classwise CP",
    "known_group_cp": "Known-group CP",
    "classification_error_slice_quantile_baseline": "Error-slice",
    "classification_error_mvcp_baseline": "Error-slice GCP",
    "kmeans_slice_quantile_baseline": "K-means",
    "kmeans_mvcp_baseline": "K-means GCP",
    "random_halfspace_only": "Random halfspace",
    "random_halfspace_batch_gcp_ablation": "Random halfspace GCP",
    "gmm_unaware_only": "GMM-unaware",
    "gmm_unaware_batch_gcp_ablation": "GMM-unaware GCP",
    "coverage_domino_only": "Coverage-Domino",
    "coverage_domino_batch_gcp_ablation": "Coverage-Domino GCP",
    "docp_pool_mvp_ablation": "DOCP-MVP",
    "docp": "DOCP",
    "Standard CP": "Standard CP",
    "Classwise CP": "Classwise CP",
    "Known-group CP": "Known-group CP",
    "Error-slice": "Error-slice",
    "Error-slice GCP": "Error-slice GCP",
    "K-means": "K-means",
    "K-means GCP": "K-means GCP",
    "Random halfspace": "Random halfspace",
    "Random halfspace GCP": "Random halfspace GCP",
    "GMM-unaware": "GMM-unaware",
    "GMM-unaware GCP": "GMM-unaware GCP",
    "Coverage-Domino": "Coverage-Domino",
    "Coverage-Domino GCP": "Coverage-Domino GCP",
    "DOCP-MVP": "DOCP-MVP",
    "DOCP": "DOCP",
}

METHOD_ORDER = {
    "Standard CP": 0,
    "Classwise CP": 1,
    "Known-group CP": 2,
    "Error-slice": 10,
    "K-means": 20,
    "Random halfspace": 30,
    "GMM-unaware": 31,
    "Coverage-Domino": 32,
    "DOCP-MVP": 90,
    "DOCP": 99,
}

MAIN_METHODS = [
    "Standard CP",
    "Classwise CP",
    "Known-group CP",
    "Error-slice",
    "K-means",
    "Random halfspace",
    "GMM-unaware",
    "Coverage-Domino",
    "DOCP",
]

ABLATION_METHODS = [
    "K-means",
    "Random halfspace",
    "GMM-unaware",
    "Coverage-Domino",
    "DOCP-MVP",
    "DOCP",
]

TABLE_COLUMNS = {
    "dataset": "Dataset",
    "method": "Method",
    "marginal_coverage": "Marg. cov.",
    "worst_discovered_slice_coverage": "Worst slice cov.",
    "worst_class_coverage": "Worst class cov.",
    "worst_known_group_coverage": "Worst known cov.",
    "avg_set_size": "Set size",
    "max_undercoverage_gap": "Max gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean CSV and LaTeX tables from DOCP summaries.")
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--outdir", type=Path, default=Path("results/latex_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_results = load_results(args.root)
    main_results = filter_methods(all_results, MAIN_METHODS)
    ablation_results = filter_methods(all_results, ABLATION_METHODS)

    write_table(all_results, args.outdir / "all_results")
    write_table(main_results, args.outdir / "main_results")
    write_table(ablation_results, args.outdir / "docp_ablation")
    write_sources(args.root, args.outdir / "sources.csv")
    print(f"wrote LaTeX tables to {args.outdir}")


def load_results(root: Path) -> pd.DataFrame:
    frames = []
    for dirname, dataset_name in DEFAULT_EXPERIMENTS.items():
        summary_path = root / dirname / "summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame["dataset"] = dataset_name
        frame["method"] = frame["method"].map(METHOD_LABELS).fillna(frame["method"])
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {root}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["method"].isin(METHOD_ORDER)].copy()
    combined["_dataset_order"] = combined["dataset"].map(_dataset_order())
    combined["_method_order"] = combined["method"].map(METHOD_ORDER)
    combined = combined.sort_values(["_dataset_order", "_method_order", "method"])
    combined = combined.drop(columns=["_dataset_order", "_method_order"])
    return combined


def filter_methods(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    filtered = frame[frame["method"].isin(methods)].copy()
    filtered["_method_order"] = filtered["method"].map(METHOD_ORDER)
    filtered["_dataset_order"] = filtered["dataset"].map(_dataset_order())
    filtered = filtered.sort_values(["_dataset_order", "_method_order"])
    return filtered.drop(columns=["_method_order", "_dataset_order"])


def write_table(frame: pd.DataFrame, stem: Path) -> None:
    table = format_table(frame)
    table.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.5f")
    stem.with_suffix(".tex").write_text(to_latex(table, caption=_caption(stem.name), label=f"tab:{stem.name}"), encoding="utf-8")


def write_sources(root: Path, path: Path) -> None:
    rows = []
    for dirname, dataset_name in DEFAULT_EXPERIMENTS.items():
        summary_path = root / dirname / "summary.csv"
        rows.append({"Dataset": dataset_name, "Source": str(summary_path), "Exists": summary_path.exists()})
    pd.DataFrame(rows).to_csv(path, index=False)


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in TABLE_COLUMNS if column in frame.columns]
    table = frame[columns].rename(columns=TABLE_COLUMNS)
    numeric = table.select_dtypes(include="number").columns
    table[numeric] = table[numeric].round(5)
    return table


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    columns = "ll" + "r" * (len(table.columns) - 2)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(escape_latex(str(column)) for column in table.columns) + r" \\",
        r"\midrule",
    ]
    current_dataset = None
    for _, row in table.iterrows():
        dataset = row.iloc[0]
        if current_dataset is not None and dataset != current_dataset:
            lines.append(r"\addlinespace")
        current_dataset = dataset
        values = [format_cell(value) for value in row]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def format_cell(value: object) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, float | np.floating):
        return f"{value:.5f}"
    return escape_latex(str(value))


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _dataset_order() -> dict[str, int]:
    return {name: index for index, name in enumerate(DEFAULT_EXPERIMENTS.values())}


def _caption(name: str) -> str:
    captions = {
        "all_results": "All cleaned experiment results.",
        "main_results": "Main experimental results.",
        "docp_ablation": "DOCP ablation results.",
    }
    return captions[name]


if __name__ == "__main__":
    main()

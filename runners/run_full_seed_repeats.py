from __future__ import annotations

import argparse
import fcntl
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO = Path('/home/ubuntu/zmh/DoCP')
PYTHON = Path('/home/ubuntu/anaconda3/envs/ucp/bin/python')

sys.path.insert(0, str(REPO))

from data.mimic_care import load_mimic_care_experiment  # noqa: E402
from experiments import images as image_experiment  # noqa: E402
from experiments import mimic_care  # noqa: E402
from reports.final_benchmark_report import DATASET_NAMES, SUMMARY_METRICS, _method_order  # noqa: E402
from runners.benchmark_tasks import MIMIC_CXR_TARGETS, NIH_TARGETS, expand_tasks, make_image_args, target_suffix  # noqa: E402

IMAGE_TASKS = {'bracs', 'ham10000', 'isic', 'rsna_5k'}
NIH_SUFFIX_TO_TARGET = {target_suffix(target): target for target in NIH_TARGETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run full-system repeated Consensus-CPScore benchmark.')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 21)))
    parser.add_argument('--outdir', type=Path, default=REPO / 'results/final_consensus_benchmark')
    parser.add_argument('--repeat-cache-root', type=Path, default=REPO / 'results/npz/repeats')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--keep-npz', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.repeat_cache_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_seed(seed, args)
    print('all requested seeds completed', flush=True)


class result_lock:
    def __init__(self, outdir: Path) -> None:
        self.path = outdir / '.seed_results.lock'
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('w')
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()



def run_seed(seed: int, args: argparse.Namespace) -> None:
    seed_name = f'seed_{seed:02d}'
    cache_root = args.repeat_cache_root / seed_name
    seed_outdir = args.outdir / 'full_seed_runs' / seed_name
    marker = seed_outdir / 'DONE'
    if marker.exists():
        print(f'[{seed_name}] already complete; skipping', flush=True)
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    seed_outdir.mkdir(parents=True, exist_ok=True)
    print(f'[{seed_name}] generating full-system caches in {cache_root}', flush=True)
    for task, family in expand_tasks(['bracs', 'nih_all', 'isic', 'ham10000', 'rsna_5k', 'mimic_care', 'mimic_cxr_all'], NIH_TARGETS):
        generate_cache(task, family, seed, cache_root, seed_outdir, args.gpu)
    print(f'[{seed_name}] evaluating final benchmark', flush=True)
    run_final_benchmark(seed, cache_root, seed_outdir / 'final')
    with result_lock(args.outdir):
        marker.write_text('done\n', encoding='utf-8')
        rebuild_seed_outputs(args.outdir)
        refresh_aggregate_outputs(args.outdir)
        write_known_subgroup_mean_std(args.outdir)
    if not args.keep_npz:
        shutil.rmtree(cache_root, ignore_errors=True)
        print(f'[{seed_name}] removed temporary npz cache', flush=True)


def generate_cache(task: str, family: str, seed: int, cache_root: Path, seed_outdir: Path, gpu: int) -> None:
    if family == 'mimic_care':
        generate_mimic_care_cache(seed, cache_root, seed_outdir)
        return
    generate_image_cache(task, family, seed, cache_root, seed_outdir, gpu)


def generate_image_cache(task: str, family: str, seed: int, cache_root: Path, seed_outdir: Path, gpu: int) -> None:
    preset = 'nih' if family == 'nih' else 'mimic_effusion' if family == 'mimic_cxr' else task
    task_args = make_image_args(preset, seed)
    if family == 'nih':
        suffix = task.removeprefix('nih_')
        task_args.nih_target = NIH_SUFFIX_TO_TARGET[suffix]
        task_args.cache_file = cache_root / f'{task}_strict.npz'
    elif family == 'mimic_cxr':
        suffix = task.removeprefix('mimic_cxr_')
        task_args.mimic_target = MIMIC_CXR_TARGETS[suffix]
        task_args.cache_file = cache_root / f'mimic_{suffix}.npz'
    elif family in IMAGE_TASKS:
        task_args.cache_file = cache_root / f'{task}.npz'
    else:
        raise ValueError(f'Unsupported image family: {family}')
    task_args.device = f'cuda:{gpu}'
    task_args.cuda = None
    task_args.epochs = 8
    task_args.lr = 1e-4
    task_args.batch_size = 64
    task_args.predict_batch_size = 128
    task_args.backbone_checkpoint = None
    task_args.cache = True
    task_args.outdir = seed_outdir / 'feature_generation' / task
    print(f'[{task_args.seed:02d}] feature cache {task}: {task_args.cache_file}', flush=True)
    data = image_experiment.load_experiment_data(task_args)
    device = image_experiment._resolve_device(task_args.device, task_args.cuda)
    probs, embeddings = image_experiment.load_or_create_features(task_args, data, device)
    print(f'[{task_args.seed:02d}] wrote {task}: probs={probs.shape}, embeddings={embeddings.shape}', flush=True)


def generate_mimic_care_cache(seed: int, cache_root: Path, seed_outdir: Path) -> None:
    args = SimpleNamespace(
        mimic_iv_root=Path('../dataset/mimic-iv-3.1'),
        mimic_ed_root=Path('../dataset/mimic-iv-ed-2.2'),
        feature_mode='text_triage',
        known_group='race_acuity',
        seed=seed,
        cache=True,
        cache_file=cache_root / 'mimic_care.npz',
        outdir=seed_outdir / 'feature_generation' / 'mimic_care',
    )
    print(f'[{seed:02d}] feature cache mimic_care: {args.cache_file}', flush=True)
    data = load_mimic_care_experiment(args.mimic_iv_root, args.mimic_ed_root, known_group=args.known_group, feature_mode=args.feature_mode, seed=seed)
    probs, embeddings = mimic_care._load_or_fit_features(args, data)
    print(f'[{seed:02d}] wrote mimic_care: probs={probs.shape}, embeddings={embeddings.shape}', flush=True)


def run_final_benchmark(seed: int, cache_root: Path, outdir: Path) -> None:
    command = [
        str(PYTHON), '-m', 'runners.run_final_consensus_benchmark',
        '--split-seed', str(seed),
        '--cache-root', str(cache_root.relative_to(REPO) if cache_root.is_relative_to(REPO) else cache_root),
        '--outdir', str(outdir.relative_to(REPO) if outdir.is_relative_to(REPO) else outdir),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def rebuild_seed_outputs(outdir: Path) -> None:
    result_frames = []
    runtime_frames = []
    for final_dir in sorted((outdir / 'full_seed_runs').glob('seed_*/final')):
        try:
            seed = int(final_dir.parent.name.removeprefix('seed_'))
        except ValueError:
            continue
        result_path = final_dir / 'all_results.csv'
        runtime_path = final_dir / 'all_runtimes.csv'
        if not result_path.exists() or not runtime_path.exists():
            continue
        results = pd.read_csv(result_path)
        results.insert(0, 'run_seed', seed)
        runtimes = pd.read_csv(runtime_path)
        runtimes.insert(0, 'run_seed', seed)
        result_frames.append(results)
        runtime_frames.append(runtimes)
    if result_frames:
        pd.concat(result_frames, ignore_index=True, sort=False).to_csv(outdir / 'all_results_by_seed.csv', index=False, float_format='%.5f')
    if runtime_frames:
        pd.concat(runtime_frames, ignore_index=True, sort=False).to_csv(outdir / 'all_runtimes_by_seed.csv', index=False, float_format='%.5f')



def append_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    frame.to_csv(path, mode='a', header=header, index=False, float_format='%.5f')


def write_known_subgroup_mean_std(outdir: Path) -> None:
    results_path = outdir / 'all_results_by_seed.csv'
    if not results_path.exists():
        return
    frame = pd.read_csv(results_path)
    metric_columns = [
        column
        for column in frame.select_dtypes(include='number').columns
        if column != 'run_seed'
    ]
    per_seed = frame.groupby(['run_seed', 'family', 'method'], as_index=False)[metric_columns].mean(numeric_only=True)
    summary = mean_std(per_seed, ['family', 'method'])
    summary.insert(0, 'dataset', summary['family'].map(DATASET_NAMES).fillna(summary['family']))
    summary.insert(2, 'method_base', summary['method'].map(method_base_name))
    summary.insert(3, 'known_subgroup', summary['method'].map(known_subgroup_name))
    summary.insert(4, 'num_seeds', int(frame['run_seed'].nunique()))
    summary = summary.drop(columns='family')
    summary['_method_order'] = summary['method'].map(_method_order)
    summary = summary.sort_values(['dataset', '_method_order', 'method']).drop(columns='_method_order').reset_index(drop=True)
    summary.to_csv(outdir / 'known_subgroup_mean_std.csv', index=False, float_format='%.5f')


def method_base_name(method: str) -> str:
    return 'Known-group CP' if method.startswith('Known-group CP') else method


def known_subgroup_name(method: str) -> str:
    if method.startswith('Known-group CP (') and method.endswith(')'):
        return method.removeprefix('Known-group CP (').removesuffix(')')
    return ''


def refresh_aggregate_outputs(outdir: Path) -> None:
    results_path = outdir / 'all_results_by_seed.csv'
    if not results_path.exists():
        return
    results = pd.read_csv(results_path)
    metrics = [column for column in SUMMARY_METRICS if column in results.columns]
    per_dataset_seed = results.groupby(['run_seed', 'family', 'method'], as_index=False)[metrics].mean(numeric_only=True)
    summary = mean_std(per_dataset_seed, ['family', 'method'])
    summary.insert(0, 'dataset', summary['family'].map(DATASET_NAMES).fillna(summary['family']))
    summary = summary.drop(columns='family')
    summary['_method_order'] = summary['method'].map(_method_order)
    summary = summary.sort_values(['dataset', '_method_order', 'method']).drop(columns='_method_order').reset_index(drop=True)
    summary.to_csv(outdir / 'main_results_mean_std.csv', index=False, float_format='%.5f')
    figure_source = summary[summary['method'].isin(['Standard CP', 'Classwise CP', 'Consensus-CPScore']) | summary['method'].str.startswith('Known-group CP')].copy()
    figure_source.to_csv(outdir / 'scope_figure_source.csv', index=False, float_format='%.5f')


def mean_std(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [column for column in frame.select_dtypes(include='number').columns if column != 'run_seed']
    mean = frame.groupby(group_columns, as_index=False)[metric_columns].mean(numeric_only=True)
    std = frame.groupby(group_columns, as_index=False)[metric_columns].std(numeric_only=True).fillna(0.0)
    mean = mean.rename(columns={column: f'{column}_mean' for column in metric_columns})
    std = std.rename(columns={column: f'{column}_std' for column in metric_columns})
    return mean.merge(std, on=group_columns, how='left')


def format_mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ['marginal_coverage', 'marginal_undercoverage', 'avg_set_size', 'worst_class_coverage', 'worst_known_group_coverage', 'max_undercoverage_gap']
    table = frame[['dataset', 'method']].copy()
    for metric in metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col not in frame:
            continue
        values = []
        for mean, std in zip(frame[mean_col], frame[std_col]):
            cell = '' if pd.isna(mean) else f'{mean:.4f} ± {std:.4f}'
            if metric == 'marginal_coverage' and not pd.isna(mean) and mean < 0.90:
                cell = f'\\textcolor{{red}}{{{cell}}}'
            values.append(cell)
        table[metric] = values
    return table


if __name__ == '__main__':
    main()

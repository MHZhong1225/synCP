# DOCP

DOCP builds a sparse coverage-driven pool of hidden undercoverage slices and
repairs them with BatchGCP recalibration. The main method is reported as
`DOCP`.

## Project Layout

```text
core/          conformal scores, metrics, and rectification
slices/        candidate slice discovery and coverage-driven pool selection
data/          synthetic and image dataset loaders
experiments/   synthetic and image experiment entry points
reports/       result formatting and report helpers
runners/       compact search utilities
```

Run experiments from the repository root:

```bash
python -m experiments.synthetic
python -m experiments.images
```

## Environment

CUDA is used by default when `--device cuda` is available.

## Synthetic Experiment

```bash
python -m experiments.synthetic --outdir results/synthetic --device cuda --seed 7
```

The public arguments are intentionally small:

| Argument | Meaning |
| --- | --- |
| `--outdir` | output directory |
| `--device` | `cuda` or `cpu` |
| `--cuda` | CUDA device id, for example `--cuda 0` |
| `--seed` | random seed |
| `--n` | synthetic sample count |
| `--alpha` | target miscoverage |
| `--components` | number of candidate components per generator |
| `--min-size` | minimum discovery-slice size |
| `--rectifier-min-size` | minimum recalibration-slice size |
| `--pool-groups` | maximum number of slices in the DOCP pool |

## Image Experiments

ImageFolder datasets are multi-class by default. BRACS is a 7-class task.
NIH is kept as a single-disease one-vs-rest task because its labels are
multi-label disease tags.

BRACS:

```bash
python -m experiments.images --dataset-root ../dataset/bracs --preset bracs --outdir results/bracs --device cuda:1
```

NIH Effusion:

```bash
python -m experiments.images --dataset-type nih --dataset-root ../dataset/NIH_ChestXrays --nih-target Effusion --known-group gender --max-samples 5000 --outdir results/nih --device cuda --cache
```

MIMIC-CXR joined with MIMIC-IV and MIMIC-IV-ED:

```bash
python -m experiments.images --preset mimic_effusion --outdir results/mimic_effusion --device cuda:0
python -m runners.run_mimic_targets --device cuda:0
```

The MIMIC experiments derive binary disease labels from the compressed CXR
reports, use patient-level splits, and evaluate known clinical groups joined
from MIMIC-IV and MIMIC-IV-ED. The all-target runner evaluates pleural
effusion, pneumonia, pneumothorax, pulmonary edema, and cardiomegaly.

RSNA Breast Cancer Detection:

```bash
python -m experiments.images --dataset-type rsna_5k --dataset-root ../dataset/rsna-breast --outdir results/rsna_5k --device cuda:1 --max-samples 5000
```

Image experiments write `summary.csv`. They also keep `features.npz` as a cache
so repeated runs can use `--cache`. Cache files are checked against the
current number of classes, so a binary cache is not reused for a multi-class
task by accident.

The image score is selected automatically: binary tasks use inverse probability
and multi-class tasks use APS prediction-set scores.

Image backbones can be selected with `--backbone`. The built-in choices are
`resnet18`, `resnet50`, `densenet121`, `chexnet`, `radimagenet-resnet50`, and
`radimagenet-densenet121`. CheXNet and RadImageNet require a local checkpoint:

```bash
python -m experiments.images --preset bracs --outdir results/bracs_chexnet --backbone chexnet --backbone-checkpoint checkpoints/chexnet.pth --device cuda
```

Feature caches are backbone-specific. Rerun without `--cache` when
switching backbones.

A compact backbone search template is provided at:

```text
configs/image_backbone_search.json
```

Run the search with:

```bash
python -m runners.image_search --config configs/image_backbone_search.json --outdir results/bracs_backbone_search --cuda 0
```

## Method Groups

Main-result baselines are methods that are not part of the DOCP candidate pool:

```text
Standard CP
Classwise CP
Known-group CP
Error-slice
K-means
```

Single-generator variants from the DOCP candidate pool are reported as
ablations, not external baselines:

```text
Random halfspace
GMM-unaware
Coverage-Domino
```

The main method is:

```text
DOCP
```

It uses the coverage-driven candidate pool and BatchGCP recalibration.

## Multi-Seed Image Runs

Image multi-seed runs use the same image entry point and keep only the
averaged result:

```bash
python -m experiments.images --preset bracs --cache --seeds 1 2 3 4 5 --outdir results/bracs --device cuda
```

To distribute seeds over two GPUs:

```bash
python -m experiments.images --preset bracs --cache --seeds 1 2 3 4 5 --outdir results/bracs --cuda 0 --cuda 1
```

The final file is:

```text
results/bracs/summary.csv
```

Per-seed directories are temporary and are deleted after aggregation.

## Output

Every CSV and terminal table rounds numeric values to five decimal places.
Rows are ordered with baselines first, ablations next, and our method `DOCP` as
the final row.

`worst_discovered_slice_coverage` is evaluated on the shared union of discovered
slices, so methods are compared on the same slice collection.

The main `DOCP` row uses a sparse fixed-size pool selected from candidate
generators on the discovery split only. Each generator slice is kept in its
original form and also split by predicted class before pool selection, so the
pool can target class-specific undercoverage rather than only global embedding
regions. By default, DOCP keeps up to eight undercoverage slices with a
round-robin pass over GMM, K-means, Coverage-Domino, and random-halfspace
candidates after enrichment scoring and redundancy removal.
The pool is then repaired on the independent recalibration split; test data are
not used for method selection.

The main `DOCP` rectifier is BatchGCP over the selected pool. The incremental
Pool-MVP repair is reported as `DOCP-MVP`.

Single-run output:

```text
summary.csv
```

Multi-seed output:

```text
summary.csv
```

No `all_runs.csv`, per-seed summaries, or slice-quality CSVs are written by
default.

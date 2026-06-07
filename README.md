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

MIMIC ED-to-ICU text task:

```bash
python -m experiments.mimic_care --outdir results/mimic_care --cache
python -m experiments.mimic_care --feature-mode text --outdir results/mimic_care_text --cache
```

This patient-level task predicts whether first ICU admission occurs within
6 hours, between 6 and 24 hours, or after 24 hours of ED arrival. The primary
model uses chief complaint plus arrival-time triage signals. Demographic
attributes are excluded from model features and used only for subgroup
evaluation. The task measures ED-to-ICU timing patterns, not causal care
quality or medical negligence. CXR content is not used because the available
local CXR index does not provide a reliable study timestamp for proving that
an image was acquired before first ICU admission.

All ISIC:

```bash
python -m experiments.images --preset isic --outdir results/isic --device cuda:0
```

The local release contains 81,722 metadata rows. The experiment keeps the
53,311 labeled dermoscopic images, then retains the 12 original diagnoses with
at least 300 images: acrochordon, actinic keratosis, basal cell carcinoma,
dermatofibroma, lichenoid keratosis, melanoma, nevus, pigmented benign
keratosis, seborrheic keratosis, solar lentigo, squamous cell carcinoma, and
vascular lesion. These classes contain 52,402 eligible images; class-aware
sampling limits the primary experiment to 20,000.

Images are read from `image_256sq.hdf5`, resized to 224 x 224, and normalized
with ImageNet statistics. Splits are stratified at the patient level when a
patient ID exists, otherwise at the lesion level, and finally at the image
level. The train/calibration/discovery/rectification/test fractions are
50/15/15/10/10. The default known group is the contributing source, enabling
evaluation of cross-source undercoverage.

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

The formal full benchmark does not report `GMM-unaware` as a standalone
baseline. The original DOCP candidate pool is kept unchanged.

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

## Full Medical Benchmark

Run BRACS, 14 NIH one-vs-rest tasks, ISIC, HAM10000, RSNA, and MIMIC Care:

```bash
conda run -n ucp python -m runners.run_full_benchmark
```

The full benchmark compares Standard CP, Classwise CP, Known-group CP, DOCP,
EE, and consensus variants. It uses 10 Coverage-Domino discovery seeds for
the stability variants and does not include a standalone GMM discovery
baseline.

Method runtimes exclude cached feature extraction and classifier training.
They include discovery, aggregation, rectification, and unified slice
auditing/evaluation. Formal outputs and the Chinese analysis are written to:

```text
results/full_benchmark/
results/full_benchmark/chinese_analysis.md
```

Formal result tables keep only the core coverage, efficiency, subgroup, task,
and selected-slice metrics. Known-group coverage is expanded into one column
per observed subgroup, such as `known_group_coverage_gender_group_0`.
Consensus methods are always placed at the end of result tables.

Discovery-source ablation for Consensus-Hard:

```bash
conda run -n ucp python -m runners.ablate_consensus_discovery \
  --tasks bracs nih_all isic ham10000 rsna_5k mimic_care \
  --nih-targets Effusion
```

This compares Coverage-Domino, direct Undercoverage-Risk discovery, K-means,
and Random-Halfspace while keeping consensus aggregation and rectification
fixed. Results are written to `results/consensus_discovery_ablation/`.

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

## Discovery Seed Stability

To isolate instability caused only by a discovery algorithm's random
initialization, keep the cached model outputs and data split fixed while
rerunning one discovery method with multiple seeds:

```bash
python -m runners.analyze_seed_stability \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --method coverage_domino \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 7 \
  --outdir results/seed_stability/coverage_domino_fixed_split
```

This diagnostic does not retrain the classifier. It fixes the data split,
labels, cached probabilities, embeddings, base conformal threshold, and
rectification split. It reports:

- the number of discovered slices for each seed;
- membership Jaccard of the union of discovered slices on a fixed test anchor;
- bidirectional best-match Jaccard between individual slices;
- per-seed rectified coverage and average set size;
- variability of worst-slice coverage after rectification.

The fixed-split experiment is intentionally separate from full multi-seed
training experiments. It measures discovery-algorithm instability rather than
mixing classifier, sampling, calibration, and discovery uncertainty.

## Isolated Variant 1: EE-Style Slice Aggregation

The EE-style variant aggregates repeated Coverage-aware discovery runs without
modifying the main DOCP implementation:

```bash
python -m variants.ee_style.run \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 7 \
  --jaccard-threshold 0.5 \
  --frequency-threshold 0.5 \
  --outdir results/variants/ee_style
```

Candidate slices are matched by membership Jaccard on the independent
rectification anchor features. `EE-Frequency` ranks candidates by repeated
selection frequency. `EE-Score` additionally uses the median discovery
undercoverage gap and slice support. Both variants use BatchGCP after
aggregation and are compared against Standard CP, Classwise CP, Known-group CP,
and the original DOCP pool.

The main result uses a pre-specified majority-frequency threshold of `0.5`.
Threshold sensitivity experiments should be written to separate result
directories rather than replacing the main result.

## Isolated Variant 2: r-value-style Rank Stability

The rank-stability variant ranks Coverage-aware slices within each discovery
run by positive undercoverage gap times square-root support:

```bash
python -m variants.rank_stability.run \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 7 \
  --top-k 3 \
  --jaccard-threshold 0.5 \
  --r-value-threshold 0.5 \
  --borda-threshold 0.5 \
  --outdir results/variants/rank_stability
```

`Rank-rValue` retains slices that repeatedly appear among the top problematic
slices. `Rank-Borda` aggregates normalized ranks and penalizes unmatched runs
with the worst rank. Both methods use anchor-set Jaccard matching and BatchGCP
rectification, without modifying the original DOCP implementation.

The pre-specified main result uses `top-k=3`. Parameter sensitivity runs must
use separate output directories; for example:

```bash
python -m variants.rank_stability.run \
  --datasets rsna_5k \
  --top-k 5 \
  --outdir results/variants/rank_stability_rsna_top5
```

## Isolated Variant 3: Empirical Bayes Latent Undercoverage

The empirical-Bayes variant aligns repeated Coverage-aware discoveries on an
anchor set, estimates a latent signed undercoverage gap, and selects slices by
posterior probability:

```bash
python -m variants.empirical_bayes.run \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 7 \
  --jaccard-threshold 0.5 \
  --posterior-threshold 0.9 \
  --prior-source all_candidates \
  --outdir results/variants/empirical_bayes
```

The main implementation estimates the empirical prior from all support-valid
pre-filter Coverage-aware GMM candidates. Estimating the prior only from slices
that already passed the positive undercoverage-gap gate causes severe
selection-induced overconfidence and is retained only as a sensitivity result.

Conservative prior sensitivity:

```bash
python -m variants.empirical_bayes.run \
  --prior-source zero \
  --outdir results/variants/empirical_bayes_zero_prior
```

## Isolated Variant 4: Consensus Latent Slice Discovery

The consensus variant clusters Coverage-Domino slices from repeated seeds in
anchor-membership space and constructs new latent consensus groups:

```bash
python -m variants.consensus_latent.run \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 7 \
  --similarity-threshold 0.3 \
  --member-threshold 0.5 \
  --min-within-stability 0.3 \
  --posterior-threshold 0.5 \
  --outdir results/variants/consensus_latent
```

Candidate nodes are clustered using average-linkage anchor Jaccard distance.
Within each latent cluster, candidates from the same seed cast one union vote,
then votes are averaged across represented seeds. `Consensus-Hard` applies
BatchGCP to majority-vote consensus groups. `Consensus-Soft` interpolates the
global and group thresholds using continuous consensus membership.

The current isolated experiment intentionally uses only Coverage-Domino.
Cross-method diversity is not evaluated and contributes no reliability bonus.

## Complex Hidden Subgroup Benchmark

Natural real datasets do not guarantee that a strong unknown undercoverage
subgroup exists. The controlled benchmark constructs nonlinear XOR, annulus,
and disjoint-union hidden groups in each real-data embedding space, then
applies a score shift without exposing hidden membership to discovery:

```bash
python -m runners.benchmark_complex_subgroups \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --shift-delta 0.12 \
  --outdir results/complex_subgroups
```

Coverage is evaluated as a target rather than a quantity to maximize. Reports
include absolute marginal target error, undercoverage, overcoverage, worst
hidden-subgroup coverage, hidden-group deficit, and average set size.

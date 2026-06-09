# SCOPE

SCOPE studies hidden undercoverage in medical prediction sets. The current
implementation uses CP-score-aware latent slice discovery, consensus aggregation
on an independent anchor split, and evidence-gated BatchGCP calibration. The
main reported method is `Consensus-CPScore`.

## Project Layout

```text
core/          conformal scores, metrics, consensus aggregation, and rectification
slices/        candidate latent-slice discovery
data/          synthetic and image dataset loaders
experiments/   synthetic and image experiment entry points
reports/       result formatting and report helpers
runners/       maintained batch runners and benchmark helpers
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
vascular lesion. These diagnoses are selected before removing HAM10000 overlap. The default ISIC
preprocessing then removes every image whose `isic_id` appears in the local
HAM10000 metadata, so ISIC and HAM10000 are disjoint in main experiments.
Class-aware sampling limits the primary ISIC experiment to 20,000 images.

Images are read from `image_256sq.hdf5`, resized to 224 x 224, and normalized
with ImageNet statistics. Splits are stratified at the patient level when a
patient ID exists, otherwise at the lesion level, and finally at the image
level. The train/calibration/discovery/rectification/test fractions are
50/15/15/10/10. The default known group is `anatom_site_general`, and evaluation also tracks
available sensitive attributes such as sex and approximate age.

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

## Method Groups

Main-result baselines:

```text
Standard CP
Classwise CP
Known-group CP
```

Main SCOPE method:

```text
Consensus-CPScore
```

`Consensus-CPScore` keeps cached model predictions fixed, proposes latent
regions using embeddings augmented by CP score structure, aggregates repeated
discoveries on the independent anchor/rectification split, and applies BatchGCP
without using test labels for method selection. The consensus aggregation core
lives in `core/consensus.py`; CP-score-aware candidate discovery lives in
`slices/discovery.py`.

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
conda run -n ucp python -m runners.run_final_consensus_benchmark
```

The final benchmark compares Standard CP, Classwise CP, Known-group CP, and
Consensus-CPScore. Consensus-CPScore uses ten discovery seeds by default.

Method runtimes exclude cached feature extraction and classifier training.
They include discovery, consensus aggregation, calibration, and evaluation.
Outputs are written to:

```text
results/final_consensus_benchmark/
```

Formal result tables keep only the core coverage, efficiency, subgroup, task,
and selected-slice metrics. Known-group coverage is expanded into one column
per observed subgroup, such as `known_group_coverage_gender_group_0`.
Consensus methods are always placed after the baselines in result tables.

MIMIC-CXR consensus benchmark with MIMIC-IV and MIMIC-IV-ED groups:

```bash
conda run -n ucp python -m runners.run_mimic_cxr_consensus
```

This evaluates five CXR report-label targets against race, ED acuity, and ICU
history groups. Results are written to `results/mimic_cxr_consensus/`.

Discovery-source ablation:

```bash
conda run -n ucp python -m runners.ablate_consensus_discovery \
  --tasks bracs nih_all isic ham10000 rsna_5k mimic_care \
  --nih-targets Effusion
```

This compares CP-Score-KMeans, Coverage-Domino, direct Undercoverage-Risk,
K-means, and Random-Halfspace while keeping consensus aggregation and
rectification fixed. Results are written to `results/consensus_discovery_ablation/`.

Seed-count ablation:

```bash
conda run -n ucp python -m runners.ablate_consensus_seed_count \
  --tasks bracs nih_all isic ham10000 rsna_5k mimic_care
```

This compares how many discovery seeds are needed for Consensus-CPScore.
Results are written to `results/consensus_seed_count_ablation/`.

## Runner Reference

The maintained files in `runners/` are:

| File | Role |
| --- | --- |
| `run_final_consensus_benchmark.py` | Main all-dataset SCOPE benchmark. |
| `run_mimic_cxr_consensus.py` | MIMIC-CXR benchmark over report targets and MIMIC-IV/ED groups. |
| `run_nih_ovr.py` | Builds NIH one-vs-rest caches/results for all selected disease labels. |
| `run_mimic_targets.py` | Builds MIMIC-CXR target caches/results for the five report-label tasks. |
| `ablate_consensus_discovery.py` | Compares discovery sources under the same consensus and BatchGCP pipeline. |
| `ablate_consensus_seed_count.py` | Compares one to twenty discovery seeds for the final discovery sources. |
| `analyze_seed_stability.py` | Diagnostic for discovery instability with fixed cached features and splits. |
| `benchmark_complex_subgroups.py` | Controlled stress test for Consensus-CPScore with synthetic hidden groups in real embedding spaces. |
| `benchmark_tasks.py` | Shared task loading and result-column helpers; not a command-line experiment. |

## Output

Every CSV and terminal table rounds numeric values to five decimal places.
Rows are ordered with baselines first and consensus methods after the baselines.

The main benchmark reports marginal coverage, undercoverage, average set size,
worst class coverage, worst known-group coverage, maximum undercoverage gap,
selected-slice count, and method runtimes.

Consensus methods discover latent regions on the discovery split, align
candidate memberships on the independent anchor split, keep only evidence-backed
consensus groups, and calibrate the selected groups with BatchGCP on the
rectification split. Test data are used only for final evaluation.

Single-run output:

```text
summary.csv
```

Multi-seed output:

```text
summary.csv
```

Temporary per-seed directories are removed by the runners that create them.
Final benchmark outputs keep aggregate CSVs, per-task CSVs, runtime CSVs, and
compact config JSON files. The exploratory `variants/` package has been removed; maintained method code now lives in `core/`, `slices/`, and `runners/`.

## Discovery Seed Stability

To isolate instability caused only by a discovery algorithm's random
initialization, keep the cached model outputs and data split fixed while
rerunning one discovery method with multiple seeds:

```bash
python -m runners.analyze_seed_stability \
  --datasets ham10000 nih mimic_effusion isic rsna_5k \
  --method kmeans \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 7 \
  --outdir results/seed_stability/kmeans_fixed_split
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

# synCP

synCP studies hidden undercoverage in medical prediction sets. The repository
implements `Consensus-CPScore`, a conformal-prediction workflow that discovers
latent subgroups with unreliable coverage, aggregates candidate slices, and
applies evidence-gated group calibration.

## Repository Layout

```text
core/          conformal scores, coverage metrics, consensus aggregation, calibration
slices/        latent subgroup discovery utilities
data/          dataset loaders and feature/prediction cache helpers
experiments/   entry points for synthetic, image, and MIMIC ED-to-ICU experiments
runners/       benchmark runners that operate on cached predictions/features
```

Generated outputs are written under `results/` and are not part of the source
release.

## Environment

The code is Python-based and was developed with a conda environment. A recent
Python 3.10+ environment with PyTorch is recommended.

```bash
conda create -n syncp python=3.11
conda activate syncp
```

Install the key packages used by the main code:

```bash
pip install numpy pandas scipy scikit-learn torch torchvision pillow tqdm
```

Additional packages are needed for some datasets or optional comparisons:

- `matplotlib` and `seaborn`: plotting and figure scripts.
- `pydicom`: RSNA breast DICOM loading.
- `h5py`: HDF5 image archives.
- `statsmodels` and `torchmetrics`: external baseline adapters.
- `pytest`: local test runs.

For GPU runs, install the PyTorch build that matches the local CUDA driver.

## Data

Datasets are not included in this repository. The loaders expect local copies of
the corresponding datasets and metadata files. The supported experiment families
include synthetic data, HAM10000, ISIC, NIH ChestXray14, RSNA Breast Cancer
Detection, MIMIC-CXR report-label tasks, and MIMIC-IV ED-to-ICU.

Use each dataset only under its original license and access policy. Point the
commands below to the local dataset roots on your machine or server.

## Quick Start

Run the synthetic sanity check:

```bash
python -m experiments.synthetic --outdir results/synthetic --device cuda
```

Run an ISIC image experiment:

```bash
python -m experiments.images \
  --dataset-type isic \
  --dataset-root /path/to/ISIC \
  --known-group anatom_site_general \
  --outdir results/isic \
  --device cuda:0 \
  --cache
```

Run an NIH ChestXray14 one-vs-rest task:

```bash
python -m experiments.images \
  --dataset-type nih \
  --dataset-root /path/to/NIH_ChestXrays \
  --nih-target Effusion \
  --known-group gender \
  --outdir results/nih_effusion \
  --device cuda:0 \
  --cache
```

Run the MIMIC ED-to-ICU experiment:

```bash
python -m experiments.mimic_care \
  --mimic-iv-root /path/to/mimic-iv \
  --mimic-ed-root /path/to/mimic-iv-ed \
  --outdir results/mimic_care \
  --cache
```

Feature and prediction caches are stored under `results/npz/` when caching is
enabled.

## Benchmark From Caches

After feature/prediction caches have been generated, run the main benchmark with:

```bash
python -m runners.run_final_consensus_benchmark \
  --cache-root results/npz \
  --outdir results/final_consensus_benchmark
```

The benchmark writes CSV summaries with marginal coverage, average set size,
worst-class coverage, worst known-group coverage, known-group coverage gaps, and
maximum undercoverage gap.

## Notes For Readers

The README is intended as a practical entry point for running the released code.
It intentionally omits the internal hyperparameter sweeps and tuning details used
during development. For paper-level interpretation, use the manuscript text and
the generated result tables together.

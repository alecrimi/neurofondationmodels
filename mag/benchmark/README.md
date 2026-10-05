# EEG-TSFM Benchmark

Diploma workshop project — AGH Kraków, 2026. Supervisor: A. Crimi.

Benchmarks 8 Time Series Foundation Models (TSFMs) on EEG signals from the
publicly available BIDS dataset **ds004504** (Alzheimer / FTD / Healthy cohort).

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Prerequisites](#prerequisites)
5. [One-time Setup](#one-time-setup)
6. [Running the Benchmark](#running-the-benchmark)
7. [Evaluating Results](#evaluating-results)
8. [Results](#results)
9. [Architecture](#architecture)
10. [Caveats and Known Issues](#caveats-and-known-issues)

---

## Overview

Each model runs in a **fully isolated Python virtual environment** to avoid
dependency conflicts (incompatible `gluonts`, `tokenizers`, `numpy`, and
`transformers` versions across models). EEG data is loaded once by the
orchestrator and passed to each model subprocess via a temporary pickle file.

Two data pipelines are supported:

| Pipeline | Input space | Channels |
|---|---|---|
| `baseline` | Scalp EEG (z-scored) | Fp1, Fp2, P3, P4 |
| `loreta` | sLORETA source parcels (fsaverage standard brain) | 6 cortical regions × 2 hemispheres |

```
benchmark/run.py  ──load EEG──▶  pipelines/baseline.py  (or loreta.py)
      │
      ├──pickle──▶  models/chronos/runner.py     (venv: chronos)
      ├──pickle──▶  models/chronos2/runner.py    (venv: chronos2)
      ├──pickle──▶  models/timesfm/runner.py     (venv: timesfm)
      ├──pickle──▶  models/moirai/runner.py      (venv: moirai)
      ├──pickle──▶  models/lag_llama/runner.py   (venv: lag_llama)
      ├──pickle──▶  models/sundial/runner.py     (venv: sundial)
      ├──pickle──▶  models/vitime/runner.py      (venv: vitime)
      └──pickle──▶  models/timefound/runner.py   (venv: timefound)
```

---

## Dataset

**ds004504** — EEG recordings, Alzheimer's disease and frontotemporal dementia.

- OpenNeuro: https://openneuro.org/datasets/ds004504
- Format: BIDS, preprocessed `.set` files (EEGLAB) in `derivatives/`
- Window: context = 512 samples, horizon = 64 samples, 5 windows per signal
- Preprocessing: z-score normalisation only (bandpass already applied in derivatives)

Download and place at:
```
mag/datasets/ds004504/
```

---

## Models

| Model | Checkpoint / Package | Paper | Notes |
|-------|----------------------|-------|-------|
| **Chronos** | `amazon/chronos-t5-small` (HuggingFace) | [Ansari et al., 2024 — arXiv 2403.07815](https://arxiv.org/abs/2403.07815) | Probabilistic, 20 sample paths |
| **Chronos-2 (Bolt)** | `amazon/chronos-bolt-base` (HuggingFace) | [Ansari et al., 2024 — arXiv 2403.07815](https://arxiv.org/abs/2403.07815) | Quantile model, p50 taken as point forecast |
| **TimesFM 2.5** | `google/timesfm-2.5-200m-pytorch` (HuggingFace) | [Das et al., 2024 — arXiv 2310.10688](https://arxiv.org/abs/2310.10688) | Point forecast |
| **Moirai 1.0** | `Salesforce/moirai-1.0-R-base` (HuggingFace) | [Liu et al., 2024 — arXiv 2402.02592](https://arxiv.org/abs/2402.02592) | Probabilistic, 20 sample paths |
| **Lag-Llama** | `time-series-foundation-models/Lag-Llama` (HuggingFace) | [Rasul et al., 2023 — arXiv 2310.08278](https://arxiv.org/abs/2310.08278) | Needs `reference/lag-llama/` repo on sys.path |
| **Sundial** | `thuml/sundial-base-128m` (HuggingFace) | [Liu et al., 2025](https://github.com/thuml/Sundial) | Probabilistic, 20 sample paths; `trust_remote_code=True` |
| **ViTime** | `reference/vitime/` repo + manual checkpoint | [Li et al., 2024 — arXiv 2408.03239](https://arxiv.org/abs/2408.03239) | Probabilistic, 20 sample paths; checkpoint download required |
| **TimeFound** | Custom reimplementation — **no public code** | [Liu et al., 2025 — arXiv 2503.04118](https://arxiv.org/abs/2503.04118) | Random-init proxy architecture from paper; Baidu Research |

### Reference repositories (cloned to `ref/`, read-only)

```
reference/chronos-forecasting/   https://github.com/amazon-science/chronos-forecasting
reference/timesfm/               https://github.com/google-research/timesfm
reference/uni2ts/                https://github.com/SalesforceAIResearch/uni2ts
reference/lag-llama/             https://github.com/time-series-foundation-models/lag-llama
reference/sundial/               https://github.com/thuml/Sundial
reference/vitime/                https://github.com/IkeYang/ViTime
```

> **reference/ is read-only.** The benchmark code never modifies these directories.
> Lag-Llama and ViTime runners add their respective `ref/` subdirectory to
> `sys.path` at runtime to import the creator's code directly.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11 – 3.13 |
| CUDA | 12.4 |
| GPU | RTX 4060 (8 GB VRAM) or better |
| Rust + Cargo | for building `tokenizers` if needed |
| Git | for cloning reference repos |
| ~25 GB free disk | for venvs + model weights |

### VRAM estimates (RTX 4060 = 8 GB)

Models run one at a time, so the table is per-model peak usage:

| Model | Approx VRAM | Notes |
|-------|-------------|-------|
| Chronos (small) | ~300 MB | Very safe |
| Chronos-2 (base) | ~700 MB | Safe |
| Lag-Llama | ~300 MB | Safe |
| Sundial | ~500 MB | Safe |
| Moirai (base) | ~600 MB | Safe |
| TimesFM 200M | ~800 MB | Safe |
| TimeFound | <100 MB | Tiny custom Transformer |
| ViTime | ~500 MB | Safe |

All models fit comfortably within 8 GB. No quantization needed.

---

## One-time Setup

### 1 — Clone the reference repositories

```bash
# From the project root (mag/)
mkdir -p ref

git clone https://github.com/amazon-science/chronos-forecasting   ref/chronos-forecasting
git clone https://github.com/google-research/timesfm               ref/timesfm
git clone https://github.com/SalesforceAIResearch/uni2ts           ref/uni2ts
git clone https://github.com/time-series-foundation-models/lag-llama ref/lag-llama
git clone https://github.com/thuml/Sundial                         ref/sundial
git clone https://github.com/IkeYang/ViTime                        ref/vitime
```

### 2 — Download the ViTime checkpoint (manual)

ViTime weights are not on HuggingFace — distributed via Google Drive.

1. Open: https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view
2. Download `ViTime_Model.pth`
3. Place at: `mag/benchmark/models/vitime/ViTime_Model.pth`

### 3 — Create venv-cuda (orchestrator environment)

venv-cuda is used by `benchmark/run.py` to load EEG data (MNE, nibabel, etc.).

```bash
# From mag/
python -m venv venv-cuda

# Install PyTorch CUDA first
venv-cuda\Scripts\pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install everything else
venv-cuda\Scripts\pip install -r benchmark/requirements-cuda.txt

# uni2ts must be installed without deps (it pins numpy<2.0 which conflicts)
venv-cuda\Scripts\pip install uni2ts==2.0.0 --no-deps
```

### 4 — Create and populate all model venvs

This is the slow step (~25 GB download; torch alone is ~2 GB per venv).

```bash
python benchmark/run.py --setup-only
```

To force-rebuild all venvs from scratch:

```bash
python benchmark/run.py --setup-only --force-setup
```

To rebuild a single model's venv:

```bash
python benchmark/run.py --models Sundial --setup-only --force-setup
```

---

## Running the Benchmark

All commands run from `mag/`.

```bash
# Quick smoke test — 3 subjects, all 8 models, baseline pipeline
python benchmark/run.py --n 3

# Full run — all 88 subjects
python benchmark/run.py

# Specific models only
python benchmark/run.py --models Chronos TimesFM Moirai --n 10

# Force re-run, ignore cached predictions
python benchmark/run.py --refresh

# sLORETA source-space pipeline
python benchmark/run.py --pipeline loreta --n 3

# CPU only (slow)
python benchmark/run.py --device cpu --n 3

# Skip venv setup (assume already done)
python benchmark/run.py --no-setup --n 3
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--models` | Space-separated list of models to run. Default: all 8. |
| `--n N` | Limit to first N subjects. Default: all subjects. |
| `--device` | `cuda` or `cpu`. Auto-detected if omitted. |
| `--pipeline` | `baseline` (scalp EEG) or `loreta` (sLORETA source space). Default: baseline. |
| `--refresh` | Ignore cached `output.pkl` and re-run predictions. |
| `--setup-only` | Only create/update venvs, skip running models. |
| `--no-setup` | Skip venv setup entirely, go straight to predictions. |
| `--force-setup` | Delete and recreate venvs even if already set up. |

---

## Evaluating Results

```bash
# Summarise baseline results + generate plots
python benchmark/evaluate.py

# Summarise LORETA results
python benchmark/evaluate.py --pipeline loreta

# Side-by-side baseline vs LORETA comparison
python benchmark/evaluate.py --compare

# Use normalised MSE instead of physical MSE
python benchmark/evaluate.py --metric mse_norm
```

### evaluate.py output

| Output | Location |
|--------|----------|
| Per-pipeline summary report | `benchmark/results/<pipeline>/summary_report.md` |
| Comparison report | `benchmark/results/comparison/comparison_report.md` |
| Overall MSE chart | `benchmark/results/<pipeline>/figures/plot_1_overall.png` |
| By clinical group chart | `benchmark/results/<pipeline>/figures/plot_2_groups.png` |
| By channel/parcel chart | `benchmark/results/<pipeline>/figures/plot_3_channels.png` |

---

## Results

Results are written to `benchmark/results/<pipeline>/<model>/`:

```
benchmark/results/
├── baseline/
│   ├── chronos/
│   │   ├── output.pkl              ← cached predictions, reused on reruns
│   │   └── chronos_metrics.csv     ← per-subject MSE (normalised + physical)
│   ├── ... (same for all 8 models)
│   ├── figures/                    ← generated by evaluate.py
│   └── summary_report.md
├── loreta/                         ← same structure, sLORETA source channels
└── comparison/
    ├── comparison_report.md
    └── figures/
```

### Metrics columns

| Column | Description |
|--------|-------------|
| `subject` | Subject ID (e.g. `sub-001`) |
| `group` | Diagnosis group (`A` = Alzheimer, `F` = FTD, `C` = Control) |
| `channel` | EEG channel or source parcel |
| `mse_norm` | MSE on z-scored signal (dimensionless; < 1.0 = better than naive predictor) |
| `mse_phys` | MSE in physical units (V² for baseline, (A/m²)² for LORETA) |

> **Important**: `mse_phys` values are NOT comparable between baseline and LORETA
> because the physical units differ (V² vs (A/m²)²). Use `mse_norm` for
> cross-pipeline comparison.

### Preliminary results (3 subjects, 4 channels, baseline pipeline)

| Model | MSE_norm | Notes |
|-------|----------|-------|
| Chronos | ~0.38 | Best overall |
| Chronos-2 | ~0.39 | |
| TimesFM | ~0.40 | |
| Moirai | ~0.41 | |
| Sundial | ~0.42 | |
| Lag-Llama | ~0.55 | GluonTS compat patches applied |
| ViTime | ~0.72 | Manual checkpoint required |
| TimeFound | ~0.85 | Random-init proxy, no pretrained weights |

**Group pattern**: Control (C) < Alzheimer (A) ≈ FTD (F) in both pipelines.

### Comparison with Michał's baseline (88 subjects, no covariates)

Michał's run used a different pipeline (no z-score normalization, physical MSE in V²).
To compare, multiply `mse_norm` by the subject's `raw_std²` to get physical MSE.

| Model | Michał MSE (V²) | Notes |
|-------|----------------|-------|
| Chronos-2 | 4.27e-10 | Best |
| TimeGPT | 4.61e-10 | Cloud API, not in this benchmark |
| Chronos | 4.74e-10 | |
| Sundial | 4.75e-10 | |
| Lag-Llama | 1.15e-09 | |
| Moirai | 2.31e-08 | |
| ViTime | 5.23e-08 | |
| TimesFM | 9.25e-07 | Degrades on AD (alpha-band slowing) |

---

## Architecture

```
mag/
├── datasets/ds004504/           EEG dataset (BIDS format)
├── ref/                         Read-only creator repos
│   ├── chronos-forecasting/
│   ├── timesfm/
│   ├── uni2ts/
│   ├── lag-llama/
│   ├── sundial/
│   └── vitime/
├── venv-cuda/                   Orchestrator venv (MNE, nibabel, PyTorch CUDA)
└── benchmark/
    ├── run.py                   ← MAIN ENTRY POINT
    ├── evaluate.py              ← RESULT ANALYSIS + PLOTS
    ├── requirements-cuda.txt    ← full pinned packages for venv-cuda
    ├── pipelines/
    │   ├── baseline.py          scalp EEG: z-score + window extraction
    │   └── loreta.py            sLORETA: MNE inverse + fsaverage parcels
    ├── models/
    │   ├── chronos/
    │   │   ├── runner.py        uses ChronosPipeline.from_pretrained()
    │   │   ├── requirements.txt
    │   │   └── venv/            isolated venv (created by run.py --setup-only)
    │   ├── chronos2/            uses BaseChronosPipeline (Bolt quantiles)
    │   ├── timesfm/             uses TimesFM_2p5_200M_torch.from_pretrained()
    │   ├── moirai/              uses MoiraiModule + MoiraiForecast (uni2ts)
    │   ├── lag_llama/           uses LagLlamaEstimator from reference/lag-llama/
    │   ├── sundial/             uses model.generate() via trust_remote_code
    │   ├── vitime/              uses ViTimePrediction from reference/vitime/
    │   └── timefound/           custom Transformer proxy (paper arXiv 2503.04118)
    └── results/
        ├── baseline/            per-model CSVs + output.pkl caches
        ├── loreta/              same, sLORETA source channels
        ├── comparison/          baseline vs LORETA comparison report + figures
        └── cache/
            └── loreta_inv_op.pkl   sLORETA inverse operator cache (expensive)
```

---

## Caveats and Known Issues

### TimeFound — no public code or weights

The TimeFound paper (arXiv 2503.04118, Baidu Research) has no associated
public repository or model weights. The runner in `benchmark/models/timefound/runner.py`
is a faithful proxy reimplementation of the encoder-decoder Transformer
architecture described in the paper, with **random initialisation**.
Results from this model reflect architecture quality only, not pretrained
knowledge. This is clearly disclosed in the runner's docstring.

### ViTime — manual checkpoint download required

ViTime's authors distribute weights only via Google Drive (not HuggingFace).
The benchmark will fail for ViTime with a clear error message if
`benchmark/models/vitime/ViTime_Model.pth` is missing.

### Sundial — Python 3.13 + transformers pin

The Sundial README pins `transformers==4.40.1`, which does not build on
Python 3.13 (tokenizers uses pyo3 0.21.2, max Python 3.12). We use
`transformers>=4.46.0` instead, which ships Python 3.13 wheels. The model
is loaded via `trust_remote_code=True` and its `generate()` API is used as
specified by the creator; a `_greedy_search` fallback is present if the
HuggingFace model card has been updated to a different interface.

### Lag-Llama — GluonTS compatibility shims

Lag-Llama's checkpoint was trained with an older GluonTS API. The runner
applies two runtime patches:
- `pandas 2.2+` frequency alias compatibility (`H→h`, `M→ME`, etc.)
- `torch.load(..., weights_only=False)` to allow non-tensor objects in the checkpoint
- `gluonts.torch.modules.loss` stub (removed in gluonts>=0.15.0, training-only)

**Important**: only the `.loss` submodule is stubbed. Stubbing the parent
`gluonts.torch.modules` package would break gluonts's own submodule loading.

### MSE units: baseline vs LORETA

`mse_phys` values differ by ~10¹¹ between pipelines — this is **not a bug**.
Baseline signals are in Volts (~30 µV std), LORETA signals are in A/m²
(~7-18 A/m² std). The dimensionless `mse_norm` is the correct metric for
any cross-pipeline comparison.

### Broken `.git/` stubs in `ref/` (Windows)

If reference repos were cloned in a session that failed midway, empty `.git/`
directories may remain. These do not affect the benchmark but will confuse git.
Delete them manually from Windows Explorer if present.

# 🧠 Brain Time Series Forecasting with Foundation Models 

---

## 🚀 Overview

This repository explores the use of **time series foundation models (TSFMs)** for predicting neural signals (e.g., EEG), with a focus on **Alzheimer’s disease analysis**.

We investigate whether **pretrained, zero-shot models** can capture neural dynamics and generalize across subjects and conditions.

---

## 🧩 Time Series Foundation Models

Below is a curated list of currently available TSFMs with links to their repositories or official pages:

### 🔹 Core Models

- **TimesFM (Google Research)**  
  https://github.com/google-research/timesfm  

- **Chronos (Amazon)**  
  https://github.com/amazon-science/chronos-forecasting  

- **Moirai (Salesforce)**  
  https://github.com/SalesforceAIResearch/uni2ts

- **Lag-Llama (ServiceNow)**  
  https://github.com/time-series-foundation-models/lag-llama  

- **TimeGPT (Nixtla)**  
  https://github.com/Nixtla/nixtla  

- **NeuroGPT**  

---

### 🔹 Emerging / Research Models

- **TimeFound**  
  https://github.com/microsoft/TimeFound  

- **Sundial**  
  https://github.com/microsoft/sundial  

- **ViTime**  
  https://github.com/ViTime/ViTime  

---

## 🧠 Dataset: Alzheimer EEG (OpenNeuro)

- 👉 https://openneuro.org/datasets/ds004504/versions/1.0.8  
There is a raw version and a preprocessed version inside "derivatives"


### Description
- EEG recordings from Alzheimer’s patients and healthy controls  
- BIDS-compliant  
- Suitable for:
  - time series forecasting  
  - connectivity analysis  
  - causal inference  

---

## ⚙️ Pipeline Overview

This project contains two independent pipeline implementations for benchmarking TSFMs on EEG data. Both target the same dataset, channels, and evaluation protocol, but differ in implementation strategy and scope.

---

### Michał's Baseline Pipeline (`Michal/`)

The reference implementation uses **actual pretrained TSFM weights** loaded via each model's official library. The key parameters:

| Parameter | Value |
|---|---|
| Dataset | ds004504 (88 subjects) |
| Channels | Fp1, Fp2, P3, P4 |
| Sampling rate | 500 Hz |
| Context length | 512 samples (~1.02 s) |
| Horizon length | 64 samples (~0.13 s) |
| Windows per subject | 5, evenly spaced via `np.linspace` |
| MSE unit | Physical Volts² |

Probabilistic models (Chronos, Chronos-2, Moirai, Lag-Llama, Sundial, ViTime) generate 20 sample paths; the median is taken as the point forecast.

A key extension in Michał's pipeline is the use of **covariates**: electrode signals are paired (P3+Fp1, P4+Fp2) and the paired signal is fed alongside the target during forecasting. This produced the best results overall (Chronos2-cov, TimeGPT-cov).

**Results summary (MSE, Volts²):**

| Model | Overall Mean MSE |
|---|---|
| Chronos2-cov | 2.07e-10 |
| TimeGPT-cov | 2.19e-10 |
| Chronos2 | 4.27e-10 |
| TimeGPT | 4.61e-10 |
| Chronos | 4.74e-10 |
| Sundial | 4.75e-10 |
| Lag-Llama | 1.15e-09 |
| Moirai | 2.31e-08 |
| ViTime | 5.23e-08 |
| TimesFM | 9.25e-07 |

Full per-group and per-electrode breakdowns are in `Michal/TSFMs_baseile_summary_report.md`.

---

### New Modular Pipeline (`new/`)

A fully refactored, component-based reimplementation of the evaluation pipeline. The high-level architecture:

```
EEGPipeline.__call__(subject_id)
   ├── 1. DataLoader          — loads BIDS .set files via MNE
   ├── 2. DataPreprocessor    — extracts raw channel voltages
   ├── 3. QualityChecker      — rejects flatline / saturated channels
   ├── 4. DataPreprocessor    — Butterworth 4th-order bandpass (0.5–40 Hz),
   │                            notch (50 Hz), global z-score
   └── 5. WindowExtractor     — 5 non-overlapping windows (512 ctx + 64 horizon)
```

**Differences from Michał's baseline:**

| Aspect | Michał's pipeline | New pipeline |
|---|---|---|
| Model weights | Actual pretrained (official libs) | Custom PyTorch implementations, random init |
| Covariate support | Yes (paired electrodes) | No |
| Signal quality filtering | Not explicit | QualityChecker: flatline (std < 1e-8 V) and saturation (std > 1e-3 V) rejection |
| Preprocessing | Not described in baseline report | Butterworth bandpass + notch + z-score |
| Probabilistic batching | Sequential (slow) | Batched 20-path tensor ops (~20× speedup) |
| MSE scaling | Physical Volts² | Physical Volts² (rescaled from z-score using raw signal std) |
| Output format | Baseline report + plots | CSVs (352 patient-electrode rows + 15 summary rows) matching baseline format |

**Implemented models** (all 9 from literature): Chronos, Chronos-2, TimesFM, Moirai, Lag-Llama, TimeGPT, Sundial, ViTime, TimeFound.

> **Note:** Because the new pipeline uses randomly initialized model architectures rather than pretrained weights, its raw MSE numbers are not directly comparable to Michał's baseline. The pipeline is designed for structural validation and methodology alignment, not pretrained model benchmarking.

**Running the new pipeline:**

```bash
# Quick smoke test (3 subjects, ~1-2 seconds)
python new/evaluate.py --subjects 3 --device cpu

# Full benchmark (88 subjects)
python new/evaluate.py --device cpu

# Generate comparison report and plots
python new/present_results.py
```

Results are written to `results/benchmark/single_signal/`.

---

### Legacy Pipeline (`old/`)

The original pipeline investigating the effect of z-score normalization and sLORETA source localization on TSFM forecasting. Archived for reproducibility. See `old/README.md` for details.

```bash
# Integration smoke test
python old/smoke_test.py --subjects 1 --no-loreta

# Full 3-phase benchmark (raw z-score, raw no-norm, sLORETA)
python old/runner.py --subjects 3 --workers 4
```

Results are written to `results/benchmark/raw_eeg` and `results/benchmark/loreta`.

---

## 📁 Repository Structure

```
neurofondationmodels/
├── Michal/
│   └── TSFMs_baseile_summary_report.md   # Baseline results (pretrained models)
├── new/
│   ├── components/
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── quality_checker.py
│   │   └── window_extractor.py
│   ├── eeg_pipeline.py       # High-level pipeline controller
│   ├── tsfm_models.py        # PyTorch model implementations + ModelRegistry
│   ├── evaluate.py           # Main benchmark entrypoint
│   └── present_results.py    # Report + figure generation
└── old/
    ├── runner.py             # 3-phase benchmark (z-score / no-norm / sLORETA)
    ├── localization_runner.py
    ├── metrics.py
    └── visualizer.py
```

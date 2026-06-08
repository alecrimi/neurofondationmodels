# TSFM Benchmark - Baseline vs LORETA Comparison

## Parameters
- **Baseline**: scalp EEG (Fp1, Fp2, P3, P4)
- **LORETA**: sLORETA source parcels, 6 cortical regions x 2 hemispheres (fsaverage)
- **Metric**: `mse_norm` (Mean MSE (normalised))

---

## Table 1 - Overall Comparison

| Model     |   Baseline Mean MSE (normalised) |   LORETA Mean MSE (normalised) | Delta% (LORETA vs Baseline)   |
|:----------|---------------------------------:|-------------------------------:|:------------------------------|
| Chronos   |                          0.37689 |                        0.54408 | +44.4%                        |
| Chronos-2 |                          0.37226 |                        0.5542  | +48.9%                        |
| Lag-Llama |                          1.8224  |                        1.0409  | -42.9%                        |
| Moirai    |                          0.39375 |                        0.58397 | +48.3%                        |
| Sundial   |                          0.41883 |                        0.54147 | +29.3%                        |
| TimeFound |                          1.1027  |                        0.88405 | -19.8%                        |
| TimesFM   |                          0.39704 |                        0.51267 | +29.1%                        |
| ViTime    |                          1.0078  |                        0.83646 | -17.0%                        |

> Positive Delta% = LORETA worse (higher MSE); negative = LORETA better.

![Overall](figures/plot_1_overall.png)

---

## Table 2 - Performance by Clinical Group

![Groups](figures/plot_2_groups.png)

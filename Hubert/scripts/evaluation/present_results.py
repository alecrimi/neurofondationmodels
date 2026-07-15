"""
Generate the benchmark summary report and figures.

Reads per-model CSVs produced by evaluation/metrics.py and outputs:
  - Figures/plot_1_overall.png
  - Figures/plot_2_groups.png
  - Figures/plot_3_electrodes.png
  - TSFMs_baseline_summary_report.md

Usage (from repo root):
    python new/evaluation/present_results.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results", "benchmark", "single_signal")
    figures_dir = os.path.join(results_dir, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  COMPILING TSFM BENCHMARK REPORT AND VISUALS")
    print(f"{'='*50}")
    print(f"Results dir: {results_dir}")

    models = ["Chronos", "Chronos-2", "TimesFM", "Moirai", "Lag-Llama",
              "TimeGPT", "Sundial", "ViTime", "TimeFound"]

    overall_performance = []
    group_performance = []
    electrode_performance = []

    for model_name in models:
        model_filename = f"{model_name.lower().replace('-', '')}_eeg_macro_eval_results.csv"
        csv_path = os.path.join(results_dir, model_filename)
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV not found: {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        df_patient = df[df['record_type'] == 'per_patient_electrode']
        if df_patient.empty:
            print(f"[WARNING] Patient data empty in {csv_path}. Skipping.")
            continue

        overall_performance.append({
            'Model': model_name,
            'Overall Mean MSE': df_patient['mse'].mean()
        })
        group_means = df_patient.groupby('group')['mse'].mean().to_dict()
        group_performance.append({
            'Model': model_name,
            'A (Alzheimer)': group_means.get('A', 0.0),
            'C (Control)':   group_means.get('C', 0.0),
            'F (FTD)':       group_means.get('F', 0.0),
            'Average':       df_patient['mse'].mean()
        })
        ch_means = df_patient.groupby('electrode')['mse'].mean().to_dict()
        electrode_performance.append({
            'Model': model_name,
            'Fp1': ch_means.get('Fp1', 0.0),
            'Fp2': ch_means.get('Fp2', 0.0),
            'P3':  ch_means.get('P3',  0.0),
            'P4':  ch_means.get('P4',  0.0),
            'Average': df_patient['mse'].mean()
        })

    if not overall_performance:
        print("[ERROR] No result CSVs processed. Exiting.")
        return

    df_overall   = pd.DataFrame(overall_performance).sort_values('Overall Mean MSE')
    df_group     = pd.DataFrame(group_performance).sort_values('Average')
    df_electrode = pd.DataFrame(electrode_performance).sort_values('Average')

    def to_sci(val): return f"{val:.5e}"

    # ── Plot 1: Overall ───────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_overall)))
    bars = plt.barh(df_overall['Model'], df_overall['Overall Mean MSE'],
                    color=colors, edgecolor='grey')
    plt.xscale('log')
    plt.xlabel('Overall Mean MSE (Volts²) — Log Scale', fontsize=12, fontweight='bold')
    plt.title('EEG resting-state TSFM: Overall Performance', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        plt.text(w*1.1, bar.get_y()+bar.get_height()/2,
                 f"{w:.2e}", va='center', ha='left', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "plot_1_overall.png"), dpi=300)
    plt.close()

    # ── Plot 2: Groups ────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    x, width = np.arange(len(df_group)), 0.25
    plt.bar(x-width, df_group['A (Alzheimer)'], width, label='A (Alzheimer)', color='#e056fd', edgecolor='black')
    plt.bar(x,       df_group['C (Control)'],   width, label='C (Control)',   color='#22a6b3', edgecolor='black')
    plt.bar(x+width, df_group['F (FTD)'],        width, label='F (FTD)',       color='#f0932b', edgecolor='black')
    plt.yscale('log')
    plt.xticks(x, df_group['Model'], rotation=15, fontsize=10, fontweight='bold')
    plt.ylabel('Mean MSE (Volts²) — Log Scale', fontsize=12, fontweight='bold')
    plt.title('Performance by Clinical Patient Group', fontsize=14, fontweight='bold', pad=15)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, shadow=True)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "plot_2_groups.png"), dpi=300)
    plt.close()

    # ── Plot 3: Electrodes ────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    x, width = np.arange(len(df_electrode)), 0.18
    plt.bar(x-1.5*width, df_electrode['Fp1'], width, label='Fp1', color='#4834d4')
    plt.bar(x-0.5*width, df_electrode['Fp2'], width, label='Fp2', color='#686de0')
    plt.bar(x+0.5*width, df_electrode['P3'],  width, label='P3',  color='#badc58')
    plt.bar(x+1.5*width, df_electrode['P4'],  width, label='P4',  color='#6ab04c')
    plt.yscale('log')
    plt.xticks(x, df_electrode['Model'], rotation=15, fontsize=10, fontweight='bold')
    plt.ylabel('Mean MSE (Volts²) — Log Scale', fontsize=12, fontweight='bold')
    plt.title('Performance by EEG Electrode', fontsize=14, fontweight='bold', pad=15)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, shadow=True)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "plot_3_electrodes.png"), dpi=300)
    plt.close()

    # ── Report ────────────────────────────────────────────────────────────────
    def fmt(df, cols):
        d = df.copy()
        for c in cols: d[c] = d[c].apply(to_sci)
        return d.to_markdown(index=False)

    report_path = os.path.join(results_dir, "TSFMs_baseline_summary_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Foundation Models Baseline (Literature-based, without Covariates)

## Input Parameters
* **Dataset**: ds004504 Alzheimer resting-state EEG (88 subjects)
* **Channels**: Fp1, Fp2, P3, P4
* **FS**: 500 Hz  |  **Context**: 512 (~1.02 s)  |  **Horizon**: 64 (~0.13 s)
* **Windows**: 5 non-overlapping, `np.linspace(0, max_start, 5)`
* **Normalisation**: z-score (derivatives already bandpass/notch filtered)
* **Metric**: MSE in Volts² (mean over 5 windows)

---

## Table 1: Overall Performance

{fmt(df_overall, ['Overall Mean MSE'])}

![Overall](Figures/plot_1_overall.png)

---

## Table 2: Performance by Patient Group

{fmt(df_group, ['A (Alzheimer)', 'C (Control)', 'F (FTD)', 'Average'])}

![Groups](Figures/plot_2_groups.png)

---

## Table 3: Performance by Electrode

{fmt(df_electrode, ['Fp1', 'Fp2', 'P3', 'P4', 'Average'])}

![Electrodes](Figures/plot_3_electrodes.png)
""")
    print(f"  -> Report: {report_path}")
    print(f"{'='*50}")
    print(f"  DONE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

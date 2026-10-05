"""
Universal visualisation — works for any pipeline × model combination.

Reads the metrics CSV produced by metrics.py and generates 3 PNG files:
  1. overall      — mean physical MSE per model, sorted low → high
  2. by_group     — mean physical MSE broken down by clinical group (A / C / F)
  3. by_electrode — mean physical MSE broken down by electrode (Fp1/Fp2/P3/P4)

All plots are saved to:
    results/figures/<pipeline>_<models>_<type>.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import pandas as pd

GROUPS = ["A", "C", "F"]
GROUP_LABELS = {"A": "Alzheimer", "C": "Healthy", "F": "FTD"}
CHANNELS = ["Fp1", "Fp2", "P3", "P4"]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# Gap fraction between bar groups (0 = touching, 1 = no bars)
_GROUP_GAP = 0.25


def _bar_chart(
    ax: plt.Axes,
    categories: list[str],
    values_per_model: dict[str, list[float]],
    ylabel: str,
    title: str,
    sort_by_value: bool = False,
) -> None:
    """
    Grouped bar chart with spacing between groups.

    sort_by_value: if True and there is exactly one value per model (i.e. the
                   overall chart), reorder categories low → high by the first
                   (and only) model's value, then sort remaining models to match.
    """
    n_models = len(values_per_model)
    n_cats = len(categories)

    model_names = list(values_per_model.keys())
    vals_matrix = np.array([values_per_model[m] for m in model_names],
                           dtype=float)  # (n_models, n_cats)

    # ── sort categories by mean value across models (low → high) ──────────
    if sort_by_value:
        mean_per_cat = np.nanmean(vals_matrix, axis=0)
        order = np.argsort(mean_per_cat)
        categories = [categories[i] for i in order]
        vals_matrix = vals_matrix[:, order]

    x = np.arange(n_cats)
    # bar width leaves a visible gap between groups
    width = (1.0 - _GROUP_GAP) / max(n_models, 1)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width

    colors = plt.cm.tab10.colors

    for i, model_name in enumerate(model_names):
        ax.bar(
            x + offsets[i],
            vals_matrix[i],
            width,
            label=model_name,
            color=colors[i % len(colors)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    if n_models > 1:
        ax.legend(fontsize=9, frameon=False)


def generate_plots(
    csv_paths: dict[str, str | Path],
    pipeline_name: str,
    out_dir: str | Path,
) -> None:
    """
    Parameters
    ----------
    csv_paths     : {model_name: path_to_metrics_csv}
    pipeline_name : used in file names and titles
    out_dir       : directory where PNGs are saved
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load CSVs ─────────────────────────────────────────────────────────
    dfs: dict[str, pd.DataFrame] = {}
    for model_name, csv_path in csv_paths.items():
        p = Path(csv_path)
        if not p.exists():
            print(f"  [!] CSV not found for {model_name}: {p}. Skipping.")
            continue
        dfs[model_name] = pd.read_csv(p)

    if not dfs:
        print("  [!] No data to visualise.")
        return

    tag = "_".join(csv_paths.keys()).replace(" ", "")

    # ── 1. overall MSE — one bar per model, sorted low → high ────────────
    model_mse = {}
    for model_name, df in dfs.items():
        per_pat = df[df["record_type"] == "per_patient_electrode"]
        model_mse[model_name] = per_pat["mse_phys"].mean()

    sorted_models = sorted(model_mse, key=model_mse.get)
    sorted_vals   = [model_mse[m] for m in sorted_models]
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(dfs)), 4))
    ax.bar(
        range(len(sorted_models)), sorted_vals,
        width=0.6,
        color=[colors[i % len(colors)] for i in range(len(sorted_models))],
    )
    ax.set_xticks(range(len(sorted_models)))
    ax.set_xticklabels(sorted_models, rotation=15, ha="right")
    ax.set_ylabel("Mean MSE (V²)")
    ax.set_title(f"{pipeline_name} — Overall physical MSE (low → high)")
    fig.tight_layout()
    out_path = out_dir / f"{pipeline_name}_{tag}_overall.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Figure: {out_path}")

    # ── 2. by clinical group ───────────────────────────────────────────────
    group_vals: dict[str, list[float]] = {}
    for model_name, df in dfs.items():
        per_pat = df[df["record_type"] == "per_patient_electrode"]
        vals = []
        for g in GROUPS:
            sub = per_pat[per_pat["group"] == g]
            vals.append(sub["mse_phys"].mean() if not sub.empty else float("nan"))
        group_vals[model_name] = vals

    labels = [GROUP_LABELS.get(g, g) for g in GROUPS]
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_chart(
        ax, labels, group_vals,
        ylabel="Mean MSE (V²)",
        title=f"{pipeline_name} — MSE by clinical group",
        sort_by_value=False,   # keep clinical order: Alzheimer / Healthy / FTD
    )
    fig.tight_layout()
    out_path = out_dir / f"{pipeline_name}_{tag}_by_group.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Figure: {out_path}")

    # ── 3. by electrode ───────────────────────────────────────────────────
    elec_vals: dict[str, list[float]] = {}
    for model_name, df in dfs.items():
        per_pat = df[df["record_type"] == "per_patient_electrode"]
        vals = []
        for ch in CHANNELS:
            sub = per_pat[per_pat["electrode"] == ch]
            vals.append(sub["mse_phys"].mean() if not sub.empty else float("nan"))
        elec_vals[model_name] = vals

    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_chart(
        ax, CHANNELS, elec_vals,
        ylabel="Mean MSE (V²)",
        title=f"{pipeline_name} — MSE by electrode",
        sort_by_value=False,
    )
    fig.tight_layout()
    out_path = out_dir / f"{pipeline_name}_{tag}_by_electrode.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Figure: {out_path}")

"""
Summarise benchmark results.

Usage (from repo root):
    python benchmark/evaluate.py                          # baseline pipeline
    python benchmark/evaluate.py --pipeline loreta        # LORETA pipeline
    python benchmark/evaluate.py --pipeline loreta_gsp    # LORETA-GSP (network harmonics)
    python benchmark/evaluate.py --compare                # all-pipeline comparison
    python benchmark/evaluate.py --metric mse_phys        # physical MSE (V²)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})

_HERE = Path(__file__).resolve().parent

_MODEL_CSV = {
    "Chronos":   ("chronos",   "chronos_metrics.csv"),
    "Chronos-2": ("chronos2",  "chronos_2_metrics.csv"),
    "TimesFM":   ("timesfm",   "timesfm_metrics.csv"),
    "Moirai":    ("moirai",    "moirai_metrics.csv"),
    "Lag-Llama": ("lag_llama", "lag_llama_metrics.csv"),
    "Sundial":   ("sundial",   "sundial_metrics.csv"),
    "ViTime":    ("vitime",    "vitime_metrics.csv"),
    "TimeFound": ("timefound", "timefound_metrics.csv"),
}

GROUPS    = ["A", "C", "F"]
GRP_LABEL = {"A": "Alzheimer", "C": "Control", "F": "FTD"}

_COLORS = {
    "Chronos": "#4834d4", "Chronos-2": "#686de0", "TimesFM": "#22a6b3",
    "Moirai": "#30336b", "Lag-Llama": "#f0932b", "Sundial": "#e056fd",
    "ViTime": "#6ab04c", "TimeFound": "#eb4d4b",
}
_GRP_COL = {"A": "#e056fd", "C": "#22a6b3", "F": "#f0932b"}
_ELC_COL = {"Fp1": "#4834d4", "Fp2": "#686de0", "P3": "#badc58", "P4": "#6ab04c"}

_PIPE_LABEL = {
    "baseline":   "Baseline",
    "loreta":     "LORETA",
    "loreta_gsp": "LORETA-GSP",
}
_PIPE_HATCH = {"baseline": "", "loreta": "//", "loreta_gsp": "xx"}
_PIPE_ALPHA = {"baseline": 0.9, "loreta": 0.65, "loreta_gsp": 0.45}


def _to_sci(v):
    return f"{v:.4e}"


def load_data(metric, results_dir):
    dfs = {}
    for model, (folder, fname) in _MODEL_CSV.items():
        p = Path(results_dir) / folder / fname
        if not p.exists():
            print(f"  [skip] {model}: not found")
            continue
        df = pd.read_csv(p)
        if metric not in df.columns:
            print(f"  [skip] {model}: column '{metric}' missing")
            continue
        dfs[model] = df
    return dfs


def get_channels(dfs):
    chans = set()
    for df in dfs.values():
        chans.update(df["channel"].dropna().unique())
    return sorted(chans)


# ── single-pipeline plots ─────────────────────────────────────────────────────

def plot_overall(dfs, metric, ylabel):
    items = sorted(dfs.items(), key=lambda kv: kv[1][metric].mean())
    names = [k for k, _ in items]
    vals  = [df[metric].mean() for _, df in items]
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(names)), 5))
    bars = ax.barh(names, vals,
                   color=[_COLORS.get(n, "#888") for n in names],
                   edgecolor="grey", height=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(f"{ylabel} — Log Scale", fontsize=12, fontweight="bold")
    ax.set_title("EEG resting-state TSFM — Overall Performance",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.12, bar.get_y() + bar.get_height() / 2,
                f"{w:.2e}", va="center", ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_by_group(dfs, metric, ylabel):
    model_names = sorted(dfs, key=lambda m: dfs[m][metric].mean())
    x = np.arange(len(model_names))
    offsets = [-0.25, 0, 0.25]
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(model_names)), 5))
    for i, g in enumerate(GROUPS):
        vals = []
        for m in model_names:
            sub = dfs[m][dfs[m]["group"] == g]
            vals.append(sub[metric].mean() if not sub.empty else float("nan"))
        ax.bar(x + offsets[i], vals, 0.25, label=GRP_LABEL[g],
               color=_GRP_COL[g], edgecolor="black", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right",
                       fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{ylabel} — Log Scale", fontsize=12, fontweight="bold")
    ax.set_title("Performance by Clinical Group", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_by_channel(dfs, metric, ylabel, channels):
    model_names = sorted(dfs, key=lambda m: dfs[m][metric].mean())
    n_ch    = len(channels)
    x       = np.arange(len(model_names))
    width   = min(0.18, 0.8 / max(n_ch, 1))
    offsets = (np.arange(n_ch) - (n_ch - 1) / 2) * width
    palette = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(model_names)), 5))
    for i, ch in enumerate(channels):
        color = _ELC_COL.get(ch, palette[i % len(palette)])
        vals = []
        for m in model_names:
            sub = dfs[m][dfs[m]["channel"] == ch]
            vals.append(sub[metric].mean() if not sub.empty else float("nan"))
        ax.bar(x + offsets[i], vals, width, label=ch,
               color=color, edgecolor="black", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right",
                       fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{ylabel} — Log Scale", fontsize=12, fontweight="bold")
    ax.set_title("Performance by Channel / Parcel", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8, frameon=True, facecolor="white",
              ncol=max(1, n_ch // 6))
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def build_report(dfs, metric, ylabel, channels, pipeline):
    def sci(v):
        return _to_sci(v) if not (isinstance(v, float) and np.isnan(v)) else "-"

    df_ov = pd.DataFrame(
        [{"Model": m, ylabel: df[metric].mean()} for m, df in dfs.items()]
    ).sort_values(ylabel)
    df_ov[ylabel] = df_ov[ylabel].apply(sci)

    rows_grp = []
    for m, df in dfs.items():
        row = {"Model": m}
        for g in GROUPS:
            sub = df[df["group"] == g]
            row[GRP_LABEL[g]] = sci(sub[metric].mean() if not sub.empty else float("nan"))
        row["Average"] = sci(df[metric].mean())
        rows_grp.append(row)
    df_grp = pd.DataFrame(rows_grp)

    rows_ch = []
    for m, df in dfs.items():
        row = {"Model": m}
        for ch in channels:
            sub = df[df["channel"] == ch]
            row[ch] = sci(sub[metric].mean() if not sub.empty else float("nan"))
        row["Average"] = sci(df[metric].mean())
        rows_ch.append(row)
    df_ch = pd.DataFrame(rows_ch)

    pipeline_note = {
        "baseline":   "Scalp EEG — Fp1, Fp2, P3, P4",
        "loreta":     "sLORETA source parcels — 6 cortical regions x 2 hemispheres (fsaverage)",
        "loreta_gsp": "LORETA-GSP — sLORETA parcels projected onto network harmonics (Euclidean connectome)",
    }.get(pipeline, pipeline)
    ch_label = "Electrode" if pipeline == "baseline" else (
        "Network Harmonic" if pipeline == "loreta_gsp" else "Source Parcel"
    )

    lines = [
        f"# TSFM Benchmark - {pipeline.capitalize()} Pipeline Results\n",
        f"## Parameters",
        f"- **Dataset**: ds004504 (Alzheimer resting-state EEG)",
        f"- **Pipeline**: {pipeline_note}",
        f"- **Context**: 512 samples  |  **Horizon**: 64 samples",
        f"- **Metric**: `{metric}` ({ylabel})\n",
        "---\n",
        "## Table 1 - Overall Performance\n",
        df_ov.to_markdown(index=False),
        "\n![Overall](figures/plot_1_overall.png)\n",
        "---\n",
        "## Table 2 - Performance by Clinical Group\n",
        df_grp.to_markdown(index=False),
        "\n![Groups](figures/plot_2_groups.png)\n",
        "---\n",
        f"## Table 3 - Performance by {ch_label}\n",
        df_ch.to_markdown(index=False),
        "\n![Channels](figures/plot_3_channels.png)\n",
    ]
    return "\n".join(lines)


# ── comparison plots ──────────────────────────────────────────────────────────
# pipes_dfs: dict {pipeline_name: {model_name: DataFrame}}

def _reference_pipeline(pipes_dfs):
    """Return the name of the first pipeline present (used for sort order)."""
    for name in ["baseline", "loreta", "loreta_gsp"]:
        if name in pipes_dfs:
            return name
    return next(iter(pipes_dfs))


def plot_comparison_overall(pipes_dfs, metric, ylabel):
    ref = _reference_pipeline(pipes_dfs)
    all_models = sorted(
        set(m for dfs in pipes_dfs.values() for m in dfs),
        key=lambda m: pipes_dfs[ref][m][metric].mean() if m in pipes_dfs[ref] else float("inf"),
    )
    pipe_names = list(pipes_dfs)
    n_pipes = len(pipe_names)
    w = 0.7 / n_pipes
    x = np.arange(len(all_models))
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(all_models)), 5))
    for pi, pname in enumerate(pipe_names):
        dfs = pipes_dfs[pname]
        offset = (pi - (n_pipes - 1) / 2) * w
        vals = [dfs[m][metric].mean() if m in dfs else float("nan") for m in all_models]
        colors = [_COLORS.get(m, "#888") for m in all_models]
        for xi, (v, c) in enumerate(zip(vals, colors)):
            ax.bar(x[xi] + offset, v, w, color=c,
                   alpha=_PIPE_ALPHA.get(pname, 0.7),
                   hatch=_PIPE_HATCH.get(pname, ""),
                   edgecolor="black",
                   label=_PIPE_LABEL.get(pname, pname) if xi == 0 else "")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{ylabel} — Log Scale", fontsize=12, fontweight="bold")
    pipe_str = " vs ".join(_PIPE_LABEL.get(p, p) for p in pipe_names)
    ax.set_title(f"{pipe_str} — Overall Performance", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_comparison_groups(pipes_dfs, metric, ylabel):
    ref = _reference_pipeline(pipes_dfs)
    all_models = sorted(
        set(m for dfs in pipes_dfs.values() for m in dfs),
        key=lambda m: pipes_dfs[ref][m][metric].mean() if m in pipes_dfs[ref] else float("inf"),
    )
    pipe_names = list(pipes_dfs)
    n_pipes, n_grp = len(pipe_names), len(GROUPS)
    w = 0.8 / (n_pipes * n_grp)
    x = np.arange(len(all_models))
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(all_models)), 5))
    slot = 0
    for gi, g in enumerate(GROUPS):
        for pi, pname in enumerate(pipe_names):
            dfs = pipes_dfs[pname]
            offset = (slot - (n_pipes * n_grp - 1) / 2) * w
            vals = [dfs[m][dfs[m]["group"] == g][metric].mean()
                    if m in dfs else float("nan") for m in all_models]
            pipe_lbl = _PIPE_LABEL.get(pname, pname)
            ax.bar(x + offset, vals, w,
                   color=_GRP_COL[g], edgecolor="black",
                   alpha=_PIPE_ALPHA.get(pname, 0.7),
                   hatch=_PIPE_HATCH.get(pname, ""),
                   label=f"{GRP_LABEL[g]} {pipe_lbl}")
            slot += 1
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{ylabel} — Log Scale", fontsize=12, fontweight="bold")
    pipe_str = " vs ".join(_PIPE_LABEL.get(p, p) for p in pipe_names)
    ax.set_title(f"{pipe_str} — Performance by Clinical Group",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8, frameon=True, ncol=n_pipes, facecolor="white")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def build_comparison_report(pipes_dfs, metric, ylabel):
    all_models = sorted(set(m for dfs in pipes_dfs.values() for m in dfs))
    ref = _reference_pipeline(pipes_dfs)
    pipe_names = list(pipes_dfs)

    rows = []
    for m in all_models:
        row = {"Model": m}
        v_ref = None
        for pname in pipe_names:
            dfs = pipes_dfs[pname]
            v = dfs[m][metric].mean() if m in dfs else float("nan")
            lbl = _PIPE_LABEL.get(pname, pname)
            row[f"{lbl} {ylabel}"] = _to_sci(v) if not np.isnan(v) else "-"
            if pname == ref:
                v_ref = v
        # Delta columns vs reference
        for pname in pipe_names:
            if pname == ref:
                continue
            dfs = pipes_dfs[pname]
            v = dfs[m][metric].mean() if m in dfs else float("nan")
            lbl = _PIPE_LABEL.get(pname, pname)
            ref_lbl = _PIPE_LABEL.get(ref, ref)
            if v_ref is not None and not (np.isnan(v) or np.isnan(v_ref)) and v_ref != 0:
                row[f"Δ% {lbl} vs {ref_lbl}"] = f"{(v - v_ref) / v_ref * 100:+.1f}%"
            else:
                row[f"Δ% {lbl} vs {ref_lbl}"] = "-"
        rows.append(row)

    table = pd.DataFrame(rows).to_markdown(index=False)
    pipe_notes = {
        "baseline":   "scalp EEG (Fp1, Fp2, P3, P4)",
        "loreta":     "sLORETA source parcels, 6 cortical regions × 2 hemispheres (fsaverage)",
        "loreta_gsp": "LORETA-GSP — sLORETA parcels projected onto network harmonics",
    }
    lines = [
        f"# TSFM Benchmark — Pipeline Comparison\n",
        "## Parameters",
    ]
    for pname in pipe_names:
        lbl = _PIPE_LABEL.get(pname, pname)
        lines.append(f"- **{lbl}**: {pipe_notes.get(pname, pname)}")
    lines += [
        f"- **Metric**: `{metric}` ({ylabel})\n",
        "---\n",
        "## Table 1 — Overall Comparison\n",
        table,
        "\n> Positive Δ% = higher MSE (worse); negative = lower MSE (better).\n",
        "![Overall](figures/plot_1_overall.png)\n",
        "---\n",
        "## Table 2 — Performance by Clinical Group\n",
        "![Groups](figures/plot_2_groups.png)\n",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric",   default="mse_norm",
                        choices=["mse_phys", "mse_norm"])
    parser.add_argument("--pipeline", default="baseline",
                        choices=["baseline", "loreta", "loreta_gsp"])
    parser.add_argument("--compare",  action="store_true",
                        help="Compare all pipelines that have results")
    args = parser.parse_args()

    metric = args.metric
    ylabel = "Mean MSE (V²)" if metric == "mse_phys" else "Mean MSE (normalised)"

    if args.compare:
        all_pipes = ["baseline", "loreta", "loreta_gsp"]
        pipes_dfs = {}
        for pname in all_pipes:
            dfs = load_data(metric, _HERE / "results" / pname)
            if dfs:
                pipes_dfs[pname] = dfs
                print(f"  Loaded pipeline '{pname}': {list(dfs.keys())}")
            else:
                print(f"  [skip] '{pname}': no results found")
        if not pipes_dfs:
            print("[ERROR] No results found for any pipeline.")
            return

        pipe_str = "_vs_".join(pipes_dfs)
        print(f"\n{'='*55}")
        print(f"  TSFM — Pipeline Comparison  [{metric}]")
        print(f"  Pipelines: {list(pipes_dfs)}")
        print(f"{'='*55}")

        out_dir = _HERE / "results" / "comparison"
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for fig, name in [
            (plot_comparison_overall(pipes_dfs, metric, ylabel), "plot_1_overall.png"),
            (plot_comparison_groups(pipes_dfs, metric, ylabel),  "plot_2_groups.png"),
        ]:
            p = fig_dir / name
            fig.savefig(p, dpi=300); plt.close(fig)
            print(f"  -> {p}")

        rp = out_dir / "comparison_report.md"
        rp.write_text(build_comparison_report(pipes_dfs, metric, ylabel), encoding="utf-8")
        print(f"  -> {rp}")
        print(f"{'='*55}\n  DONE\n{'='*55}\n")
        return

    # single pipeline
    pipeline    = args.pipeline
    results_dir = _HERE / "results" / pipeline
    figures_dir = results_dir / "figures"

    print(f"\n{'='*55}")
    print(f"  TSFM - {pipeline.capitalize()} Pipeline  [{metric}]")
    print(f"  Results: {results_dir}")
    print(f"{'='*55}")

    dfs = load_data(metric, results_dir)
    if not dfs:
        print(f"[ERROR] No CSVs in {results_dir}.")
        return

    channels = get_channels(dfs)
    print(f"  Models:   {list(dfs.keys())}")
    print(f"  Channels: {channels}")

    figures_dir.mkdir(parents=True, exist_ok=True)

    for fig, name in [
        (plot_overall(dfs, metric, ylabel),                 "plot_1_overall.png"),
        (plot_by_group(dfs, metric, ylabel),                "plot_2_groups.png"),
        (plot_by_channel(dfs, metric, ylabel, channels),    "plot_3_channels.png"),
    ]:
        p = figures_dir / name
        fig.savefig(p, dpi=300); plt.close(fig)
        print(f"  -> {p}")

    rp = results_dir / "summary_report.md"
    rp.write_text(build_report(dfs, metric, ylabel, channels, pipeline), encoding="utf-8")
    print(f"  -> {rp}")
    print(f"{'='*55}\n  DONE\n{'='*55}\n")


if __name__ == "__main__":
    main()

"""
Metrics computation from prediction cache.

Cache format (pickle):
    {
        subject_id: {
            'group': str,                # 'A', 'C', or 'F'
            channel: {
                'predictions': np.ndarray (n_windows, horizon),
                'targets':     np.ndarray (n_windows, horizon),
                'raw_std':     float,    # physical std of raw signal in Volts
            },
            ...
        },
        ...
    }

Output CSV columns:
    record_type, subject_id, group, electrode, mse_norm, mse_phys, n_windows
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["Fp1", "Fp2", "P3", "P4"]
GROUPS = ["A", "C", "F"]


def compute_metrics(cache_path: str | Path, csv_path: str | Path) -> pd.DataFrame:
    """
    Reads prediction cache, computes MSE (normalised + physical), writes CSV.
    Returns the full DataFrame.
    """
    cache_path = Path(cache_path)
    csv_path = Path(csv_path)

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    rows = []
    for subject_id, subj_data in cache.items():
        group = subj_data.get("group", "Unknown")
        for ch in CHANNELS:
            if ch not in subj_data:
                continue
            ch_data = subj_data[ch]
            preds = ch_data["predictions"]   # (n_windows, horizon)
            targets = ch_data["targets"]     # (n_windows, horizon)
            raw_std = float(ch_data["raw_std"])
            variance_scale = raw_std ** 2

            mse_per_window = np.mean((targets - preds) ** 2, axis=1)  # (n_windows,)
            mse_norm = float(np.mean(mse_per_window))
            mse_phys = mse_norm * variance_scale

            rows.append({
                "record_type": "per_patient_electrode",
                "subject_id": subject_id,
                "group": group,
                "electrode": ch,
                "mse_norm": mse_norm,
                "mse_phys": mse_phys,
                "n_windows": len(mse_per_window),
            })

    df_patient = pd.DataFrame(rows)
    if df_patient.empty:
        df_patient.to_csv(csv_path, index=False)
        return df_patient

    # Aggregate: per group × electrode
    summary_rows = []
    for g in GROUPS:
        df_g = df_patient[df_patient["group"] == g]
        if df_g.empty:
            continue

        # All electrodes combined
        summary_rows.append({
            "record_type": "group_all_electrodes",
            "subject_id": "",
            "group": g,
            "electrode": "ALL",
            "mse_norm": df_g["mse_norm"].mean(),
            "mse_phys": df_g["mse_phys"].mean(),
            "n_windows": 5,
        })

        # Per electrode
        for ch in CHANNELS:
            df_gc = df_g[df_g["electrode"] == ch]
            if df_gc.empty:
                continue
            summary_rows.append({
                "record_type": "group_per_electrode",
                "subject_id": "",
                "group": g,
                "electrode": ch,
                "mse_norm": df_gc["mse_norm"].mean(),
                "mse_phys": df_gc["mse_phys"].mean(),
                "n_windows": 5,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_final = pd.concat([df_patient, df_summary], ignore_index=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(csv_path, index=False)
    print(f"  -> Metrics saved: {csv_path} ({len(df_final)} rows)")
    return df_final

"""
Foundation-model residual causality — core computations.

This module reproduces, step by step, the mathematics of the Kaggle notebooks
``causality_chronos.ipynb``, ``causality_chronos-2.ipynb`` and
``causality_times-fm.ipynb`` (scripts_michal/causality_testing):

1. Window starts:   np.linspace(OFFSET, total_len - (ctx + hor), n_win, dtype=int)
                    with OFFSET = 3 s = 1500 samples at 500 Hz.
2. The forecaster predicts the target channel Y univariately from its own
   512-sample context (it never sees the driving channel X).
3. Residuals  e_t = y_t - y_hat_t  over the 64-sample horizon of every window.
4. For each lag L the residuals of ALL windows of one subject are pooled
   (n_obs = n_windows * horizon = 320) and two models are compared:
       restricted:  e_t = c                              (intercept only)
       full:        e_t = c + sum_{l=1..L} b_l x_{t-l}  (OLS)
   F = ((RSS_r - RSS_f) / L) / (RSS_f / (n_obs - L - 1)),   df = (L, n_obs - L - 1)
   p = F.sf(F, L, n_obs - L - 1);  if RSS_f == 0 then F = 0, p = 1.
   x_{t-l} uses the driving channel only up to t - 1, i.e. samples that precede
   the residual being explained (standard Granger-type lag structure).
5. One test per (subject, X -> Y pair, lag).  ``lag_ms = lag_steps * 1000 / fs``.

The aggregation helpers mirror ``Results_processing/tfsm_causality_processing.py``
(median F, % significant tests at alpha = 0.05, optimal lag = first lag that
maximises the % of significant tests, group-wise % at that lag).

No model is loaded here; the module only needs numpy, scipy and pandas.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

# ── Protocol constants (identical to the notebooks and to benchmark.pipelines) ──
FS = 500
CONTEXT_LEN = 512
HORIZON_LEN = 64
NUM_WINDOWS = 5
OFFSET_SAMPLES = int(3 * FS)  # 1500

DEFAULT_LAGS_SAMPLES = [5, 10, 20, 30, 40, 50]          # 10–100 ms at 500 Hz
DEFAULT_PAIRS = [("P3", "Fp1"), ("Fp1", "P3"), ("P4", "Fp2"), ("Fp2", "P4")]  # (X, Y)

GROUP_CODES = {"A": "AD", "C": "Control", "F": "FTD"}
GROUP_ALIASES = {
    "all": None,
    "ad": "A", "a": "A", "alzheimer": "A",
    "ftd": "F", "fd": "F", "f": "F",
    "control": "C", "cn": "C", "hc": "C", "c": "C",
}

NOTEBOOK_COLUMNS = [
    "subject_id", "group", "covariate_X", "target_Y",
    "lag_steps", "lag_ms", "f_stat_residual", "p_value_residual",
]


# ── Parameter parsing ─────────────────────────────────────────────────────────

def parse_int_list(text: str) -> list[int]:
    """Parse "[5, 10, 20]" or "5,10,20" into a list of ints (order kept, duplicates removed)."""
    s = str(text).strip()
    if not s:
        raise ValueError("Empty lag list.")
    if not s.startswith("["):
        s = f"[{s}]"
    try:
        values = ast.literal_eval(s)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Cannot parse lag list {text!r}; use e.g. \"[5,10,20]\".") from exc
    if isinstance(values, (int, float)):
        values = [values]
    out: list[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
            raise ValueError(f"Lag {v!r} is not an integer.")
        iv = int(v)
        if iv not in out:
            out.append(iv)
    return out


def parse_lags(text: str, unit: str = "samples", fs: int = FS,
               n_obs: int = NUM_WINDOWS * HORIZON_LEN) -> list[int]:
    """Return lags in samples. ``unit='ms'`` converts milliseconds to samples (must divide exactly)."""
    values = parse_int_list(text)
    if unit == "ms":
        step_ms = 1000.0 / fs
        lags = []
        for ms in values:
            steps = ms / step_ms
            if abs(steps - round(steps)) > 1e-9:
                raise ValueError(f"Lag {ms} ms is not a multiple of the sampling step ({step_ms:g} ms).")
            lags.append(int(round(steps)))
    elif unit == "samples":
        lags = values
    else:
        raise ValueError(f"Unknown lag unit {unit!r}.")
    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 sample (got {lag}).")
        if n_obs - lag - 1 <= 0:
            raise ValueError(f"Lag {lag} leaves no residual degrees of freedom (n_obs={n_obs}).")
    return lags


def parse_pairs(text: str | None) -> list[tuple[str, str]]:
    """Parse "P3->Fp1, Fp1->P3" into [(X, Y), ...]. ``None`` returns the notebook pairs."""
    if text is None:
        return list(DEFAULT_PAIRS)
    pairs = []
    for chunk in re.split(r"[;,]", str(text)):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.split(r"\s*(?:->|→|:)\s*", chunk)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Bad pair {chunk!r}; use the form X->Y (driver -> target).")
        x, y = parts
        if x == y:
            raise ValueError(f"Pair {chunk!r} uses the same channel twice.")
        if (x, y) not in pairs:
            pairs.append((x, y))
    if not pairs:
        raise ValueError("No pairs given.")
    return pairs


def resolve_group(text: str | None) -> str | None:
    """Map user group names to participants.tsv codes (A/C/F); ``None`` or 'all' -> None."""
    if text is None:
        return None
    key = str(text).strip().lower()
    if key not in GROUP_ALIASES:
        raise ValueError(f"Unknown group {text!r}. Use one of: all, ad, ftd (fd), control.")
    return GROUP_ALIASES[key]


# ── Windowing (verbatim notebook formula) ────────────────────────────────────

def window_starts(total_len: int, ctx: int = CONTEXT_LEN, hor: int = HORIZON_LEN,
                  n_win: int = NUM_WINDOWS, offset: int = OFFSET_SAMPLES) -> np.ndarray:
    need = ctx + hor
    available_len = total_len - offset
    if available_len < need:
        raise ValueError("Signal too short.")
    hi = total_len - need
    return np.linspace(offset, hi, n_win, dtype=int)


def window_contexts(y: np.ndarray, starts: np.ndarray, ctx: int = CONTEXT_LEN) -> list[np.ndarray]:
    return [y[s: s + ctx] for s in starts]


def window_targets(y: np.ndarray, starts: np.ndarray, ctx: int = CONTEXT_LEN,
                   hor: int = HORIZON_LEN) -> np.ndarray:
    return np.stack([y[s + ctx: s + ctx + hor] for s in starts])


# ── Residual F-test ───────────────────────────────────────────────────────────

def lagged_design(x: np.ndarray, s: int, lag: int, ctx: int = CONTEXT_LEN,
                  hor: int = HORIZON_LEN) -> np.ndarray:
    """Design matrix for one window: [1, x_{t-1}, ..., x_{t-lag}] for t in the horizon."""
    cols = [np.ones(hor)]
    for l in range(1, lag + 1):
        cols.append(x[s + ctx - l: s + ctx + hor - l])
    return np.column_stack(cols)


@dataclass
class FTestResult:
    f_stat: float
    p_value: float
    n_obs: int
    df_num: int
    df_den: int
    rss_restricted: float
    rss_full: float


def residual_f_test(residuals: np.ndarray, x: np.ndarray, starts: np.ndarray, lag: int,
                    ctx: int = CONTEXT_LEN, hor: int = HORIZON_LEN) -> FTestResult:
    """
    Pooled residual-causality F-test for one subject, one X -> Y pair and one lag.

    residuals : (n_windows, hor) array of y_true - y_pred for the target channel.
    x         : full driving-channel signal (same sampling / scaling as y).
    """
    residuals = np.asarray(residuals)
    y_res = residuals.reshape(-1)
    X = np.vstack([lagged_design(x, int(s), lag, ctx, hor) for s in starts])

    res_mean = np.mean(y_res)
    rss_restricted = float(np.sum((y_res - res_mean) ** 2))

    # OLS with intercept (the first column of X is the constant).  Fitted values
    # are identical to sklearn LinearRegression().fit(X, y) used in the notebooks.
    beta, *_ = np.linalg.lstsq(X.astype(np.float64), y_res.astype(np.float64), rcond=None)
    pred_full = X @ beta
    rss_full = float(np.sum((y_res - pred_full) ** 2))

    obs = int(y_res.shape[0])
    k = int(lag)
    df_den = obs - k - 1
    if rss_full > 0:
        f_stat = ((rss_restricted - rss_full) / k) / (rss_full / df_den)
        p_val = float(f_dist.sf(f_stat, k, df_den))
    else:
        f_stat, p_val = 0.0, 1.0
    return FTestResult(float(f_stat), p_val, obs, k, df_den, rss_restricted, rss_full)


def pair_tests(subject_id: str, group: str, x_name: str, y_name: str,
               x: np.ndarray, y: np.ndarray, predictions: np.ndarray, starts: np.ndarray,
               lags: list[int], fs: int = FS, ctx: int = CONTEXT_LEN,
               hor: int = HORIZON_LEN) -> list[dict]:
    """All lag tests for one subject and one X -> Y pair (notebook row format + extras)."""
    y_true = window_targets(y, starts, ctx, hor)
    predictions = np.asarray(predictions)
    if predictions.shape != y_true.shape:
        raise ValueError(f"Prediction shape {predictions.shape} != target shape {y_true.shape}.")
    residuals = y_true - predictions
    rows = []
    for lag in lags:
        r = residual_f_test(residuals, x, starts, lag, ctx, hor)
        rows.append({
            "subject_id": subject_id,
            "group": group,
            "covariate_X": x_name,
            "target_Y": y_name,
            "lag_steps": int(lag),
            "lag_ms": int(round(lag * 1000.0 / fs)),
            "f_stat_residual": r.f_stat,
            "p_value_residual": r.p_value,
            "n_obs": r.n_obs,
            "df_num": r.df_num,
            "df_den": r.df_den,
            "rss_restricted": r.rss_restricted,
            "rss_full": r.rss_full,
        })
    return rows


# ── Aggregation (numeric part of tfsm_causality_processing.py) ───────────────

def add_derived_columns(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = df.copy()
    out["group"] = out["group"].fillna("Unknown")
    out["Pair"] = out["covariate_X"] + " -> " + out["target_Y"]
    out["is_significant"] = (out["p_value_residual"] < alpha).astype(int)
    return out


def summarize_by_lag(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Median F and % significant tests per model, scope (ALL + each group), pair and lag."""
    d = add_derived_columns(df, alpha)
    keys = ["model", "Pair", "covariate_X", "target_Y", "lag_steps", "lag_ms"]
    frames = []
    overall = d.groupby(keys).agg(
        n_tests=("is_significant", "size"),
        Median_F=("f_stat_residual", "median"),
        Sig_Pct=("is_significant", lambda s: s.mean() * 100),
    ).reset_index()
    overall.insert(1, "scope", "ALL")
    frames.append(overall)
    for grp in sorted(g for g in d["group"].unique() if g != "Unknown"):
        sub = d[d["group"] == grp]
        g = sub.groupby(keys).agg(
            n_tests=("is_significant", "size"),
            Median_F=("f_stat_residual", "median"),
            Sig_Pct=("is_significant", lambda s: s.mean() * 100),
        ).reset_index()
        g.insert(1, "scope", grp)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def optimal_lag_table(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Table1 + Table2 of tfsm_causality_processing.py in one frame, per model:
    optimal lag = first lag (ascending) maximising Sig_Pct over ALL selected
    subjects; then Median_F, Sig_Pct and group-wise Sig_Pct at that lag.
    """
    d = add_derived_columns(df, alpha)
    rows = []
    for model, dm in d.groupby("model", sort=False):
        lag_stats = dm.groupby(["Pair", "lag_ms"]).agg(
            Median_F=("f_stat_residual", "median"),
            Sig_Pct=("is_significant", lambda s: s.mean() * 100),
            n_tests=("is_significant", "size"),
        ).reset_index()
        idx = lag_stats.groupby("Pair")["Sig_Pct"].idxmax()
        best = lag_stats.loc[idx]
        for _, b in best.iterrows():
            row = {
                "model": model, "Pair": b["Pair"], "Optimal_Lag_ms": int(b["lag_ms"]),
                "Median_F": float(b["Median_F"]), "Sig_Pct": float(b["Sig_Pct"]),
                "n_tests": int(b["n_tests"]),
            }
            at_lag = dm[(dm["Pair"] == b["Pair"]) & (dm["lag_ms"] == b["lag_ms"])]
            for grp in ["A", "C", "F"]:
                g = at_lag[at_lag["group"] == grp]
                row[f"Sig_Pct_{grp}"] = float(g["is_significant"].mean() * 100) if len(g) else np.nan
                row[f"n_tests_{grp}"] = int(len(g))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model", "Sig_Pct"], ascending=[True, False]).reset_index(drop=True)
    return out

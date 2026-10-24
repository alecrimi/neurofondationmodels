"""
Core computations for foundation-model residual-causality analysis.

Analysis overview
-----------------
1. Select time-series windows using evenly spaced start indices.
2. Use a forecasting model to predict the target channel Y from its own
   historical context. The driving channel X is not provided to the
   forecasting model.
3. Calculate prediction residuals as e_t = y_t - y_hat_t over the forecast
   horizon of each window.
4. For each subject, directed channel pair X -> Y, and lag L, pool the
   residuals across windows and compare two linear regression models:

   Restricted model:
       e_t = c

   Full model:
       e_t = c + sum(b_l * x_{t-l}, l=1..L)

   The coefficients of the full model are estimated using ordinary least
   squares (OLS). The F-statistic compares the reduction in residual sum
   of squares against the residual variance of the full model:

       F = ((RSS_restricted - RSS_full) / L)
           / (RSS_full / (n_obs - L - 1))

   The numerator and denominator degrees of freedom are:
       df_num = L
       df_den = n_obs - L - 1

   The p-value is calculated from the upper tail of the corresponding
   F-distribution. If the full-model residual sum of squares is zero or
   invalid, the test raises an exception rather than returning a potentially
   misleading statistic or p-value.

Statistical assumptions
-----------------------
The pooled F-test is intended for non-overlapping analysis windows. When
windows overlap, pooled residuals may share underlying samples or become
dependent, making the classical F-test p-values unreliable. Non-overlapping
windows alone do not guarantee all statistical assumptions: p-value
interpretation also depends on the usual assumptions of the regression
test, including an appropriate error structure.

The test evaluates whether lagged values of X provide additional linear
explanatory information about the residuals of Y. A statistically
significant result indicates residual predictive dependence; it does not,
by itself, establish physical, biological, or causal influence.

Outputs
-------
The module provides individual F-test results, per-subject and per-pair
result rows, and aggregate tables containing median F-statistics,
percentages of significant tests, and selected optimal lags.

The module does not load or run a forecasting model. It operates on
signals, predictions, residuals, and analysis metadata supplied by the
caller.

Dependencies: NumPy, SciPy, and pandas.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

# ── Protocol constants ──────────────────────────────────────────────────────────
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
    """
    Parse a string containing integer values into a list of unique integers.

    Accepts a comma-separated string such as "5,10,20", a bracketed list
    such as "[5, 10, 20]", or a single numeric value represented as text.
    The original order is preserved, and duplicate values are removed.

    Parameters
    ----------
    text : str
        Input string containing one or more integer-valued numbers.

    Returns
    -------
    list[int]
        Parsed integer values in their original order, without duplicates.

    Raises
    ------
    ValueError
        If the input is empty, cannot be parsed, or contains a value that is
        not an integer or an integer-valued number. Boolean values are rejected.
    """
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
    """
    Parse lag values and validate them for the residual-causality F-test.

    Lag values can be specified in samples or milliseconds. When milliseconds
    are used, each value must correspond exactly to an integer number of
    samples at the specified sampling frequency.

    Each lag must be at least one sample, and the expected observation count
    must leave positive denominator degrees of freedom for the F-test.

    Parameters
    ----------
    text : str
        Lag values, for example "[5, 10, 20]" or "10,20,40".
    unit : str, default="samples"
        Unit of the input values. Supported values are "samples" and "ms".
    fs : int, default=FS
        Sampling frequency in samples per second. Used to convert milliseconds
        to samples.
    n_obs : int, default=NUM_WINDOWS * HORIZON_LEN
        Expected number of pooled residual observations. Used to check that
        each lag leaves positive denominator degrees of freedom.

    Returns
    -------
    list[int]
        Validated lag values expressed in samples, preserving input order
        and removing duplicates.

    Raises
    ------
    ValueError
        If the input cannot be parsed, the unit is unsupported, a lag is less
        than one sample, a lag in milliseconds cannot be represented exactly
        at the specified sampling frequency, or the lag leaves no residual
        degrees of freedom.
    """
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
    """
    Parse directed driving-to-target channel pairs.

    Pairs can be provided using "X->Y", "X→Y", or "X:Y" notation.
    Multiple pairs may be separated by commas or semicolons.

    If text is None, the predefined default channel pairs are returned.
    Pairs are kept in input order, and duplicates are removed.

    Parameters
    ----------
    text : str or None
        String containing directed channel pairs, or None to use the defaults.

    Returns
    -------
    list[tuple[str, str]]
        Unique directed pairs represented as (driving_channel, target_channel).
        The first element is X and the second element is Y.

    Raises
    ------
    ValueError
        If a pair is malformed, a pair contains the same channel as both
        driver and target, or no valid pairs are provided.
    """
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
    """
    Normalize a user-specified participant group to its dataset code.

    Recognized aliases are mapped to the participant group codes:
    "A" for Alzheimer's disease, "C" for controls, and "F" for FTD.
    The special value "all", or None, indicates that no group filter is applied.

    Parameters
    ----------
    text : str or None
        Group name or alias, such as "ad", "control", "ftd", or "all".

    Returns
    -------
    str or None
        The corresponding group code ("A", "C", or "F"), or None if all groups
        should be included.

    Raises
    ------
    ValueError
        If the supplied group name or alias is not recognized.
    """
    if text is None:
        return None
    key = str(text).strip().lower()
    if key not in GROUP_ALIASES:
        raise ValueError(f"Unknown group {text!r}. Use one of: all, ad, ftd (fd), control.")
    return GROUP_ALIASES[key]


# ── Windowing ──────────────────────────────────────────────────────────────────

def window_starts(total_len: int, ctx: int = CONTEXT_LEN, hor: int = HORIZON_LEN,
                  n_win: int = NUM_WINDOWS, offset: int = OFFSET_SAMPLES) -> np.ndarray:
    """
    Generate evenly spaced start indices for time-series analysis windows.

    Each complete window consists of a context followed by a forecast horizon.
    The first start index is determined by the offset, and the final start
    index is the latest position at which the complete context and horizon
    fit inside the signal. Generated complete windows must not overlap.

    Parameters
    ----------
    total_len : int
        Total signal length in samples.
    ctx : int, default=CONTEXT_LEN
        Number of context samples preceding the forecast horizon.
    hor : int, default=HORIZON_LEN
        Number of forecast-horizon samples.
    n_win : int, default=NUM_WINDOWS
        Number of window start indices to generate.
    offset : int, default=OFFSET_SAMPLES
        Earliest allowed window start index, in samples.

    Returns
    -------
    np.ndarray
        Integer array of evenly spaced, non-overlapping window start indices.

    Raises
    ------
    ValueError
        If the signal is too short for the requested windows, the parameters
        are invalid, or the generated windows overlap.
    """
    if ctx < 1 or hor < 1:
        raise ValueError(
            f"`ctx` and `hor` must be positive; got ctx={ctx}, hor={hor}."
        )

    if n_win < 1:
        raise ValueError(f"`n_win` must be at least 1; got n_win={n_win}.")

    if offset < 0:
        raise ValueError(f"`offset` must be non-negative; got offset={offset}.")

    need = ctx + hor
    if total_len - offset < need:
        raise ValueError(
            "Signal too short for the requested context and forecast horizon: "
            f"total_len={total_len}, offset={offset}, ctx={ctx}, hor={hor}."
        )

    starts = np.linspace(offset, total_len - need, n_win, dtype=int)

    # Require complete analysis windows (context + horizon) not to overlap.
    if starts.size > 1:
        gaps = np.diff(starts)
        overlapping = np.flatnonzero(gaps < need)

        if overlapping.size:
            i = int(overlapping[0])
            raise ValueError(
                "Generated analysis windows overlap. "
                f"Window starts are {starts[i]} and {starts[i + 1]}, "
                f"with a gap of {gaps[i]} samples; at least {need} "
                f"samples are required for non-overlapping windows "
                f"(ctx={ctx}, hor={hor}). "
                f"total_len={total_len}, offset={offset}, n_win={n_win}. "
                "Reduce n_win or use a longer signal."
            )

    return starts


def window_contexts(y: np.ndarray, starts: np.ndarray, ctx: int = CONTEXT_LEN) -> list[np.ndarray]:
    """
    Extract historical context segments from a one-dimensional signal.

    Parameters
    ----------
    y : np.ndarray
        One-dimensional time-series signal.
    starts : np.ndarray
        Window start indices expressed in samples.
    ctx : int, default=CONTEXT_LEN
        Number of samples in each context segment.

    Returns
    -------
    list[np.ndarray]
        List of context segments, one per start index, in the same order as
        the supplied starts.
    """
    return [y[s: s + ctx] for s in starts]


def window_targets(y: np.ndarray, starts: np.ndarray, ctx: int = CONTEXT_LEN,
                   hor: int = HORIZON_LEN) -> np.ndarray:
    """
    Extract the ground-truth forecast horizons for a set of windows.

    For each window start s, the target segment contains the samples
    y[s + ctx : s + ctx + hor]. These segments can be compared with model
    predictions to calculate forecast residuals.

    Parameters
    ----------
    y : np.ndarray
        One-dimensional target-channel signal.
    starts : np.ndarray
        Window start indices expressed in samples.
    ctx : int, default=CONTEXT_LEN
        Number of historical context samples before the target horizon.
    hor : int, default=HORIZON_LEN
        Number of target samples in each forecast horizon.

    Returns
    -------
    np.ndarray
        Two-dimensional array with shape (n_windows, hor), containing the
        ground-truth target segment for each window.
    """
    return np.stack([y[s + ctx: s + ctx + hor] for s in starts])


# ── Residual F-test ───────────────────────────────────────────────────────────

def lagged_design(x: np.ndarray, s: int, lag: int, ctx: int = CONTEXT_LEN,
                  hor: int = HORIZON_LEN) -> np.ndarray:
    """
    Build the regression design matrix for one window and a specified lag.

    The first column is an intercept (ones). Each subsequent column contains
    a lagged version of the driving signal X aligned with the target residuals
    in the forecast horizon.

    For a window starting at sample s, the predictor for lag l is:

        x[s + ctx - l : s + ctx + hor - l]

    Thus, the value used for each target time t is x[t - l], and only samples
    preceding that target time are used.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional driving-channel signal, expressed in samples and
        aligned in time with the target signal.
    s : int
        Start index of the analysis window.
    lag : int
        Number of lagged predictors to include. A lag of L produces L
        predictor columns in addition to the intercept.
    ctx : int, default=CONTEXT_LEN
        Number of context samples before the forecast horizon.
    hor : int, default=HORIZON_LEN
        Number of time points in the forecast horizon.

    Returns
    -------
    np.ndarray
        Design matrix with shape (hor, lag + 1). The first column contains
        ones, and the remaining columns contain lagged values of X.
    """
    cols = [np.ones(hor)]
    for l in range(1, lag + 1):
        cols.append(x[s + ctx - l: s + ctx + hor - l])
    return np.column_stack(cols)


@dataclass
class FTestResult:
    """
    Store the numerical results of one residual-causality F-test.

    Attributes
    ----------
    f_stat : float
        Observed F-statistic measuring the improvement in fit relative to the
        residual variance of the full model.
    p_value : float
        Upper-tail probability of the observed F-statistic under the null
        hypothesis and the assumptions of the classical F-test.
    n_obs : int
        Number of pooled residual entries used in the regression.
    df_num : int
        Numerator degrees of freedom, equal to the number of lagged predictors.
    df_den : int
        Denominator degrees of freedom, equal to n_obs - df_num - 1.
    rss_restricted : float
        Residual sum of squares for the intercept-only model.
    rss_full : float
        Residual sum of squares for the model containing lagged X predictors.
    """
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
    Perform a pooled residual-causality F-test for one subject, one directed
    driving-to-target channel pair (X -> Y), and one lag.

    The input residuals are prediction errors for the target channel Y,
    calculated as y_true - y_pred over the forecast horizon of each window.
    Residuals from all supplied windows are pooled into a single response
    vector.

    Two ordinary least-squares models are compared:

        Restricted model:
            e_t = c

        Full model:
            e_t = c + sum(b_l * x_{t-l}, l=1..lag)

    The restricted model predicts the mean residual. The full model adds
    lagged values of the driving signal X as predictors. The F-statistic tests
    the joint null hypothesis that all coefficients b_l of the lagged X 
    predictors are zero.

    The classical F-test p-value is intended for non-overlapping analysis
    windows. Overlapping windows can cause pooled residual observations to
    share underlying samples or become dependent, making the nominal p-values
    unreliable. Non-overlap alone does not guarantee all test assumptions;
    the error structure must also be appropriate for classical OLS inference.

    A significant result indicates that lagged X provides additional linear
    explanatory information about the residuals of Y. It does not, by itself,
    establish physical or biological causality.

    Parameters
    ----------
    residuals : np.ndarray
        Array with shape (n_windows, hor), containing the target-channel
        residuals (y_true - y_pred). Order of rows corresponds to the order
        of entries in starts
    x : np.ndarray
        Full driving-channel signal X, sampled and time-aligned with the
        target signal Y.
    starts : np.ndarray
        Start indices of the analysis windows, expressed in samples. Their
        order must match the row order in residuals.
    lag : int
        Number of lagged X predictors to include in the full regression.
        The lag is measured in samples and must be at least one.
    ctx : int, default=CONTEXT_LEN
        Number of context samples before the forecast horizon. Used to align
        the lagged X predictors with the target residuals.
    hor : int, default=HORIZON_LEN
        Number of residuals associated with each window.

    Returns
    -------
    FTestResult object

    Raises
    ------
    ValueError
        If the input arrays have invalid dimensions or incompatible shapes,
        if residuals or lagged predictors contain non-finite values, if window
        starts are invalid or the analysis windows overlap, if lagged predictor
        slices extend beyond the driving signal, if the design matrix has an
        unexpected shape, if denominator degrees of freedom are non-positive,
        or if the full-model residual sum of squares is not positive.

    Notes
    -----
    The number of pooled residual entries is normally:

        n_obs = n_windows * hor

    The function rejects overlapping analysis windows, defined as windows
    whose start indices are separated by fewer than ctx + hor samples.

    Non-overlapping windows do not necessarily produce statistically
    independent residual observations. The caller is responsible for assessing 
    whether the statistical assumptions required for valid inference are 
    reasonable.
    """
    residuals = np.asarray(residuals)
    x = np.asarray(x)
    starts = np.asarray(starts)

    if residuals.ndim != 2:
        raise ValueError(
            "`residuals` must be a 2D array with shape "
            f"(n_windows, hor); got shape={residuals.shape}."
        )

    if residuals.shape[0] == 0:
        raise ValueError("`residuals` contains no windows.")

    if residuals.shape[1] != hor:
        raise ValueError(
            f"Residual horizon mismatch: residuals.shape[1]={residuals.shape[1]}, "
            f"but hor={hor}."
        )

    if x.ndim != 1:
        raise ValueError(
            f"`x` must be a 1D signal; got shape={x.shape}."
        )

    if starts.ndim != 1:
        raise ValueError(
            f"`starts` must be a 1D array; got shape={starts.shape}."
        )

    if starts.size == 0:
        raise ValueError("`starts` contains no window start indices.")

    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError(
            f"`starts` must contain integer sample indices; got dtype={starts.dtype}."
        )

    if residuals.shape[0] != starts.size:
        raise ValueError(
            "The number of residual windows must match the number of window starts: "
            f"n_residual_windows={residuals.shape[0]}, n_starts={starts.size}, "
            f"residuals.shape={residuals.shape}."
        )

    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError(f"`lag` must be a positive integer; got {lag!r}.")

    if ctx < 1 or hor < 1:
        raise ValueError(
            f"`ctx` and `hor` must be positive; got ctx={ctx}, hor={hor}."
        )

    if not np.isfinite(residuals).all():
        raise ValueError(
            "Non-finite values detected in `residuals`; "
            "the F-test requires finite residual observations."
        )

    y_res = residuals.reshape(-1)

    # Check that complete analysis windows do not overlap.
    ordered_starts = np.sort(starts)
    window_width = ctx + hor
    gaps = np.diff(ordered_starts)
    overlapping = np.flatnonzero(gaps < window_width)

    if overlapping.size:
        i = int(overlapping[0])
        raise ValueError(
            "Overlapping analysis windows are not supported by this F-test. "
            f"Window starts={int(ordered_starts[i])} and "
            f"{int(ordered_starts[i + 1])}, "
            f"gap={int(gaps[i])} samples, required_gap={window_width}. "
            f"lag={lag}, ctx={ctx}, hor={hor}, "
            f"n_windows={len(starts)}."
        )

    # Verify that the lagged predictor slices lie within the driving signal.
    first_required = starts + ctx - lag
    last_stop_exclusive = starts + ctx + hor - 1

    invalid_bounds = (
        (first_required < 0)
        | (last_stop_exclusive > x.size)
    )

    if invalid_bounds.any():
        bad_starts = starts[invalid_bounds].tolist()
        raise ValueError(
            "Window boundaries are incompatible with the driving signal "
            "and requested lagged predictors. "
            f"Signal length={x.size}, invalid_starts={bad_starts}, "
            f"lag={lag}, ctx={ctx}, hor={hor}."
        )

    # Build and validate one design matrix per window.
    designs = []

    for s in starts:
        design = lagged_design(x, int(s), int(lag), ctx, hor)

        expected_shape = (hor, int(lag) + 1)
        if design.shape != expected_shape:
            raise ValueError(
                "Unexpected lagged design matrix shape: "
                f"start={int(s)}, expected={expected_shape}, "
                f"actual={design.shape}, lag={lag}, ctx={ctx}, hor={hor}."
            )

        if not np.isfinite(design).all():
            raise ValueError(
                "Non-finite values detected in the lagged design matrix: "
                f"start={int(s)}, lag={lag}, ctx={ctx}, hor={hor}."
            )

        designs.append(design)

    X = np.vstack(designs)

    # Each flattened residual must have exactly one corresponding design row.
    if X.shape[0] != y_res.size:
        raise ValueError(
            "Design matrix row count does not match the number of residuals: "
            f"X.shape={X.shape}, n_residuals={y_res.size}, "
            f"residuals.shape={residuals.shape}, "
            f"n_starts={len(starts)}, lag={lag}."
        )

    if X.shape[1] != int(lag) + 1:
        raise ValueError(
            "Unexpected number of design matrix columns: "
            f"expected={int(lag) + 1}, actual={X.shape[1]}, lag={lag}."
        )

    res_mean = np.mean(y_res)
    rss_restricted = float(np.sum((y_res - res_mean) ** 2))

    # OLS with intercept (the first column of X is the constant).
    beta, *_ = np.linalg.lstsq(
        X.astype(np.float64),
        y_res.astype(np.float64),
        rcond=None,
    )
    pred_full = X @ beta
    rss_full = float(np.sum((y_res - pred_full) ** 2))

    obs = int(y_res.shape[0])
    k = int(lag)
    df_den = obs - k - 1

    if not rss_full > 0:
        raise ValueError(
            "Cannot compute residual-causality F-test: RSS_full is zero or non-positive "
            "(or NaN), so the F-statistic is undefined.\n"
            f"lag={lag}, ctx={ctx}, hor={hor}\n"
            f"starts={np.asarray(starts).tolist()}\n"
            f"n_windows={len(starts)}, obs={obs}, k={k}, df_den={df_den}\n"
            f"rss_restricted={rss_restricted}, rss_full={rss_full}"
        )

    if df_den <= 0:
        raise ValueError(
            "Cannot compute residual-causality F-test: non-positive denominator "
            f"degrees of freedom (df_den={df_den}).\n"
            f"lag={lag}, ctx={ctx}, hor={hor}\n"
            f"starts={np.asarray(starts).tolist()}\n"
            f"n_windows={len(starts)}, obs={obs}, k={k}"
        )

    f_stat = ((rss_restricted - rss_full) / k) / (rss_full / df_den)
    p_val = float(f_dist.sf(f_stat, k, df_den))

    return FTestResult(float(f_stat), p_val, obs, k, df_den, rss_restricted, rss_full)


def pair_tests(subject_id: str, group: str, x_name: str, y_name: str,
               x: np.ndarray, y: np.ndarray, predictions: np.ndarray, starts: np.ndarray,
               lags: list[int], fs: int = FS, ctx: int = CONTEXT_LEN,
               hor: int = HORIZON_LEN) -> list[dict]:
    """
    Run residual-causality F-tests for multiple lags for one subject and one
    directed driving-to-target channel pair (X -> Y).

    The ground-truth target horizons are extracted from Y, and the
    predictions are subtracted from them to obtain the residuals. One pooled
    residual-causality F-test is then performed for each requested lag.

    Parameters
    ----------
    subject_id : str
        Identifier of the subject being analyzed.
    group : str
        Participant group code or label associated with the subject.
    x_name : str
        Name of the driving channel X.
    y_name : str
        Name of the target channel Y.
    x : np.ndarray
        Full driving-channel signal.
    y : np.ndarray
        Full target-channel signal.
    predictions : np.ndarray
        Forecasted values for Y, with shape (n_windows, hor). Rows must follow
        the same order as starts.
    starts : np.ndarray
        Start indices of the analysis windows, expressed in samples.
    lags : list[int]
        Lag values to test, expressed in samples.
    fs : int, default=FS
        Sampling frequency in samples per second, used to convert lag lengths
        from samples to milliseconds in the output.
    ctx : int, default=CONTEXT_LEN
        Number of historical context samples preceding each forecast horizon.
    hor : int, default=HORIZON_LEN
        Number of samples in each forecast horizon.

    Returns
    -------
    list[dict]
        One result dictionary per tested lag. Each dictionary includes subject
        and group metadata, channel-pair identifiers, lag in samples and
        milliseconds, F-statistic, p-value, observation count, degrees of
        freedom, and restricted and full residual sums of squares.

    Raises
    ------
    ValueError
        If the prediction array shape does not match the extracted target
        horizons, or if an individual residual-causality F-test cannot be
        computed.
    """
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


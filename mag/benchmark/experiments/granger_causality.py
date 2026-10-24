"""
Core computations for classical linear Granger-causality analysis.

Analysis overview
-----------------
1. Select evenly spaced, non-overlapping context-plus-horizon windows.
2. Extract target-channel observations Y_t from each forecast horizon.
3. Build predictors from past values of Y and the driving channel X.
4. Pool horizon observations across windows and compare two OLS models:

   Restricted model:
       Y_t = c + sum(a_l * Y_{t-l}, l=1..L)

   Full model:
       Y_t = c + sum(a_l * Y_{t-l}, l=1..L)
               + sum(b_l * X_{t-l}, l=1..L)

   Each model is fitted separately, with one shared coefficient vector
   across all supplied horizons. The context supplies lag history only.
   Histories are extracted at the original sample positions; gaps between
   windows are never treated as adjacent samples.

   The F-statistic compares the reduction in residual sum of squares:

       F = ((RSS_restricted - RSS_full) / L)
           / (RSS_full / (n_obs - 2*L - 1))

   The numerator and denominator degrees of freedom are:
       df_num = L
       df_den = n_obs - 2*L - 1

   The null hypothesis is b_1 = ... = b_L = 0. The p-value is the upper-tail
   probability under the corresponding F-distribution. Singular or invalid
   fits raise an exception rather than returning a misleading result.

Statistical assumptions
-----------------------
This implementation requires non-overlapping complete analysis windows,
matching the project's residual-causality window-selection rule. This is
an implementation restriction, not a universal requirement of Granger
causality. Reusing predictor history alone is not a reason a test is invalid.
Non-overlap does not guarantee valid classical F-test p-values: the model
and its regression errors must satisfy the relevant inference assumptions.
Pooling also assumes common coefficients across the selected fragments.
P-values are nominal; no multiple-testing correction is applied here.

A significant result indicates additional linear predictive information
from X after controlling for the chosen Y lags. It does not, by itself,
establish physical or biological causality.

Outputs
-------
The module provides individual F-test results and per-subject, per-pair,
per-lag result rows. It does not load data, standardize signals, run a
foundation model, aggregate results or write files.

Unlike causality_granger.ipynb, which fits separate regressions using 576
response samples per window, this module pools the 64-sample horizons
from five windows (320 response observations by default).

Dependencies: NumPy and SciPy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import f as f_dist
FS = 500
CONTEXT_LEN = 512
HORIZON_LEN = 64
NUM_WINDOWS = 5
OFFSET_SAMPLES = 3 * FS
DEFAULT_LAGS_SAMPLES = [5, 10, 20, 30, 40, 50]
DEFAULT_PAIRS = [("P3", "Fp1"), ("Fp1", "P3"), ("P4", "Fp2"), ("Fp2", "P4")]
RESULT_COLUMNS = ["subject_id", "group", "covariate_X", "target_Y", "lag_steps", "lag_ms",
    "f_stat_granger", "p_value_granger", "model", "n_obs", "df_num", "df_den",
    "rss_restricted", "rss_full", "n_windows", "context_len", "horizon_len",
    "pipeline", "scaling", "offset_samples", "standardization_scope"]
@dataclass(frozen=True)
class FTestResult:
    """
    Store the numerical results of one Granger-causality F-test.

    Attributes
    ----------
    f_stat : float
        Improvement in fit relative to the full-model residual variance.
    p_value : float
        Upper-tail probability under the null hypothesis and classical
        F-test assumptions; no multiple-testing correction is applied.
    n_obs : int
        Number of pooled target observations used in both regressions.
    df_num : int
        Numerator degrees of freedom, equal to the number of added X lags.
    df_den : int
        Denominator degrees of freedom, equal to n_obs - 2*df_num - 1.
    rss_restricted : float
        Residual sum of squares for the intercept-plus-Y-history model.
    rss_full : float
        Residual sum of squares for the model including Y and X history.
    """
    f_stat: float
    p_value: float
    n_obs: int
    df_num: int
    df_den: int
    rss_restricted: float
    rss_full: float

def _series(values, name):
    """
    Convert an input signal to a one-dimensional float64 array.

    Parameters
    ----------
    values : array_like
        Signal values to convert.
    name : str
        Signal label used in validation messages (X or Y).

    Returns
    -------
    np.ndarray
        One-dimensional signal with dtype float64.

    Raises
    ------
    ValueError
        If conversion fails or the signal is not one-dimensional.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"`{name.lower()}` must be a 1D signal; got shape={a.shape}.")
    return a

def granger_f_test(x, y, starts, lag: int, ctx: int = CONTEXT_LEN,
                   hor: int = HORIZON_LEN) -> FTestResult:
    """
    Perform a pooled linear Granger F-test for one directed pair and lag.

    Two OLS models are fitted to the same target observations:

        Restricted model:
            Y_t = c + sum(a_l * Y_{t-l}, l=1..lag)

        Full model:
            Y_t = c + sum(a_l * Y_{t-l}, l=1..lag)
                    + sum(b_l * X_{t-l}, l=1..lag)

    All coefficients are re-estimated in each model. Each fit shares one
    coefficient vector across all windows. The null hypothesis is that all
    added X-history coefficients are zero. The response consists of actual Y
    values, not foundation-model residuals.

    Parameters
    ----------
    x : array_like
        Full, one-dimensional driving-channel signal X.
    y : array_like
        Full target-channel signal Y, equal in length and time-aligned with X.
    starts : np.ndarray
        Nonempty one-dimensional integer array of context start indices.
    lag : int
        Number of past values of each signal included in the full model.
        Must be positive and no greater than ctx.
    ctx : int, default=CONTEXT_LEN
        Number of context samples preceding the horizon. Predictor history
        is taken from original signal positions, never across artificial joins.
    hor : int, default=HORIZON_LEN
        Number of target observations contributed by each window.

    Returns
    -------
    FTestResult
        F-statistic, nominal p-value, observation count, degrees of freedom,
        and residual sums of squares for both fitted models.

    Raises
    ------
    ValueError
        If signals have invalid dimensions or unequal lengths, starts are
        invalid, complete windows overlap or exceed signal bounds, lag/history
        parameters are invalid, used values are non-finite, a design matrix is
        rank-deficient, full-model degrees of freedom are non-positive, or a
        valid F-statistic cannot be computed.

    Notes
    -----
    The number of pooled target observations is n_windows * hor. The full
    model has 2*lag + 1 parameters, giving df_den = n_obs - 2*lag - 1.
    This is an in-sample comparison of nested regressions, not an out-of-sample
    forecast evaluation. Lagged actual observations are used for each target
    time, including observed earlier samples within the same horizon.

    Complete windows must be separated by at least ctx + hor samples. This
    restriction matches the existing residual-causality implementation; it is
    not a general mathematical necessity for Granger tests. Non-overlap alone
    does not guarantee appropriate regression-error assumptions or stationarity.
    A significant result does not establish physical or biological causality.
    """
    x, y = _series(x, "X"), _series(y, "Y")
    starts = np.asarray(starts)
    if len(x) != len(y):
        raise ValueError("X and Y must have equal lengths.")
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError(f"`lag` must be a positive integer; got {lag!r}.")
    if ctx < 1 or hor < 1:
        raise ValueError(
            f"`ctx` and `hor` must be positive; got ctx={ctx}, hor={hor}."
        )
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (ctx, hor)):
        raise ValueError(f"`ctx` and `hor` must be integers; got ctx={ctx!r}, hor={hor!r}.")
    if lag > ctx:
        raise ValueError("lag must not exceed context length; history must stay inside the window.")
    if starts.ndim != 1:
        raise ValueError(f"`starts` must be a 1D array; got shape={starts.shape}.")
    if starts.size == 0:
        raise ValueError("`starts` contains no window start indices.")
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError(f"`starts` must contain integer sample indices; got dtype={starts.dtype}.")
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
    invalid_targets = (starts < 0) | (starts + ctx + hor > y.size)
    if invalid_targets.any():
        raise ValueError(
            "Window boundaries are incompatible with the target signal. "
            f"Signal length={y.size}, invalid_starts={starts[invalid_targets].tolist()}, "
            f"ctx={ctx}, hor={hor}."
        )
    invalid_bounds = (starts + ctx - lag < 0) | (starts + ctx + hor - 1 > x.size)
    if invalid_bounds.any():
        raise ValueError(
            "Window boundaries are incompatible with the driving signal "
            "and requested lagged predictors. "
            f"Signal length={x.size}, invalid_starts={starts[invalid_bounds].tolist()}, "
            f"lag={lag}, ctx={ctx}, hor={hor}."
        )
    target, restricted, drivers = [], [], []
    for s in starts:
        t = int(s) + ctx + np.arange(hor)
        target.append(y[t])
        design_r = np.column_stack([np.ones(hor)] + [y[t-l] for l in range(1, lag+1)])
        design_x = np.column_stack([x[t-l] for l in range(1, lag+1)])
        for design, expected_shape in ((design_r, (hor, int(lag) + 1)),
                                       (design_x, (hor, int(lag)))):
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
        restricted.append(design_r)
        drivers.append(design_x)
    response = np.concatenate(target)
    r = np.vstack(restricted)
    u = np.column_stack([r, np.vstack(drivers)])
    if not np.isfinite(response).all():
        raise ValueError(
            "Non-finite values detected in target observations; "
            "the F-test requires finite target observations."
        )
    n = response.size
    df_den = n - (2 * lag + 1)
    if df_den <= 0:
        raise ValueError(
            "Cannot compute Granger-causality F-test: non-positive denominator "
            f"degrees of freedom (df_den={df_den}).\n"
            f"lag={lag}, ctx={ctx}, hor={hor}\n"
            f"starts={np.asarray(starts).tolist()}\n"
            f"n_windows={len(starts)}, obs={n}, k={lag}"
        )
    br, _, rank_r, _ = np.linalg.lstsq(r, response, rcond=None)
    bu, _, rank_u, _ = np.linalg.lstsq(u, response, rcond=None)
    if rank_r != r.shape[1] or rank_u != u.shape[1]:
        raise ValueError(f"Rank-deficient regression: restricted={rank_r}/{r.shape[1]}, "
                         f"full={rank_u}/{u.shape[1]}.")
    rss_r = float(np.sum((response - r @ br)**2))
    rss_u = float(np.sum((response - u @ bu)**2))
    if not np.isfinite(rss_u) or rss_u <= 0:
        raise ValueError(
            "Cannot compute Granger-causality F-test: RSS_full is zero or non-positive "
            "(or NaN), so the F-statistic is undefined.\n"
            f"lag={lag}, ctx={ctx}, hor={hor}\n"
            f"starts={np.asarray(starts).tolist()}\n"
            f"n_windows={len(starts)}, obs={n}, k={lag}, df_den={df_den}\n"
            f"rss_restricted={rss_r}, rss_full={rss_u}"
        )
    if not np.isfinite(rss_r):
        raise ValueError("Cannot compute Granger-causality F-test: RSS_restricted is non-finite.")
    improvement = rss_r - rss_u
    tolerance = 1e-10 * max(rss_r, rss_u, np.finfo(float).tiny)
    if improvement < -tolerance:
        raise ValueError("Full-model RSS exceeds restricted-model RSS beyond numerical tolerance.")
    f_stat = (max(0.0, improvement) / lag) / (rss_u / df_den)
    p_value = float(f_dist.sf(f_stat, lag, df_den))
    if not np.isfinite(f_stat) or not np.isfinite(p_value):
        raise ValueError("Non-finite F statistic or p-value.")
    return FTestResult(float(f_stat), p_value, int(n), int(lag), int(df_den), rss_r, rss_u)

def pair_tests(subject_id: str, group: str, x_name: str, y_name: str,
               x, y, starts, lags, fs: int = FS, ctx: int = CONTEXT_LEN,
               hor: int = HORIZON_LEN) -> list[dict]:
    """
    Run Granger F-tests for multiple lags for one subject and directed pair.

    For each lag, fit one restricted model and one full model to the pooled
    horizon observations. Return one result dictionary per lag, not per window.
    Signal loading, standardization and file output are handled by the caller.

    Parameters
    ----------
    subject_id : str
        Identifier of the subject being analyzed.
    group : str
        Participant group code or label.
    x_name : str
        Name of the driving channel X.
    y_name : str
        Name of the target channel Y.
    x : array_like
        Full driving-channel signal.
    y : array_like
        Full target-channel signal, aligned with X.
    starts : np.ndarray
        Context start indices, expressed in original signal samples.
    lags : iterable of int
        Model orders to test, expressed in samples.
    fs : int, default=FS
        Sampling frequency used to convert lag lengths to milliseconds.
    ctx : int, default=CONTEXT_LEN
        Number of historical context samples in each window.
    hor : int, default=HORIZON_LEN
        Number of target observations contributed by each window.

    Returns
    -------
    list[dict]
        One dictionary per tested lag: subject/group, channel identifiers,
        lag in samples and milliseconds, F-statistic, p-value, observation
        count, degrees of freedom, residual sums of squares and window sizes.
        Pipeline/scaling metadata are added separately by the runner.

    Raises
    ------
    ValueError
        If an individual Granger F-test cannot be computed.
    """
    rows = []
    for lag in lags:
        result = granger_f_test(x, y, starts, lag, ctx, hor)
        rows.append({
            "subject_id": subject_id, "group": group,
            "covariate_X": x_name, "target_Y": y_name,
            "lag_steps": int(lag), "lag_ms": float(lag * 1000 / fs),
            "f_stat_granger": result.f_stat, "p_value_granger": result.p_value,
            "model": "Granger", "n_obs": result.n_obs,
            "df_num": result.df_num, "df_den": result.df_den,
            "rss_restricted": result.rss_restricted, "rss_full": result.rss_full,
            "n_windows": len(starts), "context_len": ctx, "horizon_len": hor,
        })
    return rows

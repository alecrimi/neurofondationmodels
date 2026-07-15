"""
TimeGPT wrapper — Nixtla cloud API via nixtla pip package.
Point forecast. Requires TIMEGPT_API_KEY environment variable.

Set your key before running:
    $env:TIMEGPT_API_KEY = "your_key_here"   (PowerShell)
    export TIMEGPT_API_KEY=your_key_here      (bash)
"""
import os
import numpy as np
import pandas as pd


class TimeGPTWrapper:
    def __init__(self):
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        from nixtla import NixtlaClient
        api_key = os.environ.get("TIMEGPT_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "TIMEGPT_API_KEY environment variable not set. "
                "Get a key at https://dashboard.nixtla.io and set it with:\n"
                "  $env:TIMEGPT_API_KEY = 'your_key'  (PowerShell)"
            )
        self._client = NixtlaClient(api_key=api_key)

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        self._load()
        # Build a DataFrame with a 1-second frequency (matches EEG context)
        df = pd.DataFrame({
            "unique_id": ["eeg"] * len(context_np),
            "ds": pd.date_range("2020-01-01", periods=len(context_np), freq="s"),
            "y": context_np.astype(float),
        })
        forecast = self._client.forecast(
            df=df,
            h=horizon_len,
            freq="s",
            model="timegpt-1",
        )
        return forecast["TimeGPT"].values[:horizon_len]

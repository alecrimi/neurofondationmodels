"""
New pipeline — modular EEGPipeline approach with explicit quality checking
and full preprocessing (bandpass 0.5–40 Hz, notch 50 Hz, z-score).

Differences vs. BaselinePipeline:
  - QualityChecker: drops flatline (std < 1e-8 V) or saturated (std > 1e-3 V) channels
  - Bandpass + notch filter applied even though derivatives are already preprocessed
    (adds an extra safety pass; can be toggled off by subclassing if needed)
  - Otherwise identical windowing: 5 windows, context=512, horizon=64
"""
import numpy as np
from scipy.stats import zscore
from . import BasePipeline, CHANNELS
from benchmark.components.quality_checker import QualityChecker
from benchmark.components.preprocessor import DataPreprocessor


class NewPipeline(BasePipeline):
    name = "new"

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self._qc = QualityChecker()
        self._prep = DataPreprocessor(sampling_rate=500)

    def process(self, subject_id: str) -> dict:
        """
        Returns {channel: {'windows': [...], 'raw_std': float}}
        """
        raw_mne = self.loader.load_subject(subject_id)  # loads from derivatives

        result = {}
        for ch in CHANNELS:
            if ch not in raw_mne.ch_names:
                continue

            raw_sig = raw_mne.get_data(picks=[ch])[0]  # Volts
            raw_std = float(np.std(raw_sig))

            # Quality gate on raw signal
            if not self._qc.check(raw_sig, ch, subject_id):
                continue

            # Bandpass + notch + z-score
            processed = self._prep.process(raw_sig).astype(np.float32)

            windows = self._extract_windows(processed)
            if not windows:
                continue

            result[ch] = {
                'windows': windows,
                'raw_std': raw_std,
            }

        return result

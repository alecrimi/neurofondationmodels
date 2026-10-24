"""
Baseline pipeline — faithful replication of the reference benchmark from:
    neurofondationmodels/Michal/TSFMs_baseile_summary_report.md

Data:           ds004504/derivatives/ (preprocessed BIDS .set files)
Preprocessing:  z-score normalisation only (data already bandpass-filtered in derivatives)
Quality check:  none
Windows:        5 evenly spaced, context=512, horizon=64
Channels:       Fp1, Fp2, P3, P4
"""
import numpy as np
from scipy.stats import zscore
from . import BasePipeline, CHANNELS, FS


class BaselinePipeline(BasePipeline):
    name = "baseline"

    def process(self, subject_id: str) -> dict:
        """
        Returns {channel: {'windows': [...], 'raw_std': float}}
        raw_std is the physical std of the raw signal (in Volts) before z-score,
        used later to scale MSE back to Volts².
        """
        raw_mne = self.loader.load_subject(subject_id)  # loads from derivatives by default

        result = {}
        for ch in CHANNELS:
            if ch not in raw_mne.ch_names:
                continue

            raw_sig = raw_mne.get_data(picks=[ch])[0]  # shape (n_samples,), in Volts
            
            offset = int(3 * FS)
            reference = raw_sig[offset:]

            if reference.size < 2:
                raise ValueError(
                    f"{subject_id}/{ch}: signal too short after subtracting offset of {offset} samples to calculate std."
                )

            mean = float(np.mean(reference))
            raw_std = float(np.std(reference, ddof=0))

            if not np.isfinite(raw_std) or raw_std <= 0:
                raise ValueError(
                    f"{subject_id}/{ch}: incorrect standard deviation: {raw_std}."
                )

            normed = ((raw_sig - mean) / raw_std).astype(np.float32)

            windows = self._extract_windows(normed, subject_id=subject_id)
            if not windows:
                continue

            result[ch] = {
                'windows': windows,
                'raw_std': raw_std,
            }

        return result

"""
Pipeline registry and base class.

Each pipeline exposes:
    process(subject_id) -> dict[channel, {'windows': [...], 'raw_std': float}]

Windows list entries: {'context': np.array(512,), 'target': np.array(64,), 'start_idx': int}
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

CHANNELS = ["Fp1", "Fp2", "P3", "P4"]
CONTEXT_LEN = 512
HORIZON_LEN = 64
NUM_WINDOWS = 5
FS = 500


class AlzheimerLoader:
    """Minimal BIDS loader for ds004504 (Alzheimer / FTD / HC, 500 Hz)."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._participants = None

    @property
    def participants(self) -> pd.DataFrame:
        if self._participants is None:
            tsv = self.dataset_path / "participants.tsv"
            self._participants = pd.read_csv(tsv, sep="\t")
        return self._participants

    def _normalize_id(self, subject_id: str) -> str:
        s = str(subject_id)
        if not s.startswith("sub-"):
            s = f"sub-{s}"
        prefix, _, num = s.partition("-")
        if num.isdigit():
            s = f"sub-{int(num):03d}"
        return s

    def get_subject_metadata(self, subject_id: str) -> dict:
        sid = self._normalize_id(subject_id)
        row = self.participants[self.participants["participant_id"] == sid]
        return row.iloc[0].to_dict() if not row.empty else {}

    def load_subject(self, subject_id: str, use_derivatives: bool = True):
        import mne

        sid = self._normalize_id(subject_id)
        filename = f"{sid}_task-eyesclosed_eeg.set"

        if use_derivatives:
            path = (
                self.dataset_path / "derivatives" / sid / "eeg" / filename
            )
            source = "derivatives"
        else:
            path = self.dataset_path / sid / "eeg" / filename
            source = "raw"

        if not path.is_file():
            raise FileNotFoundError(
                f"Patient {sid}: required {source} EEG file is missing: "
                f"{path}; no alternative data source will be loaded."
            )

        return mne.io.read_raw_eeglab(
            str(path),
            preload=True,
            verbose=False,
        )


class BasePipeline:
    name: str = "base"

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._loader = None
        self._overlap_warned_subjects = set()

    @property
    def loader(self) -> AlzheimerLoader:
        if self._loader is None:
            self._loader = AlzheimerLoader(self.dataset_path)
        return self._loader

    def get_subjects(self, n: int | None = None) -> list:
        ids = self.loader.participants['participant_id'].tolist()
        return ids[:n] if n else ids

    def get_group(self, subject_id: str) -> str:
        meta = self.loader.get_subject_metadata(subject_id)
        return meta.get('Group', 'Unknown')

    def process(self, subject_id: str) -> dict:
        """Override in subclass. Returns {channel: {'windows': [...], 'raw_std': float}}"""
        raise NotImplementedError

    def _extract_windows(self, signal, *, subject_id: str) -> list:
        import numpy as np
        win_size = CONTEXT_LEN + HORIZON_LEN
        if len(signal) < win_size:
            return []
        max_start = len(signal) - win_size
        # Paper (Sec. III-C): the first 3 s of each recording are excluded to
        # avoid electrode-settling / amplifier onset artifacts, and NUM_WINDOWS
        # windows are then placed evenly across the remainder:
        #     np.linspace(3 * FS, max_start, NUM_WINDOWS)
        start_min = int(3 * FS)
        if start_min > max_start:
            # Recording too short to drop a full 3 s — fall back to the whole range.
            start_min = 0
        starts = np.linspace(start_min, max_start, NUM_WINDOWS).astype(int)

        gaps = np.diff(starts)

        if gaps.size:
            min_gap = int(gaps.min())
            window_overlap = max(0, win_size - min_gap)
            context_overlap = max(0, CONTEXT_LEN - min_gap)

            if (
                window_overlap > 0
                and subject_id not in self._overlap_warned_subjects
            ):
                extra = (
                    ", which exceeds the horizon length"
                    if context_overlap > HORIZON_LEN
                    else ""
                )

                warnings.warn(
                    f"Patient {subject_id}: maximum overlap between adjacent "
                    f"windows is {window_overlap} samples and between context "
                    f"windows is {context_overlap} samples{extra} "
                    f"(signal length: {len(signal)}, offset: {start_min}, "
                    f"window length: {win_size}, context length: {CONTEXT_LEN}, "
                    f"horizon length: {HORIZON_LEN}); processing continues.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._overlap_warned_subjects.add(subject_id)

        return [
            {
                'context': signal[s: s + CONTEXT_LEN],
                'target':  signal[s + CONTEXT_LEN: s + win_size],
                'start_idx': s,
            }
            for s in starts
        ]


def load_pipeline(name: str, dataset_path: str):
    if name == "baseline":
        from .baseline import BaselinePipeline
        return BaselinePipeline(dataset_path)
    elif name == "new":
        from .new_pipeline import NewPipeline
        return NewPipeline(dataset_path)
    elif name == "loreta":
        from .loreta import LoretaPipeline
        return LoretaPipeline(dataset_path)
    elif name == "loreta_gsp":
        from .loreta_gsp import LoretaGSPPipeline
        return LoretaGSPPipeline(dataset_path)
    else:
        raise ValueError(f"Unknown pipeline '{name}'. Available: {AVAILABLE_PIPELINES}")


AVAILABLE_PIPELINES = ["baseline", "new", "loreta", "loreta_gsp"]

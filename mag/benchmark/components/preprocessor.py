import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore


class DataPreprocessor:
    """
    Component for selecting channels, bandpass/notch filtering, and z-score scaling.
    Used by NewPipeline only (baseline uses zscore directly without filtering).
    """
    def __init__(self, sampling_rate=500):
        self.sampling_rate = float(sampling_rate)
        nyq = self.sampling_rate / 2.0
        # Butterworth bandpass 0.5–40 Hz, 4th order, zero-phase
        self._b_bp, self._a_bp = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
        # Notch at 50 Hz, Q=30
        self._b_notch, self._a_notch = iirnotch(50.0 / nyq, Q=30)

    def select_channels(self, raw_mne, channels):
        """
        Extracts raw 1D numpy signals for each specified channel.
        Returns:
            dict: {channel_name: 1D_numpy_array_in_Volts}
        """
        signals = {}
        for ch in channels:
            if ch in raw_mne.ch_names:
                signals[ch] = raw_mne.get_data(picks=[ch])[0]
            else:
                raise ValueError(f"Channel {ch} is missing in raw MNE data.")
        return signals

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies bandpass 0.5–40 Hz, notch 50 Hz, z-score normalization.
        """
        sig = filtfilt(self._b_bp, self._a_bp, signal)
        sig = filtfilt(self._b_notch, self._a_notch, sig)
        return zscore(sig).astype(np.float32)

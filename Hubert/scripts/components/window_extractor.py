import numpy as np

class WindowExtractor:
    """
    Component for slicing 1D processed signals into context and horizon windows.
    """
    def __init__(self, context_len=512, horizon_len=64, num_windows=5):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.num_windows = num_windows

    def extract(self, signal):
        """
        Extracts num_windows non-overlapping windows evenly spaced across the signal.
        Each window is split into context and horizon targets.
        
        Returns:
            list of dicts: [{'context': ..., 'target': ..., 'start_idx': ...}]
        """
        win_size = self.context_len + self.horizon_len
        n_samples = len(signal)
        if n_samples < win_size:
            return []
            
        max_start = n_samples - win_size
        starts = np.linspace(0, max_start, self.num_windows).astype(int)
        
        windows = []
        for s in starts:
            windows.append({
                'context': signal[s : s + self.context_len],
                'target': signal[s + self.context_len : s + win_size],
                'start_idx': s
            })
        return windows

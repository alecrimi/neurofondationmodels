import numpy as np

class QualityChecker:
    """
    Component for verifying signal quality of raw EEG channels (checks for flatlines or saturation).
    """
    def __init__(self, min_std=1e-8, max_std=1e-3, min_range=1e-7):
        self.min_std = min_std
        self.max_std = max_std
        self.min_range = min_range

    def check(self, raw_sig, ch_name, subject_id):
        """
        Checks raw signal for silent/flatline states or massive saturation artifacts.
        Returns:
            bool: True if signal is of acceptable quality, False otherwise.
        """
        std_val = np.std(raw_sig)
        max_val = np.max(raw_sig)
        min_val = np.min(raw_sig)
        
        # 1. Flatline / silent check (standard deviation in Volts)
        if std_val < self.min_std or (max_val - min_val) < self.min_range:
            print(f"  [!] Subject {subject_id} | Channel {ch_name} Q-Check FAILED: SILENT/FLATLINE (std={std_val:.2e} V).")
            return False
            
        # 2. Saturation check / extreme noise (std > 1mV)
        if std_val > self.max_std:
            print(f"  [!] Subject {subject_id} | Channel {ch_name} Q-Check FAILED: MASSIVE ARTIFACT/SATURATION (std={std_val:.2e} V).")
            return False
            
        return True

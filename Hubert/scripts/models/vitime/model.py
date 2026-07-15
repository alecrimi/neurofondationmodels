"""
ViTime wrapper — cloned from IkeYang/ViTime.
Probabilistic: 20 sample paths, returns median.

Requires pretrained checkpoint ViTime_Model.pth.
Download from Google Drive:
  https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view?usp=sharing
Place it at: new/models/vitime/ViTime_Model.pth
(or set VITIME_CHECKPOINT_PATH environment variable)

repo_dir should point to new/models/vitime/repo/ (the cloned repo root).
"""
import os
import sys
import numpy as np


_DEFAULT_CKPT_NAME = "ViTime_Model.pth"


class ViTimeWrapper:
    def __init__(self, device: str = "cpu", repo_dir: str = ""):
        self.device = device
        self.repo_dir = repo_dir
        self._vitime = None

    def _find_checkpoint(self) -> str:
        # 1. Environment variable
        env_path = os.environ.get("VITIME_CHECKPOINT_PATH", "")
        if env_path and os.path.isfile(env_path):
            return env_path
        # 2. Default location next to the repo dir
        default = os.path.join(os.path.dirname(self.repo_dir), _DEFAULT_CKPT_NAME)
        if os.path.isfile(default):
            return default
        raise FileNotFoundError(
            "ViTime checkpoint not found.\n"
            "Download ViTime_Model.pth from:\n"
            "  https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view?usp=sharing\n"
            f"and place it at: {default}\n"
            "Or set the VITIME_CHECKPOINT_PATH environment variable."
        )

    def _load(self):
        if self._vitime is not None:
            return

        if self.repo_dir and self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        ckpt_path = self._find_checkpoint()

        # Set the checkpoint path in config before importing ViTimePrediction
        import config as vitime_config
        vitime_config.VITIME_MODEL_PATH = ckpt_path

        from main import ViTimePrediction
        # tempature=8 is recommended for probabilistic sampling (per repo docs)
        self._vitime = ViTimePrediction(
            device=self.device,
            model_name="MAE",
            lookbackRatio=None,
            tempature=8,
        )

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        self._load()
        # prediction() returns (horizon_len, sampleNumber) for probabilistic
        samples = self._vitime.prediction(context_np, horizon_len, sampleNumber=20)
        return np.median(samples, axis=1)  # (horizon_len,)

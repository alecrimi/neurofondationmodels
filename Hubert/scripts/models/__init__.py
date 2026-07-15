"""
ModelRegistry — unified entry point for all TSFM wrappers.

Each model wrapper in new/models/<name>/model.py exposes:
    predict(context_np: np.ndarray, horizon_len: int) -> np.ndarray

Models are loaded lazily (on first predict call) and cached.
"""
import os
import numpy as np

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAMES = [
    "Chronos", "Chronos-2", "TimesFM", "Moirai",
    "Lag-Llama", "TimeGPT", "Sundial", "ViTime", "TimeFound",
]


class ModelRegistry:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._cache: dict = {}

    def predict(self, name: str, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name].predict(context_np, horizon_len)

    def _load(self, name: str):
        if name == "Chronos":
            from new.models.chronos.model import ChronosWrapper
            return ChronosWrapper(self.device, model_id="amazon/chronos-t5-small")

        elif name == "Chronos-2":
            from new.models.chronos2.model import Chronos2Wrapper
            return Chronos2Wrapper(self.device, model_id="amazon/chronos-bolt-base")

        elif name == "TimesFM":
            from new.models.timesfm.model import TimesFMWrapper
            return TimesFMWrapper(self.device)

        elif name == "Moirai":
            from new.models.moirai.model import MoiraiWrapper
            return MoiraiWrapper(self.device)

        elif name == "TimeGPT":
            from new.models.timegpt.model import TimeGPTWrapper
            return TimeGPTWrapper()

        elif name == "Lag-Llama":
            from new.models.lag_llama.model import LagLlamaWrapper
            repo_dir = os.path.join(_MODELS_DIR, "lag_llama", "repo")
            return LagLlamaWrapper(self.device, repo_dir=repo_dir)

        elif name == "Sundial":
            from new.models.sundial.model import SundialWrapper
            return SundialWrapper(self.device)

        elif name == "ViTime":
            from new.models.vitime.model import ViTimeWrapper
            repo_dir = os.path.join(_MODELS_DIR, "vitime", "repo")
            return ViTimeWrapper(self.device, repo_dir=repo_dir)

        elif name == "TimeFound":
            from new.models.timefound.model import TimeFoundWrapper
            return TimeFoundWrapper(self.device)

        else:
            raise ValueError(f"Unknown model name: '{name}'. Available: {MODEL_NAMES}")

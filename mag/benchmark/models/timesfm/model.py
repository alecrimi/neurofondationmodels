"""
TimesFM wrapper — google/timesfm-2.5-200m-pytorch.
Point forecast (deterministic).

Uses the TimesFM 2.5 API (TimesFM_2p5_200M_torch.from_pretrained).
Install the latest package from source if the PyPI version is too old:
    pip install git+https://github.com/google-research/timesfm.git

Note: newer huggingface_hub versions inject a `proxies` kwarg into __init__,
which TimesFM_2p5_200M_torch does not accept. We monkey-patch __init__ to
absorb it before loading.
"""
import numpy as np


def _patch_timesfm_init():
    """Make TimesFM_2p5_200M_torch.__init__ absorb unexpected hub kwargs.

    huggingface_hub injects extra kwargs (proxies, resume_download, token, …)
    into from_pretrained → __init__. We strip anything the real __init__
    doesn't accept by inspecting its signature.
    """
    import inspect
    try:
        import timesfm
        cls = timesfm.TimesFM_2p5_200M_torch
        if not getattr(cls, "_patched_init", False):
            _orig = cls.__init__
            _valid = set(inspect.signature(_orig).parameters) - {"self"}
            def _patched(self, *args, **kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k in _valid}
                _orig(self, *args, **kwargs)
            cls.__init__ = _patched
            cls._patched_init = True
    except Exception:
        pass


class TimesFMWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import timesfm
        _patch_timesfm_init()

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
            )
        )
        # Move to target device — TimesFM wraps a HuggingFace model so .to() works
        if self.device != "cpu" and hasattr(self._model, "to"):
            self._model.to(self.device)

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        self._load()
        point_forecast, _ = self._model.forecast(
            horizon=horizon_len,
            inputs=[context_np],
        )
        # point_forecast shape: (1, horizon_len)
        return np.array(point_forecast[0])[:horizon_len]

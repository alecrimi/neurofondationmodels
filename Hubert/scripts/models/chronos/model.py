"""
Chronos wrapper — amazon/chronos-t5-* via chronos-forecasting pip package.
Probabilistic: 20 sample paths, returns median.
"""
import numpy as np
import torch


class ChronosWrapper:
    def __init__(self, device: str = "cpu", model_id: str = "amazon/chronos-t5-small"):
        self.device = device
        self.model_id = model_id
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        from chronos import ChronosPipeline
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.float32,
        )

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        self._load()
        context = torch.tensor(context_np, dtype=torch.float32).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            forecast = self._pipeline.predict(
                inputs=context,
                prediction_length=horizon_len,
                num_samples=20,
            )  # (1, 20, horizon_len)
        return np.median(forecast[0].numpy(), axis=0)  # (horizon_len,)

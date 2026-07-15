"""
Chronos-2 wrapper — amazon/chronos-bolt-* (improved v2) via chronos-forecasting pip package.
Probabilistic: 20 sample paths, returns median.
"""
import numpy as np
import torch


class Chronos2Wrapper:
    def __init__(self, device: str = "cpu", model_id: str = "amazon/chronos-bolt-base"):
        self.device = device
        self.model_id = model_id
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            # Chronos-Bolt uses BaseChronosPipeline in newer versions of the library
            from chronos import BaseChronosPipeline
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        except ImportError:
            # Fallback: older chronos-forecasting uses ChronosPipeline for all variants
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
            # ChronosBoltPipeline.predict() takes `inputs`, not `context`, and
            # returns quantiles of shape (batch, num_quantiles, horizon) — no num_samples arg.
            forecast = self._pipeline.predict(
                inputs=context,
                prediction_length=horizon_len,
            )  # (1, num_quantiles, horizon_len)
        # Take median quantile (index 4 = 0.5 quantile out of 9 standard quantiles)
        num_q = forecast.shape[1]
        median_idx = num_q // 2
        return forecast[0, median_idx].numpy()  # (horizon_len,)

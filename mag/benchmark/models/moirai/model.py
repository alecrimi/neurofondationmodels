"""
Moirai wrapper — Salesforce/moirai-1.0-R-base via uni2ts pip package.
Probabilistic: 20 sample paths, returns median.

Uses the GluonTS predictor interface (the official documented way).
MoiraiForecast is a Lightning module — calling its forward() directly
causes positional embedding shape mismatches; use create_predictor() instead.
"""
import numpy as np
import pandas as pd


class MoiraiWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._predictor = None
        self._context_len = None
        self._horizon_len = None

    def _load(self, context_len: int, horizon_len: int):
        if (self._predictor is not None
                and self._context_len == context_len
                and self._horizon_len == horizon_len):
            return

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-base")
        if self.device != "cpu":
            module = module.to(self.device)

        model = MoiraiForecast(
            module=module,
            prediction_length=horizon_len,
            context_length=context_len,
            patch_size=32,
            num_samples=20,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = model.create_predictor(batch_size=1)
        self._context_len = context_len
        self._horizon_len = horizon_len

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        from gluonts.dataset.pandas import PandasDataset

        context_len = len(context_np)
        self._load(context_len, horizon_len)

        df = pd.DataFrame(
            {"target": context_np},
            index=pd.date_range("2020-01-01", periods=context_len, freq="s"),
        )
        ds = PandasDataset({"series": df}, target="target")
        forecasts = list(self._predictor.predict(ds))
        samples = forecasts[0].samples  # (20, horizon_len)
        return np.median(samples, axis=0)

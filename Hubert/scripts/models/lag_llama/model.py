"""
Lag-Llama wrapper — cloned from time-series-foundation-models/lag-llama.
Checkpoint downloaded automatically from HuggingFace on first use.
Probabilistic: 20 sample paths, returns median.

repo_dir should point to new/models/lag_llama/repo/ (the cloned repo root).
"""
import os
import sys
import numpy as np
import torch


class LagLlamaWrapper:
    def __init__(self, device: str = "cpu", repo_dir: str = ""):
        self.device = device
        self.repo_dir = repo_dir
        self._predictor = None
        self._dummy_freq = "h"

    @staticmethod
    def _patch_pandas22_aliases():
        """
        pandas 2.2+ removed legacy frequency aliases (Q, M, A/Y, BM, BQ, H, T, S).
        GluonTS still uses the old names, causing two failure modes:
          (a) to_offset("Q"/"M"/...) raises ValueError in pandas 2.2+
          (b) norm_freq_str returns the new name ("QE"/"ME"/...),
              which GluonTS's lag table doesn't recognise

        Fix in three steps:
          1. Patch pandas.tseries.frequencies.to_offset at the source.
          2. Patch GluonTS's norm_freq_str in both _base and lag modules.
          3. Patch the cached `to_offset` reference INSIDE gluonts modules.
             GluonTS does `from pandas.tseries.frequencies import to_offset` at
             import time, so Step 1 alone does not update gluonts's local binding.
             We must overwrite it module-by-module.
        """
        _OLD_TO_NEW = {
            "Q": "QE", "BQ": "BQE",
            "M": "ME", "BM": "BME", "CBM": "CBME",
            "Y": "YE", "A": "YE", "BA": "BYE", "BY": "BYE",
            "H": "h",
            "T": "min",
            "S": "s",
        }
        _NEW_TO_OLD = {v: k for k, v in _OLD_TO_NEW.items()}

        # ── Step 1: patch pandas source ─────────────────────────────────────
        import pandas.tseries.frequencies as _pfreq
        if not getattr(_pfreq.to_offset, "_pandas22_patched", False):
            _orig_to_offset = _pfreq.to_offset
            def _compat_to_offset(freq, **kw):
                if isinstance(freq, str) and freq in _OLD_TO_NEW:
                    freq = _OLD_TO_NEW[freq]
                return _orig_to_offset(freq, **kw)
            _compat_to_offset._pandas22_patched = True
            _pfreq.to_offset = _compat_to_offset
        _compat_to_offset = _pfreq.to_offset  # always retrieve (may already be patched)

        # ── Step 2: patch norm_freq_str in gluonts ──────────────────────────
        # IMPORTANT: _compat_norm must NOT call _orig_norm for known old-style
        # aliases (M, Q, H, …) because _orig_norm internally calls to_offset,
        # and that module-level binding may still point to the original pandas
        # function even after Step 1/3 patches run.  Short-circuit for known
        # strings to avoid the failing call entirely.
        try:
            import gluonts.time_feature._base as _gtf_base
            import gluonts.time_feature.lag as _gtf_lag

            _orig_norm = _gtf_base.norm_freq_str
            if not getattr(_orig_norm, "_pandas22_patched", False):
                def _compat_norm(freq_str: str) -> str:
                    # Old-style alias → return as-is (GluonTS expects old names)
                    if freq_str in _OLD_TO_NEW:
                        return freq_str
                    # New-style alias → convert back to old-style
                    if freq_str in _NEW_TO_OLD:
                        return _NEW_TO_OLD[freq_str]
                    # Unknown string — try the original, catch pandas 2.2 errors
                    try:
                        result = _orig_norm(freq_str)
                        return _NEW_TO_OLD.get(result, result)
                    except Exception:
                        return freq_str
                _compat_norm._pandas22_patched = True
                _gtf_base.norm_freq_str = _compat_norm
                _gtf_lag.norm_freq_str = _compat_norm
        except Exception:
            pass

        # ── Step 3: overwrite cached to_offset in ALL loaded modules ─────────
        # Any module that did `from pandas.tseries.frequencies import to_offset`
        # at import time holds its own reference to the original function.
        # Scan every loaded module and replace the attribute wherever found.
        import sys as _sys
        for _mod in list(_sys.modules.values()):
            try:
                if hasattr(_mod, "to_offset") and not getattr(
                    _mod.to_offset, "_pandas22_patched", False
                ):
                    _mod.to_offset = _compat_to_offset
            except Exception:
                pass

    def _load(self, context_len: int, horizon_len: int):
        if self._predictor is not None:
            return

        self._patch_pandas22_aliases()

        # Add the cloned repo to the path so lag_llama imports work
        if self.repo_dir and self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        # PyTorch 2.6+ defaults weights_only=True, which blocks the many GluonTS
        # classes serialised in the Lag-Llama checkpoint. Force weights_only=False
        # (the pre-2.6 default) — checkpoint is from a trusted HuggingFace source.
        try:
            import torch.serialization as _torch_serial
            _real_load = _torch_serial.load
            if not getattr(_real_load, "_lagllama_patched", False):
                def _permissive_load(*args, **kwargs):
                    kwargs["weights_only"] = False  # force, don't just default
                    return _real_load(*args, **kwargs)
                _permissive_load._lagllama_patched = True
                _torch_serial.load = _permissive_load
                torch.load = _permissive_load
        except Exception:
            pass

        from huggingface_hub import hf_hub_download
        from lag_llama.gluon.estimator import LagLlamaEstimator

        ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
        )

        # ── Read architecture hyperparameters from the Lightning checkpoint ──
        # Hardcoding is fragile — the checkpoint may differ from any tutorial.
        _raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _hp = _raw.get("hyper_parameters", {})
        _sd = _raw.get("state_dict", {})

        n_layer        = _hp.get("n_layer", 8)
        n_head         = _hp.get("n_head", 4)
        lags_seq       = _hp.get("lags_seq", None)
        # time_feat=True adds 6 time features to the model input.
        # The released checkpoint was trained with time_feat=True, which gives
        # feature_size = len(lags_seq)*input_size + 2*input_size + 6 = 92.
        # Without it the model builds feature_size=86 → wte mismatch.
        time_feat      = _hp.get("time_feat", True)

        # n_embd_per_head from HP may be wrong (HP says 32 → n_embd=128, but
        # the weights show n_embd=144). Derive the real value from wte.weight.
        _wte_key = next((k for k in _sd if k.endswith("wte.weight")), None)
        if _wte_key is not None:
            _n_embd_actual = _sd[_wte_key].shape[0]
            n_embd_per_head = _n_embd_actual // n_head  # e.g. 144 // 4 = 36
        else:
            n_embd_per_head = _hp.get("n_embd_per_head", 32)

        # Build extra estimator kwargs from what the checkpoint stores.
        # lags_seq from the HP is a list of frequency strings (e.g. ["Q","M","W","D","H","T","S"]).
        # If absent, the estimator default is used — which gives the right 84 unique lags.
        extra_kwargs = {"time_feat": time_feat}
        if lags_seq is not None:
            extra_kwargs["lags_seq"] = lags_seq

        dummy_freq = "h"

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon_len,
            context_length=context_len,
            n_layer=n_layer,
            n_embd_per_head=n_embd_per_head,
            n_head=n_head,
            batch_size=1,
            num_parallel_samples=20,
            aug_prob=0,
            device=torch.device(self.device),
            **extra_kwargs,
        )

        # Load the checkpoint without any training — ckpt_path is already set,
        # so create_lightning_module() calls load_from_checkpoint() internally.
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self._predictor = estimator.create_predictor(transformation, lightning_module)
        self._context_len = context_len
        self._horizon_len = horizon_len
        self._dummy_freq = dummy_freq  # must match when building predict dataset

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        context_len = len(context_np)
        self._load(context_len, horizon_len)

        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset

        freq = getattr(self, "_dummy_freq", "s")
        df = pd.DataFrame(
            {"target": context_np},
            index=pd.date_range("2020-01-01", periods=context_len, freq=freq),
        )
        ds = PandasDataset({"ts": df}, target="target", freq=freq)

        forecasts = list(self._predictor.predict(ds))
        # forecasts[0].samples shape: (20, horizon_len)
        samples = forecasts[0].samples
        return np.median(samples, axis=0)  # (horizon_len,)

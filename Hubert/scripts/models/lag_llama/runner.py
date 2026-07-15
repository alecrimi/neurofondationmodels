"""
Lag-Llama isolated runner.
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API: LagLlamaEstimator from ref/lag-llama/ repo
Source:      ref/lag-llama/ added to sys.path + gluonts>=0.15.0 (new API)
Checkpoint:  downloaded automatically from HuggingFace on first run

NOTE: pandas 2.x compatibility is handled via _patch_pandas22() below.
"""
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

# Add reference/lag-llama/ to sys.path so we can import lag_llama directly
# Runner lives at benchmark/models/lag_llama/runner.py → parents[3] = mag/
_REF_LAG_LLAMA = Path(__file__).resolve().parents[3] / "reference" / "lag-llama"
if str(_REF_LAG_LLAMA) not in sys.path:
    sys.path.insert(0, str(_REF_LAG_LLAMA))


def _patch_gluonts_compat():
    """
    gluonts>=0.15.0 removed gluonts.torch.modules.loss.
    Lag-Llama's estimator.py imports DistributionLoss/NegativeLogLikelihood at
    module load time (line 33), but they're training-only — never called during
    inference.  We inject only the .loss stub; we do NOT touch the parent
    gluonts.torch.modules package so gluonts can still load its own submodules
    (e.g. lambda_layer) through the real package machinery.
    """
    import types

    if "gluonts.torch.modules.loss" not in sys.modules:
        class DistributionLoss:
            pass

        class NegativeLogLikelihood(DistributionLoss):
            pass

        loss_mod = types.ModuleType("gluonts.torch.modules.loss")
        loss_mod.DistributionLoss = DistributionLoss
        loss_mod.NegativeLogLikelihood = NegativeLogLikelihood
        sys.modules["gluonts.torch.modules.loss"] = loss_mod


def _patch_pandas22():
    """Fix pandas 2.2+ removed frequency aliases that old GluonTS still uses."""
    _OLD_NEW = {
        "Q": "QE", "BQ": "BQE", "M": "ME", "BM": "BME",
        "Y": "YE", "A": "YE", "BA": "BYE", "BY": "BYE",
        "H": "h", "T": "min", "S": "s",
    }
    _NEW_OLD = {v: k for k, v in _OLD_NEW.items()}

    import pandas.tseries.frequencies as _pfreq
    if getattr(_pfreq.to_offset, "_patched22", False):
        return
    _orig = _pfreq.to_offset

    def _compat(freq, **kw):
        if isinstance(freq, str) and freq in _OLD_NEW:
            freq = _OLD_NEW[freq]
        return _orig(freq, **kw)

    _compat._patched22 = True
    _pfreq.to_offset = _compat

    try:
        import gluonts.time_feature._base as _b
        import gluonts.time_feature.lag as _l
        _orig_n = _b.norm_freq_str
        if not getattr(_orig_n, "_patched22", False):
            def _cn(s):
                if s in _OLD_NEW:
                    return s
                if s in _NEW_OLD:
                    return _NEW_OLD[s]
                try:
                    r = _orig_n(s)
                    return _NEW_OLD.get(r, r)
                except Exception:
                    return s
            _cn._patched22 = True
            _b.norm_freq_str = _cn
            _l.norm_freq_str = _cn
    except Exception:
        pass

    for mod in list(sys.modules.values()):
        try:
            if hasattr(mod, "to_offset") and not getattr(mod.to_offset, "_patched22", False):
                mod.to_offset = _compat
        except Exception:
            pass


def _patch_torch_load():
    """torch >= 2.4 defaults weights_only=True which breaks GluonTS checkpoint."""
    import torch.serialization as _ts
    if getattr(_ts.load, "_wonly_patched", False):
        return
    _real = _ts.load

    def _permissive(*a, **kw):
        kw["weights_only"] = False
        return _real(*a, **kw)

    _permissive._wonly_patched = True
    _ts.load = _permissive
    torch.load = _permissive


def build_predictor(context_len: int, horizon_len: int, device: str):
    _patch_gluonts_compat()   # must come before any lag_llama import
    _patch_pandas22()
    _patch_torch_load()

    from huggingface_hub import hf_hub_download
    from lag_llama.gluon.estimator import LagLlamaEstimator

    ckpt_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt",
    )
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = raw_ckpt.get("hyper_parameters", {})
    sd = raw_ckpt.get("state_dict", {})

    n_layer  = hp.get("n_layer", 8)
    n_head   = hp.get("n_head", 4)
    lags_seq = hp.get("lags_seq", None)
    time_feat = hp.get("time_feat", True)

    _wte_key = next((k for k in sd if k.endswith("wte.weight")), None)
    n_embd_per_head = sd[_wte_key].shape[0] // n_head if _wte_key else hp.get("n_embd_per_head", 32)

    extra = {"time_feat": time_feat}
    if lags_seq is not None:
        extra["lags_seq"] = lags_seq

    est = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=horizon_len,
        context_length=context_len,
        n_layer=n_layer,
        n_embd_per_head=n_embd_per_head,
        n_head=n_head,
        batch_size=1,
        num_parallel_samples=20,
        aug_prob=0,
        device=torch.device(device),
        **extra,
    )
    lm = est.create_lightning_module()
    tr = est.create_transformation()
    return est.create_predictor(tr, lm)


def predict_window(predictor, context_np: np.ndarray, horizon_len: int) -> np.ndarray:
    import pandas as pd
    from gluonts.dataset.pandas import PandasDataset

    ctx_len = len(context_np)
    df = pd.DataFrame(
        {"target": context_np},
        index=pd.date_range("2020-01-01", periods=ctx_len, freq="h"),
    )
    ds = PandasDataset({"ts": df}, target="target", freq="h")
    forecasts = list(predictor.predict(ds))
    return np.median(forecasts[0].samples, axis=0)  # (H,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    device      = data["device"]
    horizon_len = data["horizon_len"]
    subjects    = data["subjects"]

    print(f"[Lag-Llama] Building predictor on {device} …")
    predictor = build_predictor(
        context_len=512,
        horizon_len=horizon_len,
        device=device,
    )
    print("[Lag-Llama] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[Lag-Llama] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                pred = predict_window(predictor, win["context"].astype(np.float32), horizon_len)
                preds.append(pred)
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[Lag-Llama] Done.")


if __name__ == "__main__":
    main()

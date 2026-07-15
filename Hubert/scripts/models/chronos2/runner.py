"""
Chronos-2 isolated runner (amazon/chronos-bolt-base).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API: BaseChronosPipeline.from_pretrained() (falls back to ChronosPipeline)
Source:      pip install chronos-forecasting>=2.0.0
"""
import argparse
import pickle
import numpy as np
import torch

# chronos-forecasting>=2.0.0 exports BaseChronosPipeline
try:
    from chronos import BaseChronosPipeline as _Pipeline
    _MODEL_ID = "amazon/chronos-bolt-base"
    _USE_BASE = True
except ImportError:
    from chronos import ChronosPipeline as _Pipeline   # type: ignore[assignment]
    _MODEL_ID = "amazon/chronos-t5-base"
    _USE_BASE = False


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

    print(f"[Chronos-2] Loading {_MODEL_ID} on {device} …")
    pipeline = _Pipeline.from_pretrained(
        _MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
    )
    print("[Chronos-2] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[Chronos-2] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                ctx = torch.tensor(win["context"], dtype=torch.float32).unsqueeze(0)
                if _USE_BASE:
                    # ChronosBoltPipeline: quantile model, no num_samples param.
                    # Returns (batch, n_quantiles, H) where quantiles = [0.1..0.9].
                    # Index 4 (0-based, out of 9) = p50 = median.
                    forecast = pipeline.predict(
                        inputs=ctx,
                        prediction_length=horizon_len,
                    )
                    arr  = forecast[0].numpy()   # (9, H)
                    pred = arr[arr.shape[0] // 2]  # p50
                else:
                    # Legacy ChronosPipeline: sample-based
                    forecast = pipeline.predict(
                        inputs=ctx,
                        prediction_length=horizon_len,
                        num_samples=20,
                    )
                    arr  = forecast[0].numpy()   # (num_samples, H)
                    pred = np.median(arr, axis=0)
                preds.append(pred)
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[Chronos-2] Done.")


if __name__ == "__main__":
    main()

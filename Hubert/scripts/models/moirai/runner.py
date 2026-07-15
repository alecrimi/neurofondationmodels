"""
Moirai isolated runner (Salesforce/moirai-1.0-R-base via uni2ts).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API: MoiraiModule.from_pretrained() + MoiraiForecast
Source:      pip install gluonts>=0.15.0 + pip install uni2ts --no-deps
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


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

    print("[Moirai] Loading Salesforce/moirai-1.0-R-base …")
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-base")

    # Build predictor once; we'll reuse it for all windows
    # context_length is fixed at 512 (matches our window extractor)
    context_length = 512

    model = MoiraiForecast(
        module=module,
        prediction_length=horizon_len,
        context_length=context_length,
        patch_size=32,
        num_samples=20,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=1)
    print("[Moirai] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[Moirai] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                ctx = win["context"].astype(np.float32)
                ctx_len = len(ctx)
                df = pd.DataFrame(
                    {"target": ctx},
                    index=pd.date_range("2020-01-01", periods=ctx_len, freq="h"),
                )
                ds = PandasDataset({"ts": df}, target="target", freq="h")
                forecasts = list(predictor.predict(ds))
                pred = np.median(forecasts[0].samples, axis=0)
                preds.append(pred)
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[Moirai] Done.")


if __name__ == "__main__":
    main()

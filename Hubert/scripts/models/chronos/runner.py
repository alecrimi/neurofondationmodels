"""
Chronos isolated runner (amazon/chronos-t5-small).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API: ChronosPipeline.from_pretrained()
Source:      pip install chronos-forecasting
"""
import argparse
import pickle
import numpy as np
import torch
from chronos import ChronosPipeline


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

    print(f"[Chronos] Loading model on {device} …")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch.float32,
    )
    print("[Chronos] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[Chronos] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                ctx = torch.tensor(win["context"], dtype=torch.float32).unsqueeze(0)
                forecast = pipeline.predict(
                    inputs=ctx,
                    prediction_length=horizon_len,
                    num_samples=20,
                )
                # forecast shape: (1, num_samples, horizon_len)
                preds.append(np.median(forecast[0].numpy(), axis=0))
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),   # (n_win, H)
                "targets":     np.array(targets),  # (n_win, H)
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[Chronos] Done.")


if __name__ == "__main__":
    main()

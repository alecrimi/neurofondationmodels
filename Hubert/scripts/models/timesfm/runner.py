"""
TimesFM isolated runner (google/timesfm-2.5-200m-pytorch).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API: timesfm.TimesFM_2p5_200M_torch.from_pretrained()
Source:      pip install timesfm>=2.0.0
"""
import argparse
import pickle
import numpy as np
import timesfm


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

    print(f"[TimesFM] Loading model on {device} …")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    model.compile(timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
    ))
    print("[TimesFM] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[TimesFM] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                ctx = win["context"].astype(np.float32)
                point_forecast, _ = model.forecast(
                    horizon=horizon_len,
                    inputs=[ctx],
                )
                pred = np.array(point_forecast[0])[:horizon_len]
                preds.append(pred)
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[TimesFM] Done.")


if __name__ == "__main__":
    main()

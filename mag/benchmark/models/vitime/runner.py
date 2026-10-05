"""
ViTime isolated runner (IkeYang/ViTime).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

Creator API (from README):
    vitime = ViTimePrediction(device='cuda:0', model_name='MAE', lookbackRatio=None, tempature=8)
    samples = vitime.prediction(xData, prediction_length, sampleNumber=100)

Source:      ref/vitime/ repo (added to sys.path)
Checkpoint:  new/models/vitime/ViTime_Model.pth
             Download: https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view
"""
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np

# Add reference/vitime/ to sys.path (creator's code)
# runner.py lives at benchmark/models/vitime/ → parents[3] = mag/
_REF_VITIME = Path(__file__).resolve().parents[3] / "reference" / "vitime"
_CKPT_PATH  = Path(__file__).resolve().parent / "ViTime_Model.pth"

if not _CKPT_PATH.exists():
    # Fallback: try ref/vitime/ for checkpoint
    _CKPT_PATH = _REF_VITIME / "ViTime_Model.pth"

if str(_REF_VITIME) not in sys.path:
    sys.path.insert(0, str(_REF_VITIME))


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

    if not _CKPT_PATH.exists():
        raise FileNotFoundError(
            f"ViTime checkpoint not found at {_CKPT_PATH}\n"
            "Download from: https://drive.google.com/file/d/1ex5ZrIKhsnLj2EuUkP9We3Bpcr1kVh5d/view\n"
            f"and place at: {Path(__file__).resolve().parent / 'ViTime_Model.pth'}"
        )

    # Point config at the checkpoint before importing ViTimePrediction
    try:
        import config as vitime_config
        vitime_config.VITIME_MODEL_PATH = str(_CKPT_PATH)
    except (ImportError, AttributeError):
        pass  # Newer ViTime versions may not use a separate config

    from main import ViTimePrediction  # from ref/vitime/

    # ViTimePrediction expects 'cuda:0' format, not bare 'cuda'
    vt_device = "cuda:0" if device == "cuda" else device
    print(f"[ViTime] Loading model on {vt_device} …")
    vitime = ViTimePrediction(
        device=vt_device,
        model_name="MAE",
        lookbackRatio=None,
        tempature=8,
    )
    print("[ViTime] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[ViTime] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                xData = win["context"].astype(np.float32)
                # samples shape: (horizon_len, sampleNumber)
                samples = vitime.prediction(xData, horizon_len, sampleNumber=20)
                pred = np.median(samples, axis=1)  # (horizon_len,)
                preds.append(pred)
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[ViTime] Done.")


if __name__ == "__main__":
    main()

"""
TimeFound isolated runner — encoder-decoder Transformer proxy (arXiv 2503.04118, Baidu Research).
Receives input.pkl, writes output.pkl, exits.
No EEG loading — data arrives pre-processed via pickle.

NOTE: No official code or weights exist. This is a faithful reimplementation
      of the architecture described in the paper (random initialisation).
"""
import argparse
import math
import pickle
import numpy as np
import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeFoundNet(nn.Module):
    """Encoder-decoder Transformer proxy from the paper."""
    def __init__(self, d_model: int = 128, nhead: int = 8, n_enc: int = 3, n_dec: int = 3):
        super().__init__()
        self.proj_in    = nn.Linear(1, d_model)
        self.pos_enc    = _PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.proj_out = nn.Linear(d_model, 1)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        m = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return m.masked_fill(m == 1, float("-inf"))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_e = self.pos_enc(self.proj_in(src))
        tgt_e = self.pos_enc(self.proj_in(tgt))
        out   = self.transformer(
            src_e, tgt_e,
            tgt_mask=self._causal_mask(tgt.size(1), tgt.device),
        )
        return self.proj_out(out)


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

    print("[TimeFound] Building model (random weights — no official checkpoint) …")
    torch.manual_seed(42)
    model = TimeFoundNet().to(device)
    model.eval()
    print("[TimeFound] Model ready.")

    results = {}
    n_subj  = len(subjects)
    for i, (subj_id, subj_data) in enumerate(subjects.items()):
        group = subj_data["group"]
        print(f"[TimeFound] Subject {i+1}/{n_subj}: {subj_id} ({group})")
        results[subj_id] = {"group": group}
        for ch, ch_data in subj_data.items():
            if ch == "group":
                continue
            preds   = []
            targets = []
            for win in ch_data["windows"]:
                ctx = win["context"].astype(np.float32)
                mu  = float(ctx.mean())
                sig = float(ctx.std()) + 1e-8
                ctx_norm = (ctx - mu) / sig

                src = torch.tensor(ctx_norm).unsqueeze(0).unsqueeze(-1).to(device)
                tgt = torch.zeros(1, horizon_len, 1, device=device)
                with torch.no_grad():
                    out = model(src, tgt)
                pred = out[0, :, 0].cpu().numpy() * sig + mu
                preds.append(pred)
                targets.append(win["target"])
            results[subj_id][ch] = {
                "predictions": np.array(preds),
                "targets":     np.array(targets),
                "raw_std":     ch_data["raw_std"],
            }

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print("[TimeFound] Done.")


if __name__ == "__main__":
    main()

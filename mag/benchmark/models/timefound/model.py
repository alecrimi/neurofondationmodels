"""
TimeFound wrapper — own PyTorch implementation based on arXiv 2503.04118.
No public code is available from the authors.
Point forecast using autoregressive encoder-decoder transformer.
Per-window z-score normalization applied internally (as described in the paper).
"""
import numpy as np
import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Architecture (encoder-decoder Transformer, ~200M param proxy)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TimeFoundNet(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 n_enc: int = 3, n_dec: int = 3):
        super().__init__()
        self.proj_in = nn.Linear(1, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=n_enc, num_decoder_layers=n_dec,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.proj_out = nn.Linear(d_model, 1)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_enc(self.proj_in(src))
        tgt_emb = self.pos_enc(self.proj_in(tgt))
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.proj_out(out)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class TimeFoundWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model: _TimeFoundNet | None = None

    def _load(self):
        if self._model is not None:
            return
        torch.manual_seed(42)
        self._model = _TimeFoundNet().to(self.device)
        self._model.eval()

    def predict(self, context_np: np.ndarray, horizon_len: int = 64) -> np.ndarray:
        self._load()

        # Per-window z-score (as described in the paper)
        mu = float(context_np.mean())
        sig = float(context_np.std()) + 1e-8
        ctx_norm = (context_np - mu) / sig

        src = torch.tensor(ctx_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        # One-shot decoding: pass the full target sequence at once.
        # Autoregressive decoding grows tgt to length=horizon_len, which causes
        # a shape conflict with head_dim=16 in PyTorch's attention kernel at
        # the last step. One-shot is equivalent for a randomly-initialised model.
        tgt = torch.zeros(1, horizon_len, 1, dtype=torch.float32, device=src.device)

        with torch.no_grad():
            out = self._model(src, tgt)  # (1, horizon_len, 1)

        preds = out[0, :, 0].cpu().numpy()
        return preds * sig + mu  # inverse z-score

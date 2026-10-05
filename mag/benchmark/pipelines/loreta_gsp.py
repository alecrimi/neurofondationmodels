"""
LORETA-GSP pipeline — sLORETA source parcels projected onto network harmonics.

Scientific rationale
--------------------
Graph Signal Processing (GSP) decomposes brain activity into eigenvectors of the
structural connectome's graph Laplacian — called *network harmonics*.  Low-index
harmonics (small eigenvalue) represent spatially smooth, structure-coupled activity;
high-index harmonics represent spatially discontinuous, structure-decoupled activity.

Projecting EEG source signals onto network harmonics allows testing whether TSFMs
can capture structure-function coupling dynamics differently across diagnostic groups
(Alzheimer / FTD / Healthy Controls).

Reference: de Wouters et al. (2026) "Source-reconstructed EEG graph signal
processing: pitfalls and workarounds", Network Neuroscience.

Implementation
--------------
1. Run the full sLORETA pipeline (via LoretaPipeline._process_parcels) to get
   z-scored parcel time series for up to 12 ROIs (6 parcels × 2 hemispheres).
2. Build the Euclidean-distance connectome from hard-coded MNI centroid coordinates
   of the Desikan-Killiany parcels used in the loreta pipeline.
3. Compute the normalised graph Laplacian: L = I - D^(-1/2) A D^(-1/2)
4. Eigendecompose L → eigenvectors Q  (columns = network harmonics, ordered by
   eigenvalue from low to high spatial frequency).
5. Project parcel signals onto harmonics: H = Q^T @ X  where X is (N_parcels, T).
6. Z-score each harmonic signal, extract windows — same as other pipelines.

Output channel names: harm_00 … harm_11  (index = harmonic order, 00 = lowest
spatial frequency = most structure-coupled activity).

Note on raw_std
---------------
Input parcel signals are already z-scored, so harmonic signals have std ≈ 1.
raw_std is the std of the *projected* (pre-z-score) harmonic signal, capturing
how much of the total signal variance lives in that harmonic.  mse_phys ≈ mse_norm
for all harmonics (both are dimensionless), but raw_std still reflects the relative
harmonic power distribution.

Usage
-----
    python benchmark/run.py --pipeline loreta_gsp --n 3

Connectome choice
-----------------
A Euclidean-distance connectome is used rather than diffusion MRI because ds004504
does not include structural imaging.  The paper (de Wouters et al.) showed that
Euclidean-distance connectomes give results similar to structural connectomes during
baseline resting-state conditions (their Figure 5 / 6).

sigma parameter: 60 mm, chosen as ≈ half the typical inter-region distance in the
12-node graph covering frontal, temporal, and parietal cortex bilaterally.
"""

import numpy as np
from scipy.stats import zscore

from . import BasePipeline
from .loreta import LoretaPipeline, PARCELS, HEMIS

# ── MNI centroid coordinates for the 12 parcels (Desikan-Killiany, MNI152) ────
# Source: Desikan et al. (2006), cross-validated with FreeSurfer label centroids.
# Coordinates in millimetres: (x, y, z).
PARCEL_MNI = {
    "src_precuneus_lh":            np.array([ -7, -58,  48], dtype=float),
    "src_precuneus_rh":            np.array([  7, -58,  48], dtype=float),
    "src_inferiorparietal_lh":     np.array([-47, -66,  30], dtype=float),
    "src_inferiorparietal_rh":     np.array([ 47, -66,  30], dtype=float),
    "src_superiortemporal_lh":     np.array([-58, -20,   2], dtype=float),
    "src_superiortemporal_rh":     np.array([ 58, -20,   2], dtype=float),
    "src_middletemporal_lh":       np.array([-65, -26,  -8], dtype=float),
    "src_middletemporal_rh":       np.array([ 65, -26,  -8], dtype=float),
    "src_superiorfrontal_lh":      np.array([-14,  32,  52], dtype=float),
    "src_superiorfrontal_rh":      np.array([ 14,  32,  52], dtype=float),
    "src_rostralmiddlefrontal_lh": np.array([-38,  47,  20], dtype=float),
    "src_rostralmiddlefrontal_rh": np.array([ 38,  47,  20], dtype=float),
}

# Gaussian kernel width for the Euclidean connectome (in mm)
_SIGMA_MM = 60.0


def _build_euclidean_laplacian(parcel_keys: list) -> np.ndarray:
    """
    Build the normalised graph Laplacian from Euclidean distances between
    MNI centroid coordinates.

    Parameters
    ----------
    parcel_keys : list of str
        Ordered list of parcel channel names (must be keys in PARCEL_MNI).

    Returns
    -------
    L : np.ndarray, shape (N, N)
        Symmetric normalised Laplacian.  Eigenvalues ∈ [0, 2].
    """
    n = len(parcel_keys)
    coords = np.array([PARCEL_MNI[k] for k in parcel_keys])  # (N, 3)

    # Gaussian adjacency: A[i,j] = exp(-||r_i - r_j||² / (2σ²)), zero diagonal
    diff = coords[:, None, :] - coords[None, :, :]    # (N, N, 3)
    dist2 = np.sum(diff ** 2, axis=-1)                # (N, N)
    A = np.exp(-dist2 / (2.0 * _SIGMA_MM ** 2))
    np.fill_diagonal(A, 0.0)

    # Degree matrix
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalised Laplacian L = I - D^(-1/2) A D^(-1/2)
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


class LoretaGSPPipeline(LoretaPipeline):
    """
    EEG → sLORETA parcels → graph Fourier transform → TSFM forecasting.

    Inherits all sLORETA machinery from LoretaPipeline.
    Overrides process() to add the GSP projection step.
    """
    name = "loreta_gsp"

    def process(self, subject_id: str) -> dict:
        """
        Returns {harm_<kk>: {'windows': [...], 'raw_std': float}}.

        Steps:
          sLORETA parcels → Euclidean connectome → graph Laplacian →
          eigendecomposition → project onto harmonics → z-score → windows
        """
        # ── 1. Get z-scored parcel time series (no windowing) ─────────────────
        parcels = self._process_parcels(subject_id)
        if not parcels:
            return {}

        # Keep only parcels whose MNI coords are known (all 12 should be present)
        keys = [k for k in parcels if k in PARCEL_MNI]
        if len(keys) < 2:
            print(f"  [!] LORETA-GSP: too few known parcels for {subject_id} "
                  f"({len(keys)} found). Skipping.")
            return {}

        # ── 2. Stack into signal matrix X: (N_parcels, T) ─────────────────────
        X = np.stack([parcels[k][0].astype(np.float64) for k in keys], axis=0)

        # ── 3. Build Euclidean-distance graph Laplacian ───────────────────────
        L = _build_euclidean_laplacian(keys)

        # ── 4. Eigendecompose: Q columns = network harmonics ──────────────────
        # eigh is correct here: L is real symmetric.
        # Returns eigenvalues in ascending order (→ columns of Q ordered from
        # lowest to highest spatial frequency).
        _eigenvalues, Q = np.linalg.eigh(L)   # Q: (N, N)

        # ── 5. Project: H = Q^T @ X  → (N, T) ────────────────────────────────
        H = Q.T @ X   # each row k is the k-th harmonic time series

        # ── 6. Z-score each harmonic and extract windows ──────────────────────
        result = {}
        for k in range(len(keys)):
            h_sig = H[k]
            raw_std = float(np.std(h_sig))
            if raw_std < 1e-12:
                continue   # degenerate harmonic (numerically zero) — skip

            normed = zscore(h_sig).astype(np.float32)
            windows = self._extract_windows(normed)
            if not windows:
                continue

            result[f"harm_{k:02d}"] = {"windows": windows, "raw_std": raw_std}

        if not result:
            print(f"  [!] LORETA-GSP: no valid harmonic signals for {subject_id}.")

        return result

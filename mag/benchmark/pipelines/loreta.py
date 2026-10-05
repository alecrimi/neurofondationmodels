"""
LORETA pipeline — sLORETA source localization using MNE + fsaverage standard brain.

Scientific rationale
--------------------
Scalp EEG electrodes record a superposition of brain sources blurred by volume
conduction through skull and scalp.  sLORETA (standardised Low Resolution
Electromagnetic Tomography) is a minimum-norm inverse method that recovers the
underlying cortical current density with zero localisation error for single
point sources (Pascual-Marqui, 2002).

Running forecasting models on source-space signals tests a neurophysiologically
richer hypothesis: do TSFMs capture structured dynamics of specific brain regions
(e.g. Default Mode Network, temporal memory areas) rather than mixed electrode
potentials?  This is particularly relevant for Alzheimer's / FTD discrimination.

Implementation
--------------
1. Load the preprocessed EEG derivative from BIDS (already filtered / ICA-cleaned).
2. Restrict to the SAME four-electrode subset as the scalp baseline
   (Fp1, Fp2, P3, P4).  The paper (Sec. III-B, and the Discussion / Limitations)
   states that the sLORETA source-space representation is derived from this
   four-electrode recording — not from the full 19-channel montage.
3. Set average reference (required for EEG source localisation).
4. Build one shared inverse operator from the fsaverage MNI brain template
   (3-layer BEM, oct5 source space) — computed once per electrode set and
   cached to disk at  benchmark/results/cache/loreta_inv_op_<n>ch_<hash>.pkl.
   The cache key encodes the channel layout so a 4-electrode operator is never
   confused with a differently-wired one.
5. Apply sLORETA (λ² = 1/9, i.e. assumed SNR = 3).
6. Extract parcel-averaged time series using the Desikan-Killiany (aparc) atlas.
7. Focus on six bilateral parcels implicated in Alzheimer's / FTD literature:
     precuneus        — Default Mode Network core, early Alzheimer's atrophy
     inferiorparietal — Angular / supramarginal gyrus; Alzheimer's
     superiortemporal — Language / memory; FTD, Alzheimer's
     middletemporal   — Temporal memory system; Alzheimer's
     superiorfrontal  — Executive function; FTD
     rostralmiddlefrontal — Prefrontal / orbitofrontal; FTD
8. Z-score + window extraction (identical to baseline pipeline).

Output channel names: "src_<parcel>_<hemi>"  (e.g. "src_precuneus_lh")
Max 12 output channels (6 parcels × 2 hemispheres).

Usage
-----
    python benchmark/run.py --pipeline loreta --n 3

Requirements
------------
    pip install mne
The fsaverage template (~50 MB) is downloaded automatically by MNE on first run
and cached in ~/mne_data/.  The inverse operator is then saved in
benchmark/results/cache/loreta_inv_op_<n>ch_<hash>.pkl (~few MB), keyed on the
electrode layout, so subsequent subjects are fast.
"""

import os
import pickle
import hashlib
import numpy as np
from scipy.stats import zscore

from . import BasePipeline, CHANNELS, CONTEXT_LEN, HORIZON_LEN, NUM_WINDOWS

# ── Parcels of interest (Desikan-Killiany atlas, both hemispheres) ────────────
PARCELS = [
    "precuneus",             # DMN hub; earliest Alzheimer's atrophy site
    "inferiorparietal",      # Angular / supramarginal; Alzheimer's signature
    "superiortemporal",      # Language & episodic memory; FTD & Alzheimer's
    "middletemporal",        # Temporal memory stream; Alzheimer's
    "superiorfrontal",       # Dorsolateral prefrontal; FTD executive deficit
    "rostralmiddlefrontal",  # Orbitofrontal / prefrontal; FTD
]
HEMIS = ["lh", "rh"]

# sLORETA regularisation: λ² = 1 / SNR²  (SNR = 3 is the conventional default,
# giving λ² = 1/9; see paper Sec. III-B).
_LAMBDA2 = 1.0 / 9.0

# Cache directory for the inverse operator (relative to this file)
_CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results", "cache")
)


def _inv_op_cache_path(raw_info) -> str:
    """
    Per-electrode-set cache path for the inverse operator.

    The inverse operator is only valid for the exact channel layout it was
    built from.  Encoding the channel names (count + short hash) into the
    filename guarantees that, e.g., a 4-electrode operator is never silently
    loaded for a 19-electrode run (or vice-versa) — a source of hard-to-spot
    scientific errors.
    """
    ch_names = list(raw_info["ch_names"])
    digest = hashlib.md5("|".join(ch_names).encode("utf-8")).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"loreta_inv_op_{len(ch_names)}ch_{digest}.pkl")


class LoretaPipeline(BasePipeline):
    """
    EEG → sLORETA source space → parcel time series → TSFM forecasting.

    Inherits BasePipeline (dataset loader, _extract_windows).
    The inverse operator is built once per channel configuration and reused.
    """
    name = "loreta"

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self._inv_op = None   # MNE inverse operator (cached after first build)
        self._src = None      # Source space object (needed for label extraction)
        self._labels = None   # Desikan-Killiany parcellation labels

    # ── Inverse operator ───────────────────────────────────────────────────────

    def _build_inv_op(self, raw_info):
        """
        Build the sLORETA inverse operator from the fsaverage BEM model.
        Saves the result to a channel-set-specific cache file (see
        _inv_op_cache_path) so subsequent runs skip this step.

        Parameters
        ----------
        raw_info : mne.Info
            Info object from the first subject's raw EEG (defines channel layout).
        """
        import mne
        from mne.datasets import fetch_fsaverage
        from mne.minimum_norm import make_inverse_operator

        print("  [LORETA] Fetching fsaverage template (downloads once, ~50 MB) ...")
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)
        subject = "fsaverage"

        # Source space: oct5 = 1026 vertices per hemisphere (good speed/resolution trade-off)
        print("  [LORETA] Setting up source space (oct5) ...")
        src = mne.setup_source_space(
            subject, spacing="oct5",
            subjects_dir=subjects_dir,
            add_dist=False, verbose=False,
        )

        # 3-layer BEM: brain (σ=0.3), skull (σ=0.006), scalp (σ=0.3)  [S/m]
        print("  [LORETA] Building BEM model ...")
        model = mne.make_bem_model(
            subject=subject, ico=3,
            conductivity=[0.3, 0.006, 0.3],
            subjects_dir=subjects_dir, verbose=False,
        )
        bem_sol = mne.make_bem_solution(model, verbose=False)

        # Forward solution (EEG only; "fsaverage" is the identity transform)
        print("  [LORETA] Computing forward solution (this may take 1–3 min) ...")
        fwd = mne.make_forward_solution(
            raw_info, trans="fsaverage", src=src, bem=bem_sol,
            eeg=True, meg=False, mindist=5.0, verbose=False,
        )

        # Ad-hoc noise covariance (identity matrix scaled to data variance).
        # No baseline or rest segment is explicitly used here; this is the
        # standard approach when individual noise data are unavailable.
        noise_cov = mne.make_ad_hoc_cov(raw_info, verbose=False)

        # Inverse operator: surface-constrained (loose=0), depth-weighted (depth=0.8)
        print("  [LORETA] Building inverse operator ...")
        inv_op = make_inverse_operator(
            raw_info, fwd, noise_cov,
            loose=0.0, depth=0.8, verbose=False,
        )

        # Persist to disk (path keyed on the exact channel layout)
        cache_path = _inv_op_cache_path(raw_info)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"inv_op": inv_op, "src": src}, fh)
        print(f"  [LORETA] Inverse operator cached → {cache_path}")

        self._inv_op = inv_op
        self._src = src

    def _ensure_inv_op(self, raw_info):
        """Load from cache if available, else build from scratch."""
        if self._inv_op is not None:
            return True  # already loaded in memory

        cache_path = _inv_op_cache_path(raw_info)
        if os.path.exists(cache_path):
            print("  [LORETA] Loading cached inverse operator ...")
            try:
                with open(cache_path, "rb") as fh:
                    data = pickle.load(fh)
                self._inv_op = data["inv_op"]
                self._src    = data["src"]
                return True
            except Exception as e:
                print(f"  [LORETA] Cache corrupt ({e}), rebuilding ...")

        try:
            self._build_inv_op(raw_info)
            return True
        except Exception as e:
            print(f"  [!] LORETA: could not build inverse operator: {e}")
            return False

    # ── Parcellation labels ────────────────────────────────────────────────────

    def _ensure_labels(self):
        """Lazy-load Desikan-Killiany labels from fsaverage (both hemispheres)."""
        if self._labels is not None:
            return True
        try:
            import mne
            from mne.datasets import fetch_fsaverage
            fs_dir = fetch_fsaverage(verbose=False)
            subjects_dir = os.path.dirname(fs_dir)
            self._labels = mne.read_labels_from_annot(
                "fsaverage", parc="aparc",
                subjects_dir=subjects_dir, verbose=False,
            )
            return True
        except Exception as e:
            print(f"  [!] LORETA: cannot load parcellation labels: {e}")
            return False

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def _process_parcels(self, subject_id: str) -> dict:
        """
        Run sLORETA and return z-scored parcel time series (no windowing yet).

        Returns
        -------
        dict  {src_<parcel>_<hemi>: (normed_signal_float32, raw_std_float)}
              Empty dict on failure.

        Used internally by process() and by LoretaGSPPipeline.
        """
        import mne
        from mne.minimum_norm import apply_inverse_raw

        # ── 1. Load raw EEG ───────────────────────────────────────────────────
        raw_mne = self.loader.load_subject(subject_id)
        raw_mne = raw_mne.copy()  # do not mutate the cached loader object

        # ── 2. Restrict to the four-electrode subset (paper Sec. III-B) ───────
        # The paper derives the sLORETA source space from the SAME four scalp
        # electrodes used for the scalp baseline (Fp1, Fp2, P3, P4), not from the
        # full 19-channel montage.  This is stated in the Methods and reiterated
        # in the Discussion / Limitations ("twelve parcels derived from a
        # four-electrode recording"; "derived from the same limited electrode
        # set").  All four are required — a subject missing any is skipped.
        montage = mne.channels.make_standard_montage("standard_1020")
        picks = [ch for ch in CHANNELS if ch in raw_mne.ch_names]
        if len(picks) < len(CHANNELS):
            missing = sorted(set(CHANNELS) - set(picks))
            print(f"  [!] LORETA: {subject_id} is missing electrode(s) {missing}; "
                  f"the four-electrode montage {CHANNELS} is required. Skipping.")
            return {}

        raw_mne.pick_channels(picks, ordered=True)
        raw_mne.set_channel_types({ch: "eeg" for ch in picks})
        try:
            raw_mne.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception as e:
            print(f"  [!] LORETA: set_montage failed for {subject_id}: {e}")
            return {}

        # ── 3. Average reference (mandatory before source localisation) ───────
        raw_mne.set_eeg_reference("average", projection=True, verbose=False)
        raw_mne.apply_proj(verbose=False)

        # ── 4. Inverse operator ───────────────────────────────────────────────
        if not self._ensure_inv_op(raw_mne.info):
            return {}

        # ── 5. Apply sLORETA ──────────────────────────────────────────────────
        try:
            stc = apply_inverse_raw(
                raw_mne, self._inv_op, _LAMBDA2,
                method="sLORETA", verbose=False,
            )
        except Exception as e:
            print(f"  [!] LORETA: apply_inverse_raw failed for {subject_id}: {e}")
            return {}

        # ── 6. Load parcellation ──────────────────────────────────────────────
        if not self._ensure_labels():
            return {}

        # ── 7. Extract parcel time series and z-score (no windowing) ─────────
        result = {}
        for parcel in PARCELS:
            for hemi in HEMIS:
                label_name = f"{parcel}-{hemi}"
                matching = [lb for lb in self._labels if lb.name == label_name]
                if not matching:
                    continue
                label = matching[0]

                try:
                    # mean_flip: averages source amplitudes within the parcel
                    # while handling the sign ambiguity of minimum-norm solutions.
                    tc = mne.extract_label_time_course(
                        [stc], [label], self._src,
                        mode="mean_flip", verbose=False,
                    )
                    # tc is a list (one entry per stc); each entry: (n_labels, n_times)
                    sig = tc[0][0].astype(np.float64)   # shape: (n_times,)
                except Exception as e:
                    print(f"  [!] LORETA: label extraction failed "
                          f"({label_name}, {subject_id}): {e}")
                    continue

                raw_std = float(np.std(sig))
                if raw_std < 1e-15:
                    continue  # degenerate / all-zero source — skip

                normed = zscore(sig).astype(np.float32)
                result[f"src_{parcel}_{hemi}"] = (normed, raw_std)

        return result

    def process(self, subject_id: str) -> dict:
        """
        Returns {src_<parcel>_<hemi>: {'windows': [...], 'raw_std': float}}.

        Steps:
          load EEG -> set montage -> average reference ->
          sLORETA -> extract parcel signals -> z-score -> windows
        """
        parcels = self._process_parcels(subject_id)

        result = {}
        for key, (normed, raw_std) in parcels.items():
            windows = self._extract_windows(normed)
            if not windows:
                continue
            result[key] = {"windows": windows, "raw_std": raw_std}

        if not result:
            print(f"  [!] LORETA: no valid parcel signals for {subject_id}.")

        return result

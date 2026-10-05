"""
EEG-TSFM Benchmark runner.

Loads EEG data ONCE via benchmark/pipelines/baseline.py (or loreta.py), then
dispatches to per-model subprocesses (each in its own venv) via temp pickle files.
Model venvs need NO MNE / EEG libraries — only the model packages.

Interface
---------
Input pickle  → {subjects: {subj_id: {group, ch: {windows, raw_std}}}, horizon_len, device}
Output pickle ← {subj_id: {group, ch: {predictions, targets, raw_std}}}

Usage
-----
    python benchmark/run.py --n 3
    python benchmark/run.py --models Chronos TimesFM --n 5 --device cuda
    python benchmark/run.py --setup-only
    python benchmark/run.py --no-setup --refresh
    python benchmark/run.py --force-setup    # rebuild all venvs from scratch
    python benchmark/run.py --pipeline loreta --n 3
"""

import sys
import os
import argparse
import subprocess
import shutil
import pickle
import tempfile
import csv
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent       # benchmark/
_ROOT     = _HERE.parent                           # mag/ (project root)
_MODELS   = _HERE / "models"
_RESULTS  = _HERE / "results" / "baseline"
_DATASET  = _ROOT / "datasets" / "ds004504"
_REF      = _ROOT / "reference"

# Add root to sys.path so that benchmark.pipelines is importable
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# PyTorch CUDA wheel index — adjust if your CUDA version differs
_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
_TORCH_VER   = "torch==2.6.0+cu124"

ALL_MODELS = [
    "Chronos",
    "Chronos-2",
    "TimesFM",
    "Moirai",
    "Lag-Llama",
    "Sundial",
    "ViTime",
    "TimeFound",
]

_FOLDER = {
    "Chronos":   "chronos",
    "Chronos-2": "chronos2",
    "TimesFM":   "timesfm",
    "Moirai":    "moirai",
    "Lag-Llama": "lag_llama",
    "Sundial":   "sundial",
    "ViTime":    "vitime",
    "TimeFound": "timefound",
}


# ── venv helpers ───────────────────────────────────────────────────────────────

def _venv_dir(folder: str) -> Path:
    return _MODELS / folder / "venv"


def _venv_python(folder: str) -> Path:
    venv = _venv_dir(folder)
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _venv_pip(folder: str) -> Path:
    venv = _venv_dir(folder)
    return venv / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")


def _run(cmd: list, **kwargs):
    """Run command, print it, raise on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Exit {result.returncode}: {' '.join(str(c) for c in cmd)}")


def setup_venv(model_name: str, device: str, force: bool = False):
    folder    = _FOLDER[model_name]
    model_dir = _MODELS / folder
    venv      = _venv_dir(folder)
    done_flag = venv / ".isolated_setup_done"

    if done_flag.exists() and not force:
        print(f"[setup] {model_name}: already set up (delete venv/.isolated_setup_done to redo)")
        return

    print(f"\n{'='*60}\n  Setting up: {model_name}\n{'='*60}")

    if venv.exists() and force:
        shutil.rmtree(venv)

    _run([sys.executable, "-m", "venv", str(venv)])
    python = str(_venv_python(folder))
    pip    = str(_venv_pip(folder))

    # Use `python -m pip` for the self-upgrade — pip.exe can't replace itself on Windows
    _run([python, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # Install PyTorch
    if device == "cpu":
        _run([pip, "install", "torch==2.6.0", "-q"])
    else:
        _run([pip, "install", _TORCH_VER, "--index-url", _TORCH_INDEX, "-q"])

    # Model requirements
    req = model_dir / "requirements.txt"
    if req.exists():
        _run([pip, "install", "-r", str(req), "-q"])

    # Extra per-model steps
    if model_name == "Moirai":
        # uni2ts pins numpy~=1.26 (no Python 3.13 wheel) — install without deps.
        # All needed runtime deps are in requirements.txt with compatible versions.
        _run([pip, "install", "uni2ts", "--no-deps", "-q"])

    done_flag.touch()
    print(f"[setup] {model_name}: done.")


# ── EEG data loading ───────────────────────────────────────────────────────────

def load_eeg_data(n: int | None, device: str, horizon_len: int = 64,
                  pipeline_name: str = "baseline") -> dict:
    """
    Load EEG windows for all subjects using the chosen pipeline.
      baseline — scalp channels Fp1/Fp2/P3/P4
      loreta   — sLORETA source parcels via fsaverage standard brain
    Returns the input-pickle payload.
    """
    if pipeline_name == "loreta":
        from benchmark.pipelines.loreta import LoretaPipeline
        pipeline = LoretaPipeline(str(_DATASET))
    elif pipeline_name == "loreta_gsp":
        from benchmark.pipelines.loreta_gsp import LoretaGSPPipeline
        pipeline = LoretaGSPPipeline(str(_DATASET))
    else:
        from benchmark.pipelines.baseline import BaselinePipeline
        pipeline = BaselinePipeline(str(_DATASET))

    print(f"\nLoading EEG data from: {_DATASET}  [pipeline: {pipeline_name}]")
    subject_ids = pipeline.get_subjects(n)
    print(f"  Subjects: {len(subject_ids)}")

    subjects = {}
    for i, subj_id in enumerate(subject_ids):
        print(f"  [{i+1}/{len(subject_ids)}] {subj_id}", end="", flush=True)
        try:
            group     = pipeline.get_group(subj_id)
            ch_data   = pipeline.process(subj_id)
            if not ch_data:
                print(" — no windows, skipping")
                continue
            subjects[subj_id] = {"group": group, **ch_data}
            print(f" — {group}, {len(ch_data)} channels")
        except Exception as exc:
            print(f" — ERROR: {exc}")

    return {
        "subjects":   subjects,
        "horizon_len": horizon_len,
        "device":     device,
    }


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_and_save_metrics(model_name: str, output: dict, results_dir: Path):
    """Compute MSE (normalised + physical) and write CSVs."""
    import numpy as np

    rows = []
    for subj_id, subj_data in output.items():
        group = subj_data.get("group", "Unknown")
        for ch, ch_res in subj_data.items():
            if ch == "group":
                continue
            preds   = np.array(ch_res["predictions"])   # (n_win, H)
            targets = np.array(ch_res["targets"])        # (n_win, H)
            raw_std = float(ch_res["raw_std"])

            mse_norm = float(np.mean((preds - targets) ** 2))
            mse_phys = mse_norm * (raw_std ** 2)

            rows.append({
                "subject":   subj_id,
                "group":     group,
                "channel":   ch,
                "mse_norm":  mse_norm,
                "mse_phys":  mse_phys,
            })

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{model_name.lower().replace('-', '_')}_metrics.csv"
    fieldnames = ["subject", "group", "channel", "mse_norm", "mse_phys"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Group-level summary
    import collections
    by_group: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append((r["mse_norm"], r["mse_phys"]))

    print(f"\n  Metrics — {model_name}")
    for grp, vals in sorted(by_group.items()):
        mn  = sum(v[0] for v in vals) / len(vals)
        mp  = sum(v[1] for v in vals) / len(vals)
        print(f"    {grp}: MSE_norm={mn:.6f}  MSE_phys={mp:.4e}")
    print(f"  Saved: {csv_path}")
    return csv_path


# ── run one model ──────────────────────────────────────────────────────────────

def run_model(model_name: str, payload: dict, refresh: bool = False,
              results_root: Path = None) -> bool:
    """
    Write input pickle → run runner.py subprocess → read output pickle.
    Returns True on success.
    """
    folder    = _FOLDER[model_name]
    python    = _venv_python(folder)
    runner    = _MODELS / folder / "runner.py"
    res_dir   = (results_root or _RESULTS) / folder
    cache_pkl = res_dir / "output.pkl"

    if not python.exists():
        print(f"[run] {model_name}: venv not found — run --setup-only first")
        return False
    if not runner.exists():
        print(f"[run] {model_name}: runner.py not found at {runner}")
        return False

    if cache_pkl.exists() and not refresh:
        print(f"[run] {model_name}: loading cached results from {cache_pkl}")
        with open(cache_pkl, "rb") as f:
            output = pickle.load(f)
        compute_and_save_metrics(model_name, output, res_dir)
        return True

    print(f"\n{'='*60}\n  Running: {model_name}\n{'='*60}")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf_in:
        in_path = tf_in.name
    out_path = in_path.replace(".pkl", "_out.pkl")

    try:
        with open(in_path, "wb") as f:
            pickle.dump(payload, f)

        cmd = [str(python), str(runner), "--input", in_path, "--output", out_path]
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[run] {model_name}: FAILED (exit {result.returncode})")
            return False

        with open(out_path, "rb") as f:
            output = pickle.load(f)

        res_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_pkl, "wb") as f:
            pickle.dump(output, f)

        compute_and_save_metrics(model_name, output, res_dir)
        print(f"[run] {model_name}: OK")
        return True

    except Exception as exc:
        print(f"[run] {model_name}: EXCEPTION — {exc}")
        return False
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ── summary ────────────────────────────────────────────────────────────────────

def print_summary(run_results: dict[str, bool], results_root: Path):
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for name, ok in run_results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    ok_n = sum(run_results.values())
    print(f"\n  {ok_n}/{len(run_results)} models succeeded.")
    print(f"  Results: {results_root}\n{'='*60}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG-TSFM benchmark runner")
    parser.add_argument("--models",       nargs="+", default=ALL_MODELS,
                        help=f"Models to run. Default: all. Choices: {ALL_MODELS}")
    parser.add_argument("--n",            type=int, default=None,
                        help="Limit to first N subjects")
    parser.add_argument("--device",       default=None,
                        help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--refresh",      action="store_true",
                        help="Ignore cached output and re-run predictions")
    parser.add_argument("--setup-only",   action="store_true",
                        help="Only create/update venvs, skip running models")
    parser.add_argument("--no-setup",     action="store_true",
                        help="Skip venv setup (assume already done)")
    parser.add_argument("--force-setup",  action="store_true",
                        help="Recreate venvs even if .isolated_setup_done exists")
    parser.add_argument("--pipeline",     default="baseline",
                        choices=["baseline", "loreta", "loreta_gsp"],
                        help="Data pipeline: baseline (scalp EEG), loreta "
                             "(sLORETA source parcels), or loreta_gsp "
                             "(sLORETA parcels projected onto network harmonics). "
                             "Default: baseline")
    args = parser.parse_args()

    for m in args.models:
        if m not in ALL_MODELS:
            parser.error(f"Unknown model '{m}'. Available: {ALL_MODELS}")

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    print(f"\nDevice: {args.device}")

    # results/baseline  or  results/loreta
    results_root = _HERE / "results" / args.pipeline
    results_root.mkdir(parents=True, exist_ok=True)

    # Phase 1 — venv setup
    if not args.no_setup:
        print(f"\n{'─'*60}\n  Phase 1: Venv setup\n{'─'*60}")
        for model_name in args.models:
            try:
                setup_venv(model_name, args.device, force=args.force_setup)
            except Exception as exc:
                print(f"[setup] {model_name}: FAILED — {exc}")

    if args.setup_only:
        print("\nSetup done. Rerun without --setup-only to run predictions.")
        return

    # Phase 2 — load EEG data (once, shared across all models)
    print(f"\n{'─'*60}\n  Phase 2: Loading EEG data  [{args.pipeline}]\n{'─'*60}")
    payload = load_eeg_data(args.n, args.device, pipeline_name=args.pipeline)
    if not payload["subjects"]:
        print("No subjects loaded — check dataset path.")
        return
    print(f"  Loaded {len(payload['subjects'])} subjects.")

    # Phase 3 — run models
    print(f"\n{'─'*60}\n  Phase 3: Running models\n{'─'*60}")
    run_results: dict[str, bool] = {}
    for model_name in args.models:
        try:
            run_results[model_name] = run_model(
                model_name, payload,
                refresh=args.refresh,
                results_root=results_root,
            )
        except Exception as exc:
            print(f"[run] {model_name}: EXCEPTION \u2014 {exc}")
            run_results[model_name] = False

    print_summary(run_results, results_root)


if __name__ == "__main__":
    main()

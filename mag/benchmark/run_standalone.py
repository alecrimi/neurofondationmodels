"""
Standalone benchmark orchestrator.

Creates a separate Python venv for each TSFM model, installs its dependencies,
then runs each model's run_model.py as a subprocess. Results are collected in
new/results/standalone/<model>/.

Usage
-----
# All models, 3 subjects, auto-detect GPU
python new/run_standalone.py --n 3

# Specific models only
python new/run_standalone.py --models Chronos TimesFM --n 5

# Force re-run, ignore cached predictions
python new/run_standalone.py --refresh

# CPU only
python new/run_standalone.py --device cpu --n 3

# Only set up venvs, don't run predictions yet
python new/run_standalone.py --setup-only

Requirements
------------
- Python 3.11+ on PATH (the venvs inherit the system Python)
- CUDA 12.4 + RTX 4060 for GPU runs (or use --device cpu)
- new/models/lag_llama/repo/ cloned (for Lag-Llama)
- new/models/vitime/repo/ cloned + new/models/vitime/ViTime_Model.pth (for ViTime)
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent         # new/
_ROOT        = _HERE.parent                            # mag/
_STANDALONE  = _HERE / "standalone"
_RESULTS_DIR = _HERE / "results" / "standalone"
_DATASET     = _ROOT / "datasets" / "ds004504"

# PyTorch CUDA wheel index — change if your CUDA version differs
_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
_TORCH_VER   = "torch==2.6.0+cu124"

# ── model registry ─────────────────────────────────────────────────────────────
# Each entry: (display_name, folder_name, extra_setup_fn_name_or_None)
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


# ── helpers ────────────────────────────────────────────────────────────────────

def venv_python(model_folder: str) -> Path:
    venv_dir = _STANDALONE / model_folder / "venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_pip(model_folder: str) -> Path:
    venv_dir = _STANDALONE / model_folder / "venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def run(cmd: list, **kwargs):
    """Run a command, stream output, raise on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def setup_venv(model_name: str, device: str, force: bool = False):
    """Create venv and install dependencies for a model. Skips if already done."""
    folder = _FOLDER[model_name]
    model_dir = _STANDALONE / folder
    venv_dir  = model_dir / "venv"
    done_flag = venv_dir / ".setup_done"

    if done_flag.exists() and not force:
        print(f"[setup] {model_name}: venv already set up (delete venv/.setup_done to re-run)")
        return

    print(f"\n{'='*60}")
    print(f"  Setting up venv for: {model_name}")
    print(f"{'='*60}")

    # Create venv
    if venv_dir.exists() and force:
        shutil.rmtree(venv_dir)
    run([sys.executable, "-m", "venv", str(venv_dir)])

    pip = str(venv_pip(folder))

    # Upgrade pip
    run([pip, "install", "--upgrade", "pip", "-q"])

    # Install PyTorch (CUDA or CPU build)
    if device == "cpu":
        run([pip, "install", "torch==2.6.0", "-q"])
    else:
        run([pip, "install", _TORCH_VER, "--index-url", _TORCH_INDEX, "-q"])

    # Install model requirements
    req_file = model_dir / "requirements.txt"
    run([pip, "install", "-r", str(req_file), "-q"])

    # Extra steps per model
    if model_name == "Moirai":
        # uni2ts has a hard dep on gluonts~=0.14 — install without deps
        run([pip, "install", "uni2ts", "--no-deps", "-q"])

    done_flag.touch()
    print(f"[setup] {model_name}: done.")


# ── run one model ──────────────────────────────────────────────────────────────

def run_model(model_name: str, args) -> bool:
    """Run a single model's run_model.py in its own venv. Returns True on success."""
    folder     = _FOLDER[model_name]
    python     = str(venv_python(folder))
    script     = str(_STANDALONE / folder / "run_model.py")
    output_dir = str(_RESULTS_DIR / folder)

    if not Path(python).exists():
        print(f"[run] {model_name}: venv not found — run with --setup-only first or skip --no-setup")
        return False

    cmd = [
        python, script,
        "--dataset", str(_DATASET),
        "--output",  output_dir,
        "--device",  args.device,
    ]
    if args.n:
        cmd += ["--n", str(args.n)]
    if args.refresh:
        cmd.append("--refresh")

    print(f"\n{'='*60}")
    print(f"  Running: {model_name}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"[run] {model_name}: {status}")
    return ok


# ── summary ────────────────────────────────────────────────────────────────────

def print_summary(results: dict[str, bool]):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        symbol = "✓" if ok else "✗"
        print(f"  {symbol}  {name}")

    ok_count = sum(results.values())
    print(f"\n  {ok_count}/{len(results)} models succeeded.")
    print(f"  Results in: {_RESULTS_DIR}")
    print(f"{'='*60}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG-TSFM standalone benchmark runner")
    parser.add_argument("--models",     nargs="+", default=ALL_MODELS,
                        help=f"Models to run. Default: all. Options: {ALL_MODELS}")
    parser.add_argument("--n",          type=int, default=None,
                        help="Limit to first N subjects")
    parser.add_argument("--device",     default=None,
                        help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--refresh",    action="store_true",
                        help="Ignore existing caches and re-run predictions")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only create/update venvs, skip running models")
    parser.add_argument("--no-setup",   action="store_true",
                        help="Skip venv setup (assume already done)")
    parser.add_argument("--force-setup", action="store_true",
                        help="Recreate venvs even if .setup_done exists")
    args = parser.parse_args()

    # Validate models
    for m in args.models:
        if m not in ALL_MODELS:
            parser.error(f"Unknown model '{m}'. Available: {ALL_MODELS}")

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    print(f"\nDevice: {args.device}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── venv setup ─────────────────────────────────────────────────────────────
    if not args.no_setup:
        print(f"\n{'─'*60}")
        print("  Phase 1: Setting up venvs")
        print(f"{'─'*60}")
        for model_name in args.models:
            try:
                setup_venv(model_name, args.device, force=args.force_setup)
            except Exception as e:
                print(f"[setup] {model_name}: FAILED — {e}")

    if args.setup_only:
        print("\nSetup done. Run without --setup-only to execute predictions.")
        return

    # ── run models ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Phase 2: Running models")
    print(f"{'─'*60}")

    run_results: dict[str, bool] = {}
    for model_name in args.models:
        try:
            run_results[model_name] = run_model(model_name, args)
        except Exception as e:
            print(f"[run] {model_name}: EXCEPTION — {e}")
            run_results[model_name] = False

    print_summary(run_results)


if __name__ == "__main__":
    main()

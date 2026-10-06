"""
Foundation-model residual causality runner.

Each model is run separately.  Univariate forecasts of every target channel are
computed once per (model, subject, channel) and cached, so changing only the lag
list or the subject subset does not require new forecasts.

Usage (from the project root, i.e. the folder that contains benchmark/)
-----
  # notebook defaults: lags [5,10,20,30,40,50] samples (10–100 ms), 4 scalp pairs
  python benchmark/run_residual_causality.py --models Chronos --device cpu

  # several models, only AD subjects, lags given in milliseconds
  python benchmark/run_residual_causality.py --models Chronos TimesFM --group ad --lags "[10,20,40]" --lag-unit ms

  # explicit subjects instead of a group
  python benchmark/run_residual_causality.py --models Chronos-2 --subjects sub-001 sub-002 37

  # quick smoke test on the first 2 selected subjects
  python benchmark/run_residual_causality.py --models Chronos --limit 2 --device cpu --no-setup

Outputs: benchmark/results/residual_causality/<pipeline>/<run_name>/
  <model>_residual_causality_results.csv   one row per subject × pair × lag
                                           (notebook columns first, compatible
                                           with tfsm_causality_processing.py)
  summary_by_lag.csv                       median F and % significant per scope
                                           (ALL, A, C, F), pair and lag
  optimal_lag.csv                          optimal lag per pair (+ group-wise %)
  subjects.csv                             selected subjects, group, data source
  run_config.json                          parameters, git commit, runner hashes
  environment_<model>.txt                  pip freeze of each model venv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent   # benchmark/
_ROOT = _HERE.parent                      # project root (mag/)
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark import run as bench                       # noqa: E402  (venv + runner registry)
from benchmark.pipelines import AlzheimerLoader          # noqa: E402
from benchmark.experiments import residual_causality as rc  # noqa: E402

_RESULTS_ROOT = _HERE / "results" / "residual_causality"
_CACHE_ROOT = _RESULTS_ROOT / "_forecast_cache"


# ── helpers ───────────────────────────────────────────────────────────────────

def _sha1_file(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _sha1_arrays(arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _model_slug(model_name: str) -> str:
    return bench._FOLDER[model_name].replace("_", "")


def _resolve_models(names: list[str]) -> list[str]:
    lookup = {}
    for m in bench.ALL_MODELS:
        for alias in {m, m.lower(), bench._FOLDER[m], bench._FOLDER[m].replace("_", ""),
                      m.lower().replace("-", "")}:
            lookup[alias.lower()] = m
    out = []
    for n in names:
        key = n.lower()
        if key not in lookup:
            raise SystemExit(f"Unknown model {n!r}. Available: {bench.ALL_MODELS}")
        if lookup[key] not in out:
            out.append(lookup[key])
    return out


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
                           text=True, timeout=10)
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except Exception:
        return None


def _pip_freeze(python: Path) -> str:
    try:
        r = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True,
                           text=True, timeout=120)
        return r.stdout if r.returncode == 0 else f"# pip freeze failed: {r.stderr}"
    except Exception as exc:
        return f"# pip freeze failed: {exc}"


# ── data loading ──────────────────────────────────────────────────────────────

class SignalSource:
    """Loads full-length signals for the requested channels of one subject."""

    def __init__(self, dataset: Path, pipeline: str, scaling: str):
        self.dataset = Path(dataset)
        self.pipeline = pipeline
        self.scaling = scaling
        self.loader = AlzheimerLoader(str(self.dataset))
        self._loreta = None

    def data_source(self, sid: str) -> str:
        deriv = self.dataset / "derivatives" / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        raw = self.dataset / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        if deriv.exists():
            return "derivatives"
        if raw.exists():
            return "raw_bids"
        return "missing"

    def load(self, sid: str, channels: set[str]) -> dict[str, np.ndarray]:
        if self.pipeline == "baseline":
            from scipy.stats import zscore
            raw = self.loader.load_subject(sid)
            out = {}
            for ch in channels:
                if ch not in raw.ch_names:
                    continue
                sig = raw.get_data(picks=[ch])[0]           # Volts, float64
                if self.scaling == "zscore":
                    sig = zscore(sig)                        # as BaselinePipeline
                out[ch] = np.asarray(sig, dtype=np.float32)  # as notebooks (float32)
            return out
        if self.pipeline == "loreta":
            if self._loreta is None:
                from benchmark.pipelines.loreta import LoretaPipeline
                self._loreta = LoretaPipeline(str(self.dataset))
            parcels = self._loreta._process_parcels(sid)   # z-scored parcel signals
            return {k: np.asarray(v[0], dtype=np.float32) for k, v in parcels.items() if k in channels}
        raise ValueError(f"Unsupported pipeline {self.pipeline!r}")


def select_subjects(loader: AlzheimerLoader, group_code: str | None,
                    subject_ids: list[str] | None, limit: int | None) -> pd.DataFrame:
    part = loader.participants.copy()
    part["participant_id"] = part["participant_id"].astype(str)
    if subject_ids:
        wanted = [loader._normalize_id(s) for s in subject_ids]
        known = set(part["participant_id"])
        unknown = [s for s in wanted if s not in known]
        if unknown:
            raise SystemExit(f"Subjects not found in participants.tsv: {unknown}")
        sel = part.set_index("participant_id").loc[list(dict.fromkeys(wanted))].reset_index()
    else:
        sel = part if group_code is None else part[part["Group"] == group_code]
    if limit:
        sel = sel.head(limit)
    return sel[["participant_id", "Group"]].rename(columns={"participant_id": "subject_id",
                                                           "Group": "group"}).reset_index(drop=True)


# ── forecasting through runners ──────────────────────────────────────

def run_runner(python: Path, runner: Path, payload: dict) -> dict:
    """Write input pickle -> run runner.py in its venv -> read output pickle."""
    fd, in_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    out_path = in_path.replace(".pkl", "_out.pkl")
    try:
        with open(in_path, "wb") as f:
            pickle.dump(payload, f)
        result = subprocess.run([str(python), str(runner), "--input", in_path, "--output", out_path])
        if result.returncode != 0:
            raise RuntimeError(f"runner exited with code {result.returncode}")
        with open(out_path, "rb") as f:
            return pickle.load(f)
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


class ForecastCache:
    """Per (model, subject, channel) cache of univariate forecasts, validated on load."""

    def __init__(self, root: Path, protocol: dict, enabled: bool = True):
        self.protocol = protocol
        self.enabled = enabled
        tag = hashlib.sha1(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = root / tag
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))

    def _path(self, sid: str, ch: str) -> Path:
        return self.dir / f"{sid}__{ch}.npz"

    def get(self, sid: str, ch: str, starts: np.ndarray, ctx_hash: str):
        p = self._path(sid, ch)
        if not (self.enabled and p.exists()):
            return None
        try:
            z = np.load(p, allow_pickle=False)
            if not np.array_equal(z["starts"], starts) or str(z["ctx_hash"]) != ctx_hash:
                return None
            return z["predictions"]
        except Exception:
            return None

    def put(self, sid: str, ch: str, starts: np.ndarray, ctx_hash: str, preds: np.ndarray):
        if self.enabled:
            np.savez(self._path(sid, ch), starts=starts, ctx_hash=np.array(ctx_hash),
                     predictions=np.asarray(preds, dtype=np.float64))


def forecast_targets(model: str, subjects: dict, targets_needed: dict, device: str,
                     protocol_base: dict, refresh: bool) -> dict:
    """
    Return {(sid, ch): predictions (n_win, H)} for every needed target channel,
    running the model's runner only for entries missing from the cache.
    """
    folder = bench._FOLDER[model]
    python = bench._venv_python(folder)
    runner = bench._MODELS / folder / "runner.py"
    if not python.exists():
        raise RuntimeError(f"venv not found for {model}: {python} (run without --no-setup)")
    if not runner.exists():
        raise RuntimeError(f"runner.py not found: {runner}")

    protocol = dict(protocol_base, model=model,
                    runner_sha1=_sha1_file(runner),
                    requirements_sha1=_sha1_file(bench._MODELS / folder / "requirements.txt"))
    cache = ForecastCache(_CACHE_ROOT / _model_slug(model), protocol, enabled=True)

    preds: dict = {}
    payload_subjects: dict = {}
    pending: list = []
    for sid, chans in targets_needed.items():
        info = subjects[sid]
        for ch in chans:
            y = info["signals"][ch]
            starts = info["starts"][ch]
            contexts = rc.window_contexts(y, starts)
            ctx_hash = _sha1_arrays(contexts)
            cached = None if refresh else cache.get(sid, ch, starts, ctx_hash)
            if cached is not None:
                preds[(sid, ch)] = cached
                continue
            targets = rc.window_targets(y, starts)
            payload_subjects.setdefault(sid, {"group": info["group"]})[ch] = {
                "windows": [{"context": c, "target": t, "start_idx": int(s)}
                            for c, t, s in zip(contexts, targets, starts)],
                "raw_std": float(np.std(y)),
            }
            pending.append((sid, ch, starts, ctx_hash))

    n_cached = len(preds)
    print(f"[{model}] forecasts: {n_cached} from cache, {len(pending)} to compute")
    if pending:
        payload = {"subjects": payload_subjects, "horizon_len": rc.HORIZON_LEN, "device": device}
        output = run_runner(python, runner, payload)
        for sid, ch, starts, ctx_hash in pending:
            try:
                p = np.asarray(output[sid][ch]["predictions"], dtype=np.float64)
            except KeyError:
                print(f"  [!] {model}: runner returned no forecast for {sid}/{ch}")
                continue
            if p.shape != (len(starts), rc.HORIZON_LEN) or not np.all(np.isfinite(p)):
                print(f"  [!] {model}: invalid forecast for {sid}/{ch} (shape {p.shape})")
                continue
            cache.put(sid, ch, starts, ctx_hash, p)
            preds[(sid, ch)] = p
    return preds


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Foundation-model residual causality (Kaggle-notebook maths, Hubert's model runners)")
    p.add_argument("--models", nargs="+", required=True,
                   help=f"One or more of {bench.ALL_MODELS}; each is run separately.")
    p.add_argument("--lags", default=str(rc.DEFAULT_LAGS_SAMPLES),
                   help='Lag list in square brackets, e.g. "[5,10,20,30,40,50]" (default: notebook lags).')
    p.add_argument("--lag-unit", choices=["samples", "ms"], default="samples",
                   help="Unit of --lags (default: samples, as in the notebooks; 1 sample = 2 ms).")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--group", default="all",
                     help="all | ad | ftd (alias fd) | control  (default: all)")
    sel.add_argument("--subjects", nargs="+",
                     help="Explicit subject IDs (sub-001 or 1); overrides group selection.")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N selected subjects (like LIMIT_PATIENTS).")
    p.add_argument("--pairs", default=None,
                   help='Driver->target pairs, e.g. "P3->Fp1,Fp1->P3" (default: the 4 notebook pairs).')
    p.add_argument("--pipeline", choices=["baseline", "loreta"], default="baseline",
                   help="Signal space (default: baseline scalp EEG). loreta requires --pairs with parcel names.")
    p.add_argument("--scaling", choices=["raw", "zscore"], default=None,
                   help="baseline only: raw volts as in the notebooks (default) or full-signal z-score as in run.py.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--dataset", default=str(bench._DATASET), help="Path to ds004504")
    p.add_argument("--run-name", default=None, help="Output sub-folder name (default: timestamp)")
    p.add_argument("--refresh", action="store_true", help="Ignore cached forecasts and recompute them")
    p.add_argument("--no-setup", action="store_true", help="Skip model venv setup")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    models = _resolve_models(args.models)
    lags = rc.parse_lags(args.lags, unit=args.lag_unit)
    pairs = rc.parse_pairs(args.pairs)
    if args.pipeline == "loreta" and args.pairs is None:
        raise SystemExit("--pipeline loreta needs explicit --pairs with parcel names, "
                         "e.g. \"src_precuneus_lh->src_superiorfrontal_lh\".")
    scaling = args.scaling or ("raw" if args.pipeline == "baseline" else "zscore")
    if args.pipeline == "loreta" and scaling != "zscore":
        raise SystemExit("The loreta pipeline only provides z-scored parcel signals (--scaling zscore).")
    group_code = None if args.subjects else rc.resolve_group(args.group)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    run_name = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _RESULTS_ROOT / args.pipeline / run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output folder already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {args.device} | pipeline: {args.pipeline} ({scaling}) | models: {models}")
    print(f"Lags (samples): {lags} -> ms: {[int(l * 1000 / rc.FS) for l in lags]}")
    print(f"Pairs (X -> Y): {[f'{x}->{y}' for x, y in pairs]}")
    if "TimeFound" in models:
        print("  [note] TimeFound runner uses random weights (no official checkpoint).")

    if not args.no_setup:
        for m in models:
            try:
                bench.setup_venv(m, args.device)
            except Exception as exc:
                print(f"[setup] {m}: FAILED — {exc}")

    # ── subjects and signals (loaded once, shared by all models) ──────────────
    source = SignalSource(Path(args.dataset), args.pipeline, scaling)
    sel = select_subjects(source.loader, group_code, args.subjects, args.limit)
    if sel.empty:
        raise SystemExit("No subjects selected.")
    print(f"Selected subjects: {len(sel)}")

    needed_channels = {c for pair in pairs for c in pair}
    subjects: dict = {}
    subj_rows = []
    for i, row in sel.iterrows():
        sid, grp = row["subject_id"], row["group"]
        status, note = "ok", ""
        try:
            signals = source.load(sid, needed_channels)
            starts = {}
            for ch, sig in signals.items():
                starts[ch] = rc.window_starts(len(sig))
            missing = sorted(needed_channels - set(signals))
            if missing:
                note = f"missing channels {missing}"
            subjects[sid] = {"group": grp, "signals": signals, "starts": starts}
        except Exception as exc:
            status, note = "skipped", str(exc)
        subj_rows.append({"subject_id": sid, "group": grp, "status": status,
                          "data_source": source.data_source(sid), "note": note})
        print(f"  [{i + 1}/{len(sel)}] {sid} ({grp}) — {status}{(': ' + note) if note else ''}")
    pd.DataFrame(subj_rows).to_csv(out_dir / "subjects.csv", index=False)

    targets_needed: dict = {}
    for sid, info in subjects.items():
        for x, y in pairs:
            if x in info["signals"] and y in info["signals"]:
                targets_needed.setdefault(sid, set()).add(y)
    targets_needed = {k: sorted(v) for k, v in targets_needed.items()}

    protocol_base = {
        "pipeline": args.pipeline, "scaling": scaling, "dataset": str(Path(args.dataset).resolve()),
        "context_len": rc.CONTEXT_LEN, "horizon_len": rc.HORIZON_LEN,
        "num_windows": rc.NUM_WINDOWS, "offset_samples": rc.OFFSET_SAMPLES,
    }

    # ── per model: forecasts -> residual F-tests ──────────────────────────────
    all_rows = []
    model_info = {}
    for model in models:
        print(f"\n{'=' * 60}\n  Residual causality: {model}\n{'=' * 60}")
        try:
            preds = forecast_targets(model, subjects, targets_needed, args.device,
                                     protocol_base, args.refresh)
        except Exception as exc:
            print(f"[{model}] FAILED — {exc}")
            model_info[model] = {"status": f"failed: {exc}"}
            continue

        rows = []
        for sid, info in subjects.items():
            for x, y in pairs:
                if (sid, y) not in preds or x not in info["signals"]:
                    continue
                sig_x, sig_y = info["signals"][x], info["signals"][y]
                if len(sig_x) != len(sig_y):
                    print(f"  [!] {sid}: {x} and {y} differ in length; pair skipped")
                    continue
                rows += rc.pair_tests(sid, info["group"], x, y, sig_x, sig_y,
                                      preds[(sid, y)], info["starts"][y], lags)
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"[{model}] no tests computed")
            model_info[model] = {"status": "no results"}
            continue
        extra = [c for c in df.columns if c not in rc.NOTEBOOK_COLUMNS]
        df = df[rc.NOTEBOOK_COLUMNS + extra]
        df.insert(len(rc.NOTEBOOK_COLUMNS), "model", model)
        csv_path = out_dir / f"{_model_slug(model)}_residual_causality_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{model}] {len(df)} tests -> {csv_path}")
        all_rows.append(df)

        folder = bench._FOLDER[model]
        (out_dir / f"environment_{_model_slug(model)}.txt").write_text(
            _pip_freeze(bench._venv_python(folder)), encoding="utf-8")
        model_info[model] = {
            "status": "ok", "n_tests": int(len(df)),
            "runner_sha1": _sha1_file(bench._MODELS / folder / "runner.py"),
        }

    if all_rows:
        res = pd.concat(all_rows, ignore_index=True)
        rc.summarize_by_lag(res, args.alpha).to_csv(out_dir / "summary_by_lag.csv", index=False)
        rc.optimal_lag_table(res, args.alpha).to_csv(out_dir / "optimal_lag.csv", index=False)

    config = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "models": model_info,
        "lags_samples": lags,
        "lags_ms": [int(l * 1000 / rc.FS) for l in lags],
        "pairs": [f"{x}->{y}" for x, y in pairs],
        "group": "explicit subjects" if args.subjects else args.group,
        "subjects_requested": args.subjects,
        "limit": args.limit,
        "alpha": args.alpha,
        "device": args.device,
        **protocol_base,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()

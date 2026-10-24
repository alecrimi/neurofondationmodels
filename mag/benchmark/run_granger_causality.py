"""Run classical bivariate Granger tests; save one CSV and run_config.json.

Install beside benchmark/run_residual_causality.py. Install the core module
as benchmark/experiments/granger_causality.py. Uses the existing residual
module only for CLI parsing; no foundation models or model venvs are needed.

Examples:
  python benchmark/run_granger_causality.py --group all
  python benchmark/run_granger_causality.py --subjects sub-001 sub-002 --limit 2
  python benchmark/run_granger_causality.py --lags "[10,20,40]" --lag-unit ms
  python benchmark/run_granger_causality.py --pipeline loreta --pairs "src_precuneus_lh->src_superiorfrontal_lh"

Default: 5 disjoint 512+64 windows, offset 1500, z-score fitted on samples
1500 onward, one pooled test over 320 horizon observations per pair/lag.
CSV p-values are nominal/unadjusted. No stationarity test, differencing or
multiple-testing correction is applied automatically.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (str(_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark.pipelines import AlzheimerLoader
from benchmark.experiments import residual_causality as rc
from benchmark.experiments import granger_causality as gc


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=_ROOT / "datasets" / "ds004504")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--group", default="all", help="all | ad | ftd | control")
    selection.add_argument("--subjects", nargs="+", help="Explicit subject IDs, e.g. sub-001 or 1")
    p.add_argument("--limit", type=int)
    p.add_argument("--pairs", help="Comma-separated X->Y pairs; required for loreta")
    p.add_argument("--lags", default=str(gc.DEFAULT_LAGS_SAMPLES))
    p.add_argument("--lag-unit", choices=["samples", "ms"], default="samples")
    p.add_argument("--pipeline", choices=["baseline", "loreta"], default="baseline")
    p.add_argument("--scaling", choices=["zscore", "raw"], default="zscore",
                   help="baseline: default zscore with statistics fitted after 3 s; loreta requires zscore")
    p.add_argument("--run-name", help="New output subdirectory; default timestamp")
    return p


def select_subjects(loader, group_code, subject_ids, limit):
    part = loader.participants.copy()
    part["participant_id"] = part["participant_id"].astype(str)
    if part["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant_id values in participants.tsv.")
    if subject_ids:
        wanted = list(dict.fromkeys(loader._normalize_id(s) for s in subject_ids))
        unknown = sorted(set(wanted) - set(part["participant_id"]))
        if unknown:
            raise ValueError(f"Unknown subjects: {unknown}")
        selected = part.set_index("participant_id").loc[wanted].reset_index()
    else:
        selected = part if group_code is None else part[part["Group"] == group_code]
    if limit is not None:
        selected = selected.head(limit)
    return selected[["participant_id", "Group"]]


def normalize(signal):
    signal = np.asarray(signal, dtype=np.float64)
    reference = signal[gc.OFFSET_SAMPLES:]
    if reference.size < 2 or not np.isfinite(reference).all():
        raise ValueError("Too few finite samples after the 3-second offset.")
    mean, std = float(reference.mean()), float(reference.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Zero or invalid post-offset standard deviation.")
    return (signal - mean) / std


class SignalSource:
    def __init__(self, dataset, pipeline, scaling):
        self.dataset, self.pipeline, self.scaling = Path(dataset), pipeline, scaling
        self.loader = AlzheimerLoader(str(self.dataset))
        self.loreta = None

    def load(self, sid, channels):
        derivative = self.dataset / "derivatives" / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        if not derivative.is_file():
            raise FileNotFoundError(f"Required derivatives EEG file is missing: {derivative}")
        if self.pipeline == "baseline":
            raw = self.loader.load_subject(sid, use_derivatives=True)
            signals = {ch: raw.get_data(picks=[ch])[0] for ch in sorted(channels) if ch in raw.ch_names}
        else:
            if self.loreta is None:
                from benchmark.pipelines.loreta import LoretaPipeline
                self.loreta = LoretaPipeline(str(self.dataset))
            parcels = self.loreta._process_parcels(sid)
            signals = {ch: values[0] for ch, values in parcels.items() if ch in channels}
        if not signals:
            raise ValueError("No requested channels/parcels available.")
        # Re-fitting an already z-scored parcel is an affine re-normalisation;
        # this also supports the older full-recording LORETA implementation.
        return {ch: normalize(sig) if self.scaling == "zscore" else np.asarray(sig, dtype=np.float64)
                for ch, sig in signals.items()}


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.pipeline == "loreta" and (args.pairs is None or args.scaling != "zscore"):
        parser.error("loreta requires explicit --pairs and --scaling zscore")
    try:
        lags = rc.parse_lags(args.lags, unit=args.lag_unit, fs=gc.FS,
                             n_obs=gc.NUM_WINDOWS * gc.HORIZON_LEN)
        if max(lags) > gc.CONTEXT_LEN or gc.NUM_WINDOWS * gc.HORIZON_LEN <= 2 * max(lags) + 1:
            raise ValueError("Lags exceed context length or leave no full-model degrees of freedom.")
        pairs = rc.parse_pairs(args.pairs)
        group = None if args.subjects else rc.resolve_group(args.group)
        source = SignalSource(args.dataset, args.pipeline, args.scaling)
        selected = select_subjects(source.loader, group, args.subjects, args.limit)
        if selected.empty:
            raise ValueError("No subjects selected.")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        parser.error("--run-name must be a single directory name")
    output_dir = _HERE / "results" / "granger_causality" / args.pipeline / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, str(Path(__file__).resolve())] + (list(argv) if argv is not None else sys.argv[1:]),
        "git_commit": git_commit(), "method": "bivariate_linear_granger_pooled_horizons",
        "dataset": str(args.dataset.resolve()), "pipeline": args.pipeline,
        "scaling": args.scaling, "standardization_scope": "post_offset" if args.scaling == "zscore" else None,
        "context_len": gc.CONTEXT_LEN, "horizon_len": gc.HORIZON_LEN,
        "num_windows": gc.NUM_WINDOWS, "offset_samples": gc.OFFSET_SAMPLES,
        "lags_samples": lags, "lags_ms": [v * 1000 / gc.FS for v in lags],
        "pairs": [f"{x}->{y}" for x, y in pairs],
        "group": args.group if not args.subjects else "explicit subjects",
        "subjects_requested": args.subjects, "limit": args.limit,
        "p_values": "nominal, unadjusted", "status": "running",
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    channels = {ch for pair in pairs for ch in pair}
    rows, n_subjects, skipped_subjects, failed_tests = [], 0, 0, 0
    for i, (_, subject) in enumerate(selected.iterrows(), 1):
        sid = str(subject["participant_id"])
        grp = "Unknown" if pd.isna(subject["Group"]) else str(subject["Group"])
        try:
            signals = source.load(sid, channels)
            starts = {ch: rc.window_starts(len(sig)) for ch, sig in signals.items()}
        except Exception as exc:
            skipped_subjects += 1
            print(f"WARNING: patient {sid} skipped: {exc}", flush=True)
            continue
        before = len(rows)
        problems = []
        missing = sorted(channels - set(signals))
        if missing:
            problems.append(f"missing channels/parcels {missing}")
        for x, y in pairs:
            if x not in signals or y not in signals:
                continue
            for lag in lags:
                try:
                    result = gc.pair_tests(sid, grp, x, y, signals[x], signals[y], starts[y], [lag])
                    for row in result:
                        row.update(pipeline=args.pipeline, scaling=args.scaling,
                                   offset_samples=gc.OFFSET_SAMPLES,
                                   standardization_scope="post_offset" if args.scaling == "zscore" else "none")
                    rows.extend(result)
                except ValueError as exc:
                    failed_tests += 1
                    problems.append(f"{x}->{y}, lag={lag}: {exc}")
        if len(rows) > before:
            n_subjects += 1
        else:
            skipped_subjects += 1
        print(f"[{i}/{len(selected)}] {sid} ({grp}): {len(rows)-before} tests", flush=True)
        if problems:
            print(f"WARNING: patient {sid}: " + "; ".join(problems), flush=True)
    df = pd.DataFrame(rows, columns=gc.RESULT_COLUMNS)
    csv_path = output_dir / "granger_causality_results.csv"
    df.to_csv(csv_path, index=False)
    config.update(status="ok" if rows else "no_results", n_tests=len(rows),
                  n_subjects_with_results=n_subjects, n_subjects_skipped=skipped_subjects,
                  n_failed_tests=failed_tests)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} tests: {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Single entry point for multichannel preparation and covariate forecasting.

TimesFM and Chronos-2 retain their existing templates. Moirai predicts one
explicit target using only the historical contexts of covariate channels.
Use --prepare-only for data preparation without model dependencies.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd

_HERE=Path(__file__).resolve().parent
_ROOT=_HERE.parent
for entry in (str(_ROOT),str(_HERE)):
    if entry not in sys.path:sys.path.insert(0,entry)
from benchmark.multivariate import aligned_windows,prepare_inputs,evaluate
from benchmark.pipelines import CHANNELS,HORIZON_LEN

FOLDERS={'TimesFM':'timesfm','Chronos-2':'chronos2','Moirai':'moirai'}


def main(argv=None):
    """Reuse EEG pipelines, dispatch one model adapter and save common outputs."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=list(FOLDERS),default='TimesFM')
    p.add_argument('--pipeline',choices=['baseline','loreta'],default='baseline')
    p.add_argument('--dataset',type=Path,default=_ROOT/'datasets'/'ds004504')
    p.add_argument('--channels',nargs='+',help='Ordered input channels for TimesFM/Chronos-2 templates')
    p.add_argument('--target',help='Moirai: channel to forecast')
    p.add_argument('--covariates',nargs='+',help='Moirai: historical covariate channels')
    p.add_argument('--n',type=int)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--num-samples',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-name')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args(argv)
    if args.n is not None and args.n<1:p.error('--n must be positive')
    if args.num_samples<1:p.error('--num-samples must be positive')
    if args.model=='Moirai':
        if not args.target or not args.covariates or args.channels:
            p.error('Moirai requires --target and --covariates, instead of --channels')
        channels=[args.target,*args.covariates]
    else:
        if args.target or args.covariates:p.error('--target/--covariates currently apply to Moirai only')
        if args.pipeline=='loreta' and not args.channels:p.error('loreta requires explicit --channels')
        channels=args.channels or list(CHANNELS)
    if len(channels)<2 or len(set(channels))!=len(channels):p.error('Specify at least two distinct input channels')
    name=args.run_name or dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if Path(name).name!=name or name in ('.','..'):p.error('--run-name must be a single directory name')
    folder=FOLDERS[args.model]
    destination=_HERE/'results'/'multivariate'/args.pipeline/folder/name
    if destination.exists():p.error(f'Output directory already exists: {destination}')
    if args.pipeline=='baseline':
        from benchmark.pipelines.baseline import BaselinePipeline
        pipeline=BaselinePipeline(str(args.dataset))
    else:
        from benchmark.pipelines.loreta import LoretaPipeline
        pipeline=LoretaPipeline(str(args.dataset))
    subjects={}
    for sid in pipeline.get_subjects(args.n):
        sid=pipeline.loader._normalize_id(sid)
        try:
            derivative=args.dataset/'derivatives'/sid/'eeg'/f'{sid}_task-eyesclosed_eeg.set'
            if not derivative.is_file():raise FileNotFoundError(f'Required derivatives EEG file is missing: {derivative}')
            windows=aligned_windows(pipeline.process(sid),channels,HORIZON_LEN)
            candidate={sid:dict(group=pipeline.get_group(sid),windows=windows)}
            # Validate context scaling while failure can still be reported per patient.
            prepare_inputs(candidate,channels,HORIZON_LEN,args.device,args.target,args.num_samples,args.seed)
            subjects.update(candidate)
        except Exception as exc:
            print(f'WARNING: patient {sid} skipped: {exc}',flush=True)
    if not subjects:raise SystemExit('No aligned multivariate windows available.')
    payload,evaluation=prepare_inputs(subjects,channels,HORIZON_LEN,args.device,args.target,args.num_samples,args.seed)
    destination.mkdir(parents=True,exist_ok=False)
    config=dict(model=args.model,pipeline=args.pipeline,channels=channels,
                target=args.target,covariates=args.covariates,dataset=str(args.dataset.resolve()),
                device=args.device,horizon_len=HORIZON_LEN,status='prepared',
                standardization_scope='context_only' if args.target else 'existing_pipeline',
                seed=args.seed,num_samples=args.num_samples,subjects=list(subjects))
    config_path=destination/'run_config.json'
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    if args.prepare_only:
        with open(destination/'prepared_multivariate.pkl','wb') as f:
            pickle.dump(dict(inference=payload,evaluation=evaluation),f)
        print(f'Prepared {len(subjects)} patients; no inference: {destination}');return 0
    model_dir=_HERE/'models'/folder
    python=model_dir/'venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    adapter=model_dir/'runner_multivariate.py'
    try:
        if not python.is_file():raise FileNotFoundError(f'Model environment not found: {python}')
        with tempfile.TemporaryDirectory() as tmp:
            input_path,output_path=Path(tmp)/'input.pkl',Path(tmp)/'output.pkl'
            with open(input_path,'wb') as f:pickle.dump(payload,f)
            subprocess.run([str(python),str(adapter),'--input',str(input_path),'--output',str(output_path)],check=True)
            with open(output_path,'rb') as f:predictions=pickle.load(f)
        rows=evaluate(predictions,evaluation,channels,HORIZON_LEN,args.model,args.target)
        pd.DataFrame(rows).to_csv(destination/'multivariate_predictions_metrics.csv',index=False)
        with open(destination/'output.pkl','wb') as f:
            pickle.dump(dict(predictions=predictions,evaluation=evaluation),f)
        config.update(status='ok',n_rows=len(rows))
    except Exception:
        config['status']='failed';config_path.write_text(json.dumps(config,indent=2),encoding='utf-8');raise
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Saved {len(rows)} metrics: {destination}');return 0


if __name__=='__main__':raise SystemExit(main())

"""Moirai 1.0 adapter: one target with past-only covariates.

Input/output uses benchmark.multivariate's common pickle contract. Context
matrices contain only historical samples in payload['channels'] order.
The result has shape (windows,1,horizon); no covariate future is requested.
Model imports are lazy so input-contract tests require no model environment.
"""
from __future__ import annotations
import argparse
import inspect
import pickle
import numpy as np
import pandas as pd

MODEL_ID='Salesforce/moirai-1.0-R-base'


def build_entries(payload):
    """Return GluonTS records and (subject,window) order from past contexts.

    target is a 1D historical series, past_feat_dynamic_real is (K,T).
    Evaluation labels and future dynamic covariates are rejected/absent.
    """
    if payload.get('schema_version')!=1:raise ValueError('Unsupported multivariate input schema.')
    channels=payload['channels'];target=payload.get('target')
    if target not in channels or len(set(channels))!=len(channels) or len(channels)<2:
        raise ValueError('Moirai requires one target and distinct historical covariates.')
    target_idx=channels.index(target);cov_idx=[i for i,ch in enumerate(channels) if ch!=target]
    expected_cov=[channels[i] for i in cov_idx]
    if payload.get('covariates')!=expected_cov:raise ValueError('Covariate order does not match input channels.')
    entries,order=[],[];context_length=None
    for sid,subject in payload['subjects'].items():
        for i,w in enumerate(subject['windows']):
            if set(w)!={'context','start_idx'}:raise ValueError('Adapter windows must contain historical context and start_idx only.')
            a=np.asarray(w['context'],dtype=np.float32)
            if a.ndim!=2 or a.shape[0]!=len(channels) or a.shape[1]<1 or not np.isfinite(a).all():
                raise ValueError(f'{sid}/{i}: invalid historical context shape/values: {a.shape}.')
            if context_length is None:context_length=a.shape[1]
            if a.shape[1]!=context_length:raise ValueError('Context lengths must match.')
            entries.append(dict(item_id=f'{sid}__{i}',start=pd.Period('2000-01-01',freq='2ms'),
                                target=a[target_idx],past_feat_dynamic_real=a[cov_idx]))
            order.append((sid,i))
    if not entries:raise ValueError('No historical input windows available.')
    return entries,order,context_length


def predict_payload(payload):
    """Load the project checkpoint once and return sample-median forecasts."""
    entries,order,t=build_entries(payload)
    import torch
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast,MoiraiModule
    seed=int(payload.get('seed',42));torch.manual_seed(seed);np.random.seed(seed)
    device=payload.get('device','cpu')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable in the Moirai environment.')
        torch.cuda.manual_seed_all(seed)
    h=int(payload['horizon_len'])
    module=MoiraiModule.from_pretrained(MODEL_ID)
    model=MoiraiForecast(module=module,prediction_length=h,context_length=t,patch_size=32,
                        num_samples=int(payload.get('num_samples',20)),target_dim=1,
                        feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=len(payload['covariates']))
    if 'device' not in inspect.signature(model.create_predictor).parameters:
        raise RuntimeError('Installed uni2ts create_predictor lacks device; verify the installed API.')
    predictor=model.create_predictor(batch_size=1,device=torch.device(device))
    iterator=iter(predictor.predict(ListDataset(entries,freq='2ms',one_dim_target=True)))
    results={sid:{'predictions':[]} for sid in payload['subjects']}
    for sid,i in order:
        try:forecast=next(iterator)
        except StopIteration as exc:raise ValueError('Fewer forecasts returned than input windows.') from exc
        samples=np.asarray(forecast.samples)
        if samples.ndim==3 and samples.shape[-1]==1:samples=samples[...,0]
        if samples.ndim!=2 or samples.shape[1]!=h or not np.isfinite(samples).all():
            raise ValueError(f'{sid}/{i}: invalid forecast samples: {samples.shape}.')
        results[sid]['predictions'].append(np.median(samples,axis=0)[None,:])
    try:next(iterator)
    except StopIteration:pass
    else:raise ValueError('More forecasts returned than input windows.')
    for value in results.values():value['predictions']=np.stack(value['predictions'])
    return results


def main(argv=None):
    """Read trusted local contexts and write forecasts; no EEG loading."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    with open(args.input,'rb') as f:payload=pickle.load(f)
    result=predict_payload(payload)
    with open(args.output,'wb') as f:pickle.dump(result,f)


if __name__=='__main__':main()

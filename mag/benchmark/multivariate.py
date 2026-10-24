"""Shared window alignment, historical input preparation and evaluation.

Used by run_multivariate.py for templates and Moirai target-with-covariates.
No model-specific dependencies, EEG reading or additional window selection.
"""
from __future__ import annotations
import copy
import numpy as np


def aligned_windows(channel_data, channels, horizon_len):
    """Pack pipeline windows as channel-by-time arrays with identical starts.

    Returns windows containing context (C,T), targets (C,H), raw_std (C,)
    and start_idx. Channel order follows channels. Raises ValueError for
    missing channels, incompatible windows or invalid data/scales.
    """
    if len(channels)<2 or len(set(channels))!=len(channels):
        raise ValueError('Specify at least two distinct channels.')
    missing=[ch for ch in channels if ch not in channel_data]
    if missing:raise ValueError(f'Missing requested channels/parcels: {missing}.')
    counts=[len(channel_data[ch]['windows']) for ch in channels]
    if not counts[0] or len(set(counts))!=1:
        raise ValueError(f'Channel window counts must match and be positive: {counts}.')
    scales=np.asarray([channel_data[ch]['raw_std'] for ch in channels],dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Channel scales must be positive and finite.')
    packed=[]
    for i in range(counts[0]):
        ws=[channel_data[ch]['windows'][i] for ch in channels]
        starts=[int(w['start_idx']) for w in ws]
        if len(set(starts))!=1:
            raise ValueError(f'Channels are not time-aligned at window {i}: {starts}.')
        contexts=[np.asarray(w['context'],dtype=np.float32) for w in ws]
        targets=[np.asarray(w['target'],dtype=np.float32) for w in ws]
        if any(a.ndim!=1 for a in contexts+targets):
            raise ValueError('Context and target arrays must be one-dimensional per channel.')
        if len({len(a) for a in contexts})!=1 or not len(contexts[0]):
            raise ValueError('Context lengths must match and be positive.')
        if any(len(a)!=horizon_len for a in targets):
            raise ValueError('Target lengths do not match the requested horizon.')
        context,target=np.stack(contexts),np.stack(targets)
        if not np.isfinite(context).all() or not np.isfinite(target).all():
            raise ValueError('Non-finite context or target values.')
        packed.append(dict(start_idx=starts[0],context=context,targets=target,raw_std=scales.copy()))
    return packed


def forecast_payload(subjects,channels,horizon_len,device):
    """Build the original context-only template payload; no evaluation labels."""
    return dict(schema_version=1,channels=list(channels),horizon_len=horizon_len,
                device=device,subjects={sid:dict(group=info['group'],windows=[
                    dict(start_idx=w['start_idx'],context=w['context']) for w in info['windows']])
                    for sid,info in subjects.items()})


def prepare_inputs(subjects,channels,horizon_len,device,target=None,num_samples=20,seed=42):
    """Prepare shared input and separate evaluation data.

    With target=None the original multivariate template contract is retained.
    With target specified, restandardize every channel using only its own
    window context. Transform labels with those fixed parameters and adjust
    raw_std so mse_original reverses both affine transforms. This cancels
    earlier recording-level z-scaling up to floating-point rounding, not
    earlier offline filtering. The output channel list then contains only
    the target, while context matrices retain target and all covariates.
    """
    evaluation=copy.deepcopy(subjects)
    if target is not None:
        if target not in channels:raise ValueError('Target must belong to input channels.')
        for sid,info in evaluation.items():
            for w in info['windows']:
                a=np.asarray(w['context'],dtype=np.float64)
                mu=a.mean(axis=1);std=a.std(axis=1,ddof=0)
                if not np.isfinite(std).all() or np.any(std<=0):
                    raise ValueError(f'{sid}: zero or invalid context standard deviation.')
                w['context']=((a-mu[:,None])/std[:,None]).astype(np.float32)
                w['targets']=(np.asarray(w['targets'],dtype=np.float64)-mu[:,None])/std[:,None]
                w['raw_std']=np.asarray(w['raw_std'],dtype=np.float64)*std
    payload=forecast_payload(evaluation,channels,horizon_len,device)
    if target is not None:
        payload.update(target=target,covariates=[ch for ch in channels if ch!=target],
                       output_channels=[target],standardization_scope='context_only',
                       num_samples=num_samples,seed=seed)
    return payload,evaluation


def evaluate(predictions,subjects,channels,horizon_len,model_name,target=None):
    """Validate forecasts (W,C_out,H) and return per-window/channel MSE rows.

    Existing template calls retain C_out=len(channels). If target is supplied,
    C_out=1. mse_original is in squared pre-z-score pipeline units.
    """
    if set(predictions)!=set(subjects):raise ValueError('Adapter output subject IDs do not match the input.')
    output_channels=[target] if target is not None else channels
    rows=[]
    for sid,info in subjects.items():
        pred=np.asarray(predictions[sid]['predictions'],dtype=np.float64)
        expected=(len(info['windows']),len(output_channels),horizon_len)
        if pred.shape!=expected or not np.isfinite(pred).all():
            raise ValueError(f'{sid}: invalid forecast shape/values; expected {expected}, got {pred.shape}.')
        for i,w in enumerate(info['windows']):
            for j,ch in enumerate(output_channels):
                c=channels.index(ch)
                mse=float(np.mean((pred[i,j]-w['targets'][c])**2))
                rows.append(dict(subject_id=sid,group=info['group'],model=model_name,
                                 channel=ch,input_channels=','.join(channels),
                                 window_index=i,start_idx=w['start_idx'],mse_norm=mse,
                                 mse_original=mse*float(w['raw_std'][c])**2))
    return rows

"""Aggregate saved experiment results and generate PNG figures.

Run from the project root: python benchmark/analyze_results.py
Requires numpy, pandas and matplotlib. Does not run prediction models.
Reads supported CSVs only, selects the newest valid file per configuration,
averages windows within subjects, and keeps missing measurements as NaN.
Zero-height N/A bars are display placeholders, never statistical observations.
No figure is generated unless it contains at least one real measurement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=['Chronos','Chronos-2','TimesFM','Moirai','Lag-Llama','Sundial','ViTime','TimeFound','TimeGPT']
PIPES=['baseline','loreta','loreta_gsp']
GROUPS=['A','C','F']
GROUP_NAMES={'A':'AD','C':'Control','F':'FTD'}
ALIASES={re.sub('[^a-z0-9]','',s.lower()):s for s in MODELS+['Granger']}
ALIASES['chronosbolt']='Chronos-2'


def model_name(name):
    return ALIASES.get(re.sub('[^a-z0-9]','',str(name).lower()),str(name))


def filename(text):
    return re.sub('[^a-zA-Z0-9_-]+','_',str(text))[:80]+'_'+hashlib.sha1(str(text).encode()).hexdigest()[:8]


def has_data(values):
    """True for any finite measurement, including a genuine measured zero."""
    return bool(np.isfinite(np.asarray(values,dtype=float)).any())


def col(df,names,required=True):
    for name in names:
        if name in df:return name
    if required:raise ValueError(f'Missing column: expected one of {names}.')
    return None


def text_value(df,name,fallback):
    if name not in df:return str(fallback)
    vals=df[name].dropna().astype(str).unique()
    if len(vals)>1:raise ValueError(f'Mixed {name} in one CSV: {vals.tolist()}')
    return str(vals[0]) if len(vals) else str(fallback)


def metadata(path):
    p=path.parent/'run_config.json'
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}


def configuration(config):
    keys=['dataset','context_len','context_length','horizon_len','num_windows','offset_samples',
          'scaling','standardization_scope','zscore_reference','preprocessing_version','checkpoint','num_samples']
    result={k:config[k] for k in keys if k in config}
    if 'context_length' in result:result['context_len']=result.pop('context_length')
    return json.dumps(result,sort_keys=True,default=str)


def parse_forecast(df,parts,config):
    """Normalize supported univariate and multivariate metric tables."""
    if 'record_type' in df:df=df[df.record_type=='per_patient_electrode'].copy()
    if df.empty:raise ValueError('No individual-subject rows.')
    sid=col(df,['subject_id','subject']);channel=col(df,['channel','target_Y','electrode','target'])
    multi=parts[0] in ('multivariate','covariates')
    pipe=parts[1] if multi else parts[0]
    if pipe not in PIPES:raise ValueError('Unknown pipeline.')
    model=model_name(text_value(df,'model',config.get('model',parts[2] if multi else parts[1])))
    scope=text_value(df,'standardization_scope',config.get('standardization_scope',config.get('zscore_reference','unspecified')))
    inputs=''
    if multi:
        field=col(df,['input_channels','covariates','covariate_used'],False)
        inputs=' | '.join(sorted(df[field].dropna().astype(str).unique())) if field else ','.join(config.get('covariates',config.get('channels',[])))
        inputs=inputs or 'unspecified'
    target=str(config.get('target') or (','.join(sorted(df[channel].dropna().astype(str).unique())) if multi else 'all'))
    mode='multivariate' if multi else 'univariate'
    variant=f'{pipe} / {mode} / scaling={scope}'
    if multi:variant+=f' / targets={target} / inputs={inputs}'
    result=pd.DataFrame({'subject_id':df[sid].astype(str),'channel':df[channel].astype(str),
                         'group':df.group.fillna('Unknown').astype(str) if 'group' in df else 'Unknown'})
    for metric in ('mse_norm','mse_original'):
        field=metric if metric in df else ('mse_phys' if metric=='mse_original' and 'mse_phys' in df else None)
        result[metric]=pd.to_numeric(df[field],errors='coerce') if field else np.nan
        bad=result[metric].notna() & (~np.isfinite(result[metric]) | (result[metric]<0))
        if bad.any():raise ValueError(f'Invalid {metric}.')
    if not has_data(result[['mse_norm','mse_original']]):raise ValueError('No finite forecast metric.')
    result['model']=model;result['pipeline']=pipe;result['mode']=mode;result['variant']=variant
    key=('forecast',pipe,model,mode,target,inputs,scope,configuration(config))
    return result,key


def parse_causality(df,parts,config):
    """Read one pooled result per subject/pair/lag, not window-level tests."""
    kind,pipe=parts[:2]
    fcol=col(df,['f_stat_residual','f_stat_granger','f_stat'])
    pcol=col(df,['p_value_residual','p_value_granger','p_value'])
    for name in ['subject_id','covariate_X','target_Y']:
        if name not in df:raise ValueError(f'Missing {name}.')
    if 'window_index' in df:raise ValueError('Per-window causality output is not supported by this pooled-result analysis.')
    result=pd.DataFrame({'subject_id':df.subject_id.astype(str),
                         'group':df.group.fillna('Unknown').astype(str) if 'group' in df else 'Unknown',
                         'pair':df.covariate_X.astype(str)+' -> '+df.target_Y.astype(str),
                         'F':pd.to_numeric(df[fcol],errors='coerce'),'p':pd.to_numeric(df[pcol],errors='coerce')})
    field=col(df,['lag_ms','lag_steps'])
    result['lag_ms']=pd.to_numeric(df[field],errors='coerce')*(2 if field=='lag_steps' else 1)
    if not np.isfinite(result[['F','p','lag_ms']].to_numpy()).all() or not result.p.between(0,1).all() or (result.F<0).any():raise ValueError('Invalid F/p/lag values.')
    if result.duplicated(['subject_id','pair','lag_ms']).any():raise ValueError('Duplicate subject/pair/lag tests.')
    fallback='Granger' if kind=='granger_causality' else parts[-1].split('_residual_causality_results')[0]
    model=model_name(text_value(df,'model',fallback))
    scope=text_value(df,'standardization_scope',config.get('standardization_scope',config.get('zscore_reference','unspecified')))
    scaling=text_value(df,'scaling',config.get('scaling','unspecified'))
    result['model']=model;result['pipeline']=pipe;result['kind']=kind
    result['scope']=scope;result['scaling']=scaling
    key=(kind,pipe,model,scope,scaling,configuration(config),tuple(sorted(result.lag_ms.unique())))
    return result,key


def load_results(root):
    """Read supported metric CSVs and select the newest run per configuration."""
    items=[];manifest=[]
    for path in sorted(root.rglob('*.csv')):
        parts=path.relative_to(root).parts
        if len(parts)<3 or any(p.startswith('_') for p in parts):continue
        ordinary=parts[0] in PIPES and len(parts)==3 and path.name.endswith('_metrics.csv')
        multi=parts[0] in ('multivariate','covariates') and path.name.endswith('metrics.csv')
        causal=parts[0] in ('residual_causality','granger_causality') and path.name.endswith('causality_results.csv')
        if not (ordinary or multi or causal):continue
        record={'path':str(path.resolve()),'status':'','reason':'','n_subjects':0}
        manifest.append(record)
        try:
            config=metadata(path)
            if config.get('status') in ('failed','running','prepared','no_results'):raise ValueError(f'Run status: {config["status"]}.')
            df=pd.read_csv(path)
            if df.empty:raise ValueError('Empty CSV.')
            frame,key=(parse_causality if causal else parse_forecast)(df,parts,config)
            try:stamp=pd.Timestamp(config['created']).timestamp() if config.get('created') else path.stat().st_mtime
            except Exception:stamp=path.stat().st_mtime
            frame['source_file']=str(path.resolve())
            token=hashlib.sha1(repr(key).encode()).hexdigest()[:7];frame['protocol_id']=token
            record['n_subjects']=frame.subject_id.nunique()
            items.append(dict(frame=frame,key=key,stamp=stamp,path=str(path),record=record))
        except Exception as exc:record.update(status='skipped',reason=str(exc))
    latest={}
    for item in items:
        old=latest.get(item['key'])
        if old is None or (item['stamp'],item['path'])>(old['stamp'],old['path']):latest[item['key']]=item
    forecasts=[];causes=[]
    for item in items:
        if latest[item['key']] is not item:
            item['record'].update(status='superseded',reason='Newer valid run for the same configuration.');continue
        item['record'].update(status='selected',reason='Newest valid run for this configuration.')
        (forecasts if item['key'][0]=='forecast' else causes).append(item['frame'])
    f=pd.concat(forecasts,ignore_index=True) if forecasts else pd.DataFrame()
    c=pd.concat(causes,ignore_index=True) if causes else pd.DataFrame()
    if not f.empty:
        for (m,v),g in f.groupby(['model','variant']):
            if g.protocol_id.nunique()>1:
                mask=(f.model==m)&(f.variant==v)
                f.loc[mask,'variant']=f.loc[mask,'variant']+' / protocol='+f.loc[mask,'protocol_id']
    return f,c,pd.DataFrame(manifest)


def forecast_aggregates(frame,metric,models):
    """Average windows, then channels, then subjects; keep missing slots empty."""
    keys=['model','variant','pipeline','mode','subject_id','group','channel']
    channel=frame.groupby(keys,dropna=False)[['mse_norm','mse_original']].mean().reset_index()
    subject=channel.groupby(keys[:-1],dropna=False)[['mse_norm','mse_original']].mean().reset_index()
    variants=sorted(subject.variant.unique())
    for pipe in ['baseline','loreta']:
        for mode in ['univariate','multivariate']:
            if subject[(subject.pipeline==pipe)&(subject['mode']==mode)].empty:variants.append(f'{pipe} / {mode} / no data')
    variants=sorted(variants);rows=[]
    for group in ['ALL',*GROUPS]:
        sub=subject if group=='ALL' else subject[subject.group==group]
        for model in models:
            for variant in variants:
                values=sub[(sub.model==model)&(sub.variant==variant)][metric].dropna()
                available=bool(len(values))
                rows.append(dict(model=model,variant=variant,clinical_group=group,metric=metric,
                                 mean=float(values.mean()) if available else np.nan,
                                 n_subjects=len(values),available=available,
                                 plot_value=float(values.mean()) if available else 0.))
    return channel,subject,pd.DataFrame(rows)


def causality_aggregates(frame,alpha):
    rows=[];keys=['kind','pipeline','model','scope','scaling','protocol_id']
    for vals,part in frame.groupby(keys,dropna=False):
        meta=dict(zip(keys,vals))
        for group in ['ALL',*GROUPS]:
            sub=part if group=='ALL' else part[part.group==group]
            for (pair,lag),g in sub.groupby(['pair','lag_ms']):
                rows.append(dict(**meta,clinical_group=group,pair=pair,lag_ms=lag,
                                 median_F=g.F.median(),significant_pct=float((g.p<alpha).mean()*100),n_subjects=g.subject_id.nunique()))
    return pd.DataFrame(rows)


def figure_jobs(channel,subject,grid,causal,metric,models,alpha):
    """Build only nonempty plot specifications; never turn absent data into evidence."""
    jobs=[]
    if grid is not None and not grid.empty:
        variants=sorted(grid.variant.unique())
        for group in ['ALL',*GROUPS]:
            for i in range(0,len(variants),6):
                batch=variants[i:i+6]
                sub=grid[(grid.clinical_group==group)&grid.variant.isin(batch)]
                if sub.available.any():
                    jobs.append(dict(kind='bars',name=f'forecast_bars_{group}_{i//6+1}',
                                     title=f'{group}: mean {metric} across subjects',table=sub,
                                     variants=batch,models=models,label=metric+' (scaling and coverage may differ)'))
        for variant,g in channel.groupby('variant'):
            matrix=g.pivot_table(index='model',columns='channel',values=metric,aggfunc='mean').reindex(index=models)
            if has_data(matrix):jobs.append(dict(kind='heat',name='forecast_channels_'+filename(variant),title=variant,table=matrix,label=metric,percent=False))
            s=subject[subject.variant==variant]
            matrix=s.groupby(['model','group'])[metric].mean().unstack('group').reindex(index=models,columns=GROUPS)
            if has_data(matrix):jobs.append(dict(kind='heat',name='forecast_groups_'+filename(variant),title=variant+' / clinical groups',table=matrix,label=metric,percent=False))
    if causal is not None and not causal.empty:
        keys=['kind','pipeline','model','scope','scaling','protocol_id']
        for vals,g in causal.groupby(keys,dropna=False):
            title=' / '.join(map(str,vals));pairs=sorted(g.pair.unique());lags=sorted(g.lag_ms.unique())
            for group in ['ALL',*GROUPS]:
                sub=g[g.clinical_group==group]
                for value,percent in [('median_F',False),('significant_pct',True)]:
                    matrix=sub.pivot(index='pair',columns='lag_ms',values=value).reindex(index=pairs,columns=lags)
                    if has_data(matrix):
                        label=f'Nominal significant subjects (%), p < {alpha:g}' if percent else 'Median F statistic'
                        jobs.append(dict(kind='heat',name='causality_'+filename(title)+'_'+group+'_'+value,
                                         title=title+' / '+group,table=matrix,label=label,percent=percent))
            sub=g[g.clinical_group=='ALL']
            if has_data(sub.significant_pct):jobs.append(dict(kind='line',name='causality_lag_profile_'+filename(title),title=title,table=sub))
    return jobs


def render(jobs,destination):
    """Save PNGs only. No directory is created if there are no figures."""
    if not jobs:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    destination.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        if job['kind']=='bars':
            table=job['table'];variants=job['variants'];models=job['models']
            # Missing placeholders never decide whether a chart contains data.
            if not table.available.any():continue
            fig,ax=plt.subplots(figsize=(13,5.8));x=np.arange(len(models));width=.82/len(variants)
            colors=plt.get_cmap('tab20')
            for i,variant in enumerate(variants):
                sub=table[table.variant==variant].set_index('model').reindex(models)
                pos=x+(i-(len(variants)-1)/2)*width
                ax.bar(pos,sub.plot_value.fillna(0),width,label=variant,color=colors(i%20),edgecolor='black',linewidth=.35)
                missing=~sub.available.fillna(False).astype(bool).to_numpy()
                for point in pos[missing]:ax.text(point,.01,'N/A',transform=ax.get_xaxis_transform(),rotation=90,ha='center',va='bottom',fontsize=5,color='gray')
            ax.set_xticks(x,models,rotation=25,ha='right');ax.set_ylim(bottom=0)
            ax.set_ylabel(job['label']);ax.legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.01,1));ax.grid(axis='y',alpha=.2)
            fig.text(.01,.005,'N/A = no data; zero-height display placeholder, not a measured zero.',fontsize=7)
        elif job['kind']=='heat':
            matrix=job['table'];a=matrix.to_numpy(dtype=float)
            if not has_data(a):continue
            fig,ax=plt.subplots(figsize=(max(7,.65*len(matrix.columns)),max(3.5,.4*len(matrix.index))))
            cmap=plt.get_cmap('Blues' if job['percent'] else 'YlOrRd').copy();cmap.set_bad('#eeeeee')
            valid=a[np.isfinite(a)];vmax=100 if job['percent'] else max(float(valid.max()),np.finfo(float).eps)
            im=ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cmap,vmin=0,vmax=vmax)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    value=a[i,j];text='N/A' if not np.isfinite(value) else (f'{value:.1f}' if job['percent'] else f'{value:.2g}')
                    ax.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if np.isfinite(value) and value>.65*vmax else 'black')
            ax.set_xticks(range(len(matrix.columns)),[str(v) for v in matrix.columns],rotation=35,ha='right')
            ax.set_yticks(range(len(matrix.index)),matrix.index)
            fig.colorbar(im,ax=ax,label=job['label'])
        else:
            table=job['table']
            if not has_data(table.significant_pct):continue
            fig,ax=plt.subplots(figsize=(9,4.5))
            for pair,g in table.groupby('pair'):
                g=g.sort_values('lag_ms');ax.plot(g.lag_ms,g.significant_pct,marker='o',label=pair)
            ax.set(xlabel='Lag (ms)',ylabel='Nominal significant subjects (%)',ylim=(0,100));ax.legend(fontsize=7);ax.grid(alpha=.2)
        ax.set_title(job['title']);fig.tight_layout()
        fig.savefig(destination/(job['name']+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--output',type=Path)
    p.add_argument('--metric',choices=['mse_norm','mse_original'],default='mse_norm')
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--tables-only',action='store_true')
    args=p.parse_args(argv)
    if not args.results.is_dir():p.error(f'Results directory does not exist: {args.results}')
    if not 0<args.alpha<1:p.error('--alpha must be between 0 and 1')
    forecast,cause,manifest=load_results(args.results)
    if forecast.empty and cause.empty:
        print('No usable numerical results. No figures generated.')
        for entry in manifest.to_dict('records'):print(f'{entry["status"]}: {entry["path"]}: {entry["reason"]}')
        return 0
    dest=args.output or args.results/'final_analysis'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if dest.exists() and any(dest.iterdir()):p.error('Choose a new/empty output directory.')
    dest.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(dest/'source_manifest.csv',index=False)
    models=MODELS+sorted(set(forecast.model if not forecast.empty else [])-set(MODELS))
    channels=subjects=grid=None
    if not forecast.empty:
        channels,subjects,grid=forecast_aggregates(forecast,args.metric,models)
        channels.to_csv(dest/'forecast_subject_channel_means.csv',index=False)
        subjects.to_csv(dest/'forecast_subject_means.csv',index=False)
        grid.to_csv(dest/'forecast_overall_and_groups.csv',index=False)
    causal=causality_aggregates(cause,args.alpha) if not cause.empty else None
    if causal is not None:causal.to_csv(dest/'causality_summary.csv',index=False)
    jobs=figure_jobs(channels,subjects,grid,causal,args.metric,models,args.alpha)
    if not args.tables_only:render(jobs,dest/'figures')
    pd.DataFrame([dict(name=j['name'],kind=j['kind'],title=j['title']) for j in jobs],
                 columns=['name','kind','title']).to_csv(dest/'figure_manifest.csv',index=False)
    print(f'Selected files: {(manifest.status=="selected").sum()}')
    print(f'PNG figure specifications: {len(jobs)}'+(' (tables-only; not rendered)' if args.tables_only else ''))
    print(f'Saved aggregate results: {dest.resolve()}')
    print('Overview MSE comparisons are descriptive; inspect scaling labels, subject counts and target coverage.')
    return 0


if __name__=='__main__':raise SystemExit(main())

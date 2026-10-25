"""Pure deterministic ranking utilities shared by both notebooks."""
import numpy as np
import pandas as pd

def stable_rank(values):
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all(), 'Nonfinite candidate scores'
    order=np.argsort(-values,kind='stable');ranks=np.empty(len(values),int);ranks[order]=np.arange(1,len(values)+1)
    return ranks

def rrf_order(neural_scores,weight=0.15,offset=3.0):
    assert weight>=0 and offset>=0
    n=len(neural_scores);baseline=np.arange(1,n+1)
    # Zero weight guarantees exact baseline order even when neural scores are unavailable.
    if weight==0:return np.arange(n)
    ranks=stable_rank(neural_scores)
    return np.argsort(-(1/(offset+baseline)+weight/(offset+ranks)),kind='stable')

def metrics(frame,score='prediction'):
    out=[]
    for mid,g in frame.groupby('molecule_id',sort=False):
        g=g.sort_values('baseline_rank',kind='stable')
        order=np.argsort(-g[score].to_numpy(),kind='stable')
        labels=g.label.to_numpy()[order];where=np.flatnonzero(labels)
        r=int(where[0]+1) if len(where) else None
        out.append(dict(molecule_id=mid,mrr25=1/r if r and r<=25 else 0.,top1=int(r==1),
                        candidate_recall=int(bool(len(where))),rank=r))
    return pd.DataFrame(out)

BASE_FEATURES=['baseline_rr','main_rr','prefusion_rr','engine2_rr','pubchem_rr','engine_agreement',
    'library_max','proxy_gl','ice_missing','gl_missing','ice_z','gl_z',
    'neural_cos_z','neural_fp_z','neural_score_z','neural_rr','neural_std',
    'n_spectra_log','n_adducts','neural_top_gap','neural_agrees_top1','engine2_agrees_top1',
    'neural_delta_rank','neural_x_library','neural_x_agreement','neural_x_sim_missing',
    'main_missing','engine2_missing','pubchem_missing']

def make_features(frame):
    f=frame.copy()
    for src in ['baseline','main','prefusion','engine2','pubchem']:
        r=pd.to_numeric(f[src+'_rank'],errors='coerce');f[src+'_rr']=np.where(r>0,1/(3+r),0)
        if src in ['main','engine2','pubchem']:f[src+'_missing']=(~(r>0)).astype(float)
    for src in ['ice','gl','neural_cos','neural_fp','neural_score']:
        v=pd.to_numeric(f[src],errors='coerce');v=v.where(np.isfinite(v))
        if src in ['ice','gl']:f[src+'_missing']=v.isna().astype(float)
        grp=v.groupby(f.molecule_id,sort=False)
        f[src+'_z']=((v-grp.transform('mean'))/grp.transform(lambda s:s.std(ddof=0)).clip(lower=1e-6)).fillna(0).clip(-5,5)
    f['neural_rank']=0.;f['neural_top_gap']=0.;f['neural_agrees_top1']=0.;f['engine2_agrees_top1']=0.
    for _,g in f.groupby('molecule_id',sort=False):
        idx=g.index;rank=stable_rank(g.neural_score.values);f.loc[idx,'neural_rank']=rank
        vals=np.sort(g.neural_score_z.values)[::-1]
        f.loc[idx,'neural_top_gap']=float(vals[0]-vals[1]) if len(vals)>1 else 0.
        f.loc[idx,'neural_agrees_top1']=float(g.iloc[np.argmin(rank)].baseline_rank==1)
        f.loc[idx,'engine2_agrees_top1']=float(((g.baseline_rank==1)&(g.engine2_rank==1)).any())
    f['neural_rr']=1/(3+f.neural_rank)
    f['engine_agreement']=((f.main_rank>0).astype(int)+(f.engine2_rank>0).astype(int)+(f.pubchem_rank>0).astype(int))
    f['n_spectra_log']=np.log1p(f.n_spectra)
    f['neural_delta_rank']=(f.baseline_rank-f.neural_rank)/25
    f['neural_x_library']=f.neural_score_z*f.library_max
    f['neural_x_agreement']=f.neural_score_z*f.engine_agreement
    f['neural_x_sim_missing']=f.neural_score_z*f.ice_missing*f.gl_missing
    X=f[BASE_FEATURES].astype(float).replace([np.inf,-np.inf],np.nan).fillna(0.)
    assert np.isfinite(X.values).all()
    return X

def residual_scores(frame,probability,beta):
    p=np.clip(np.asarray(probability,float),1e-5,1-1e-5)
    raw=pd.Series(np.log(p/(1-p)),index=frame.index)
    grp=raw.groupby(frame.molecule_id,sort=False)
    z=((raw-grp.transform('mean'))/grp.transform(lambda s:s.std(ddof=0)).clip(lower=1e-6)).fillna(0).clip(-5,5)
    correction=float(beta)*z.values
    if 'neural_ok' in frame:correction=np.where(frame.neural_ok.values,correction,0.)
    return -np.log(frame.baseline_rank.values.astype(float))+correction

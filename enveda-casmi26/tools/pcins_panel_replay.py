import sys, numpy as np, pandas as pd
D=sys.argv[1]
mc=pd.read_parquet(D+'/main_candidates.parquet')
print(mc.scenario.value_counts().to_dict())
c=pd.read_parquet(D+'/pubchem_candidates.parquet', columns=['molecule','connectivity','scorer_key','is_truth','in_pool','log_substances','log_pubmed'])
f=pd.read_parquet(D+'/panel_fpnet_scores.parquet', columns=['ab_full','fold0_full'])
s=pd.read_parquet(D+'/pubchem_signals.parquet', columns=['same_formula'])
c=pd.concat([c.reset_index(drop=True),f.reset_index(drop=True),s.reset_index(drop=True)],axis=1)
c=c[np.isfinite(c.fold0_full)&np.isfinite(c.ab_full)]
c['ident']=np.where(c.scorer_key!='',c.scorer_key,c.connectivity)
g=c.groupby('molecule')
for col in ['fold0_full','ab_full']:
    c['z_'+col]=(c[col]-g[col].transform('mean'))/(g[col].transform('std')+1e-9)
c['pop']=c.log_substances+c.log_pubmed
tt=mc[mc.is_truth].drop_duplicates('molecule')
tform=dict(zip(tt.molecule,tt.formula))
print('truth formulas',len(tform))
pcg={m:x for m,x in c[c.same_formula.astype(bool)].groupby('molecule')}
def rr(lst,truthkey):
    for i,k in enumerate(lst[:25]):
        if k==truthkey: return 1/(i+1)
    return 0.0
for scen in sorted(mc.scenario.unique()):
    m=mc[mc.scenario==scen]
    res={}
    for mol,x in m.groupby('molecule'):
        x=x.sort_values('rank'); keys=list(x.candidate_key); forms=list(x.formula)
        lib=float(x.lib_max.iloc[0]) if 'lib_max' in x else 0
        res[mol]=(keys,forms,lib)
    mols=list(res)
    out={}
    for name,score in [('base',None),('S=zAB+.5z0+.25pop',lambda p:(p.z_ab_full+0.5*p.z_fold0_full+0.25*p['pop'])),
                       ('fold0 only',lambda p:p.z_fold0_full),('AB only',lambda p:p.z_ab_full)]:
        for K,N in ([(25,0)] if score is None else [(10,5),(5,5),(15,5),(10,10)]):
            v=[]
            for mol in mols:
                keys,forms,lib=res[mol]
                tk=mol
                lst=list(keys)
                if score is not None and lib<0.9 and mol in pcg and forms and forms[0]==tform.get(mol):
                    p=pcg[mol]; p=p[~p.in_pool | p.is_truth] if scen=='structure_removed_from_pool' else p[~p.in_pool]
                    p=p.assign(sc=score(p)).sort_values('sc',ascending=False)
                    seen=set(lst); ins=[]
                    for k,t in zip(p.ident,p.is_truth):
                        kk=mol if t else k
                        if kk in seen: continue
                        seen.add(kk); ins.append(kk)
                        if len(ins)>=N: break
                    lst=(lst[:K]+ins+lst[K:])[:25]
                v.append(rr(lst,tk))
            out[f'{name} K{K}N{N}']=np.mean(v)
    print(scen, len(mols)); [print(f'   {k:32s} {v:.4f}') for k,v in out.items()]

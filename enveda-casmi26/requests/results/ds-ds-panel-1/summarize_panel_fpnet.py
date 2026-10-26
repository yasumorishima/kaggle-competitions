"""Reproduce the published fold-safe panel summary from cached per-query results.

No inference, RDKit or model weights are needed. Inputs are in the panel or
fold-safe dataset. The public panel is development evidence, not hidden-test
validation. Cached rows missing for an arm are padded with zero to all 250 IDs.
"""
import argparse,json,pathlib
import numpy as np,pandas as pd
def summarize(per_query,panel_ids):
 d=pd.read_csv(per_query);ids=pd.read_parquet(panel_ids).molecule.astype(str).tolist();assert len(ids)==250 and len(set(ids))==250
 rng=np.random.default_rng(20261019);boot=rng.integers(0,len(ids),size=(20000,len(ids)));out=[]
 assert not d.duplicated(['molecule','mode','weight']).any()
 for (mode,w),g in d.groupby(['mode','weight']):
  a=g.set_index('molecule').reindex(ids);a[['mrr','top1','truth_shortlisted']]=a[['mrr','top1','truth_shortlisted']].fillna(0)
  b=d[(d['mode']==mode)&(d.weight==0)].set_index('molecule').reindex(ids)
  rec={'mode':mode,'weight':float(w),'n':len(ids),'mrr25':float(a.mrr.mean()),'top1':float(a.top1.mean()),'truth_shortlisted':float(a.truth_shortlisted.astype(float).mean())}
  for metric in ['mrr','top1']:
   v=a[metric].to_numpy(float);rec[metric+'_ci95']=np.quantile(v[boot].mean(1),[.025,.975]).tolist()
   if w>=0:
    delta=v-b[metric].fillna(0).to_numpy(float);rec['delta_'+metric]=float(delta.mean());rec['delta_'+metric+'_ci95']=np.quantile(delta[boot].mean(1),[.025,.975]).tolist()
    rec['improved_'+metric]=int((delta>0).sum());rec['worse_'+metric]=int((delta<0).sum())
  if w>=0:rec['top1_identity_changed']=int((a.top_identity.fillna('')!=b.top_identity.fillna('')).sum())
  out.append(rec)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--per-query',type=pathlib.Path,required=True);p.add_argument('--panel',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 a.out.write_text(json.dumps(summarize(a.per_query,a.panel),indent=2)+'\n')

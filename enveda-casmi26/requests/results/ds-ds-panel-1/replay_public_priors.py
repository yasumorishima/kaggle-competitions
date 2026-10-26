"""Replay annotation-only priors from the public panel dataset, without a GPU.

python replay_public_priors.py --panel /kaggle/input/casmi26-natural-product-panel-benchmark --out ./prior_replay
No weights are fitted. This is exact-formula documentation ranking, not full
pipeline inference. Missing answers score zero; exact ties are averaged.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,pathlib
import numpy as np
import pandas as pd

RULES={
 'documentation':lambda g:g['pop'].to_numpy(np.float64),
 'documentation + spectral':lambda g:g['pop'].to_numpy(np.float64)+.5*g.spectral_source.to_numpy(np.float64),
 'documentation + spectral + NP':lambda g:g['pop'].to_numpy(np.float64)+.5*g.spectral_source.to_numpy(np.float64)+g.np_member.to_numpy(np.float64),
 'v10 composite prior':lambda g:g.prior_v10.to_numpy(np.float64),
 'LOTUS references only':lambda g:np.log1p(g.lotus_refs.to_numpy(np.float64)),
}
def expected_rr(score,truth):
 if not truth.any():return 0.
 assert truth.sum()==1
 target=score[truth][0];above=int((score>target).sum());ties=int((score==target).sum())
 stop=min(above+ties,25)
 return float((1/np.arange(above+1,stop+1,dtype=np.float64)).sum()/ties) if stop>above else 0.
def replay(panel,out):
 out.mkdir(parents=True,exist_ok=True)
 base=pd.read_parquet(panel/'pubchem_candidates.parquet');signals=pd.read_parquet(panel/'pubchem_signals.parquet')
 assert len(base)==len(signals) and np.array_equal(base.tier_row,signals.tier_row) and np.array_equal(base.molecule.astype(str),signals.molecule.astype(str))
 ids=pd.read_parquet(panel/'panel.parquet').molecule.astype(str).tolist();assert len(ids)==250 and len(set(ids))==250
 g=signals.copy();g['identity']=base.scorer_key.where(base.scorer_key.ne(''),base.connectivity);g['is_truth']=base.is_truth
 g=g[g.same_formula].copy();assert g.identity.ne('').all()
 rows=[]
 for mol,group in g.groupby('molecule',observed=True,sort=True):
  truth=group.groupby('identity',sort=True).is_truth.max();index=truth.index
  for rule,fn in RULES.items():
   values=pd.Series(fn(group),index=group.identity.to_numpy()).groupby(level=0,sort=True).max().reindex(index).to_numpy()
   rows.append({'molecule':str(mol),'rule':rule,'mrr25':expected_rr(values,truth.to_numpy(bool)),'truth_present':bool(truth.any()),'identities':len(index)})
 details=pd.DataFrame(rows)
 complete=pd.MultiIndex.from_product([ids,list(RULES)],names=['molecule','rule']).to_frame(index=False).merge(details,on=['molecule','rule'],how='left',validate='one_to_one')
 complete['mrr25']=complete.mrr25.fillna(0.);complete['truth_present']=complete.truth_present.fillna(False);complete['identities']=complete.identities.fillna(0).astype(int)
 summary=complete.groupby('rule',sort=False).agg(queries=('molecule','size'),truth_present=('truth_present','sum'),mrr25=('mrr25','mean')).reset_index()
 baseline=float(summary.loc[summary.rule=='documentation','mrr25'].iloc[0]);summary['delta_documentation']=summary.mrr25-baseline
 complete.to_parquet(out/'prior_rule_per_query.parquet',index=False);summary.to_csv(out/'prior_rule_summary.csv',index=False)
 report={'protocol':'all public PubChem candidates with exact truth formula; maximum score per scorer_key, fallback raw connectivity; analytical exact ties; no pool exclusion, spectral score or shortlist',
  'denominator':250,'missing_truth_zero':True,'weights':{'spectral_source':.5,'np_member':1.,'log1p_lotus_refs':.5},
  'pop_lam_in_notebook':.15,'pop_lam_in_this_evaluation':'not applied: a positive scalar does not change prior-only ranks',
  'prior_v10_precision':'float32 operations as in the notebook; float16 documentation inputs; float64 log1p counts rounded to float32 before weighting',
  'no_training':True,'hidden_test_claim':False,'candidate_rows':len(base),'same_formula_rows':len(g),'summary':summary.to_dict('records')}
 (out/'prior_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();replay(a.panel,a.out)

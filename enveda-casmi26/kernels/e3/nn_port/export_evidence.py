"""Executed in the frozen pipeline process AFTER every original final rule."""
import math,json,shutil
from pathlib import Path
import pandas as pd

def export(scope, root):
    root=Path(root);frame=pd.read_csv(root/'submission.csv',dtype=str,keep_default_na=False)
    shutil.copy2(root/'submission.csv',root/'submission_baseline0465.csv')
    processed=set(scope['te'].molecule_id.astype(str));identity=scope['metric_identity']
    def rank_map(smis):
        out={}
        for rank,s in enumerate(smis,1):
            key=identity(s)
            if key is not None:out.setdefault(key,rank)
        return out
    def finite(v):
        try:return float(v) if v is not None and math.isfinite(float(v)) else None
        except (ValueError,TypeError):return None
    rows=[]
    for mid,answer in frame.itertuples(index=False):
        candidates=answer.split(';')
        assert 1<=len(candidates)<=25 and len(candidates)==len(set(candidates)),mid
        active=mid in processed
        base=scope['BASE'].get(mid,([],[],0.))
        maps={
            'main':rank_map(base[0]),
            'prefusion':rank_map(scope.get('PREFUSION_LISTS',{}).get(mid,[])),
            'engine2':rank_map(scope.get('ENG',{}).get(mid,{}).get('smiles',[])),
            'pubchem':rank_map(scope.get('PC',{}).get(mid,{}).get('pc',[]))}
        ice=scope.get('ICE_SCORES',{}).get(mid,{}) or {}
        gl=scope['gl_of'](mid) or {}
        for rank,s in enumerate(candidates,1):
            ident=identity(s);key=scope['chem'].score_key(s)
            row=dict(molecule_id=mid,smiles=s,metric_key=ident[0] if ident else None,baseline_rank=rank,
                processed=active,library_max=finite(base[2]) or 0.,proxy_gl=mid in scope.get('N1_GL_MIDS',set()),
                ice=finite(ice.get(s,ice.get(key))),gl=finite(gl.get(s,gl.get(key))))
            row.update({name+'_rank':mapping.get(ident,0) for name,mapping in maps.items()});rows.append(row)
    pd.DataFrame(rows).to_parquet(root/'frozen_candidate_evidence.parquet',index=False)
    # Save the effective query spectra used by the pipeline (12 rows' molecules in visible smoke, all in rerun).
    scope['te'].to_parquet(root/'effective_query_spectra.parquet',index=False)
    (root/'frozen_run_info.json').write_text(json.dumps(dict(processed_molecules=len(processed),
        output_molecules=len(frame),is_full_run=bool(scope['IS_RERUN']),
        note='Baseline file saved after the original Engine2 rank-2 insertion; candidate-specific PubChem promotion fix included.'),indent=2))
    print('EXPORTED FROZEN FINAL25',len(rows),'candidate rows',flush=True)

import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
from ranking import rrf_order

def apply(evidence,baseline,output,weight=.25,offset=3.):
    f=pd.read_parquet(evidence);base=pd.read_csv(baseline,dtype=str,keep_default_na=False)
    groups={mid:g for mid,g in f.groupby('molecule_id',sort=False)};rows=[];audit=[]
    assert base.molecule_id.is_unique and set(groups)==set(base.molecule_id)
    for mid,text in base.itertuples(index=False):
        g=groups[mid].sort_values('baseline_rank',kind='stable');original=text.split(';')
        assert g.smiles.tolist()==original,mid
        active=bool(g.processed.iloc[0]) and bool(g.neural_ok.all())
        order=rrf_order(g.neural_score.values,weight,offset) if active else np.arange(len(g))
        new=g.smiles.iloc[order].tolist()
        assert len(new)==len(original) and set(new)==set(original),mid
        assert len(new)<=25 and len(set(new))==len(new)
        rows.append((mid,';'.join(new)))
        audit.append(dict(molecule_id=mid,processed=active,changed=new!=original,
            changed_top1=new[0]!=original[0],candidate_count=len(new),weight=weight))
    result=pd.DataFrame(rows,columns=['molecule_id','smiles'])
    assert result.molecule_id.tolist()==base.molecule_id.tolist()
    result.to_csv(output,index=False)
    out=Path(output).parent;pd.DataFrame(audit).to_csv(out/'neural_rerank_audit.csv',index=False)
    report=dict(method='baseline rank + neural rank reciprocal-rank fusion AFTER every baseline rule',
        neural_weight=weight,rrf_offset=offset,weight_is_validated=False, validation_scope="Fixed-weight submission experiment; no validation performed",
        candidate_set_preserved=True,changed_rows=sum(x['changed'] for x in audit),
        changed_top1=sum(x['changed_top1'] for x in audit),rows=len(rows),
        frozen_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
        experimental_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())
    (out/'neural_submission_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report

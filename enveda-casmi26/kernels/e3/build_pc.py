# PubChem-only protection (ours, 10-18): run by exec() inside build.py (sees cells, find_cell, code_cell, pcfix, pcvote).
# Finding (validation e3val4, np-examples, class 3 = truth only in the PubChem tier): the gated merge puts the truth at
# MRR 0.80 / top1 0.74, but the two-engine fusion drops it to 0.56 / 0.35 (48 top-1 lost, 0 gained) and the later
# overlays to 0.46 / 0.22: a PubChem-only candidate has no engine-2 vote, so pool candidates voted by both engines pass it.
#   pcvote=1: in the fusion, a PubChem-only candidate gets the engine-2 vote it would have at its main-list rank.
#   pcfix=1 : after the last base cell, every PubChem-only candidate of the merged list returns to its merged-list rank
#             when a later stage moved it down (or out); nothing else is reordered.

hook_pc_snap = """# ==== ours (yasunorim; pcvote/pcfix): PubChem-only candidates of the gated merge (the submission.csv of this point) ====
import pandas as pd
_PC_S1 = {str(r.molecule_id): [x for x in str(r.smiles).split(';') if x and x != 'CCO']
          for r in pd.read_csv('submission.csv', dtype=str, keep_default_na=False).itertuples(index=False)}
PC_ONLY = {}
for _m, _l in _PC_S1.items():
    _b = set(BASE.get(_m, ([], [], 0.0))[0]) if _m in BASE else set()
    PC_ONLY[_m] = {s: r for r, s in enumerate(_l, 1) if s not in _b}
print('PCONLY molecules with PubChem-only candidates in the merged list', sum(1 for v in PC_ONLY.values() if v),
      '| at rank 1', sum(1 for v in PC_ONLY.values() if 1 in v.values()), flush=True)
if 'VAL_C2' in globals():  # validation only: keep the PubChem-only sets so per-molecule rules can be scored offline
    import json as _pj
    _pj.dump(PC_ONLY, open('/kaggle/working/val_pconly.json', 'w'))
"""

hook_pc_fix = """# ==== ours (yasunorim; pcfix): PubChem-only candidates return to their merged-list rank ====
import pandas as pd
from rdkit import Chem as _PCh, RDLogger as _PRL
_PRL.DisableLog('rdApp.*')
_pck = {}
def _pckey(s):
    if s not in _pck:
        try:
            if 'metric_identity' in globals():
                _pck[s] = metric_identity(s) or s
            else:
                _m = _PCh.MolFromSmiles(s)
                _pck[s] = _PCh.MolToInchiKey(_m)[:14] if _m is not None else s
        except Exception:
            _pck[s] = s
    return _pck[s]
try:
    _sub = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _out, _nmol, _ntop = [], 0, 0
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _l = [x for x in str(_s).split(';') if x and x != 'CCO']
        _pco = PC_ONLY.get(str(_m), {})
        _new = list(_l)
        for _c, _r1 in sorted(_pco.items(), key=lambda kv: kv[1]):
            _kc = _pckey(_c)
            _cur = next((i for i, x in enumerate(_new) if _pckey(x) == _kc), None)
            if _cur is not None and _cur + 1 <= _r1:
                continue
            _item = _new.pop(_cur) if _cur is not None else _c
            _new.insert(min(_r1 - 1, len(_new)), _item)
        _new = _new[:25]
        if _new != _l[:25]:
            _nmol += 1
            _ntop += int(bool(_new) and bool(_l) and _new[0] != _l[0])
        _out.append((_m, ';'.join(_new) if _new else _s))
    _fs = pd.DataFrame(_out, columns=['molecule_id', 'smiles'])
    assert len(_fs) == len(_sub) and _fs.molecule_id.is_unique and (_fs.smiles.str.len() > 0).all()
    assert _fs.smiles.map(lambda s: len(s.split(';'))).max() <= 25
    _fs.to_csv('submission.csv', index=False)
    print('PCFIX molecules changed', _nmol, '| top-1 changed', _ntop, flush=True)
except Exception as _e:
    print('PCFIX failed, submission unchanged:', repr(_e), flush=True)
"""

hook_late_snap = """# ==== ours (yasunorim; latecut): keep the list of the final metric pass (before the later overlays) ====
import shutil as _lsh
_lsh.copy('submission.csv', '/tmp/ours_after_metric_pass.csv')
print('LATECUT snapshot after the final metric pass', flush=True)
"""
hook_late_restore = """# ==== ours (yasunorim; latecut): undo the later overlays (identity overlay, E005-E007, E2R2) ====
# validation e3val3/e3val4 (np-examples): they lowered c1 0.947 -> 0.933 and class 3 0.560 -> 0.459
import os as _los, shutil as _lsh
if _los.path.isfile('/tmp/ours_after_metric_pass.csv'):
    _lsh.copy('/tmp/ours_after_metric_pass.csv', 'submission.csv')
    print('LATECUT restored the final-metric-pass list', flush=True)
else:
    print('LATECUT snapshot missing, submission unchanged', flush=True)
"""
if latecut:
    _im = find_cell("METRIC_SOURCE_SHA256 =")
    cells.insert(_im + 1, code_cell(hook_late_snap))
    cells.append(code_cell(hook_late_restore))
if pcfix or pcvote or pcdump or pcguard:
    _ig = find_cell("'''gated PubChem merge")
    cells.insert(_ig + 1, code_cell(hook_pc_snap))
if pcvote:
    _if = find_cell("ALPHA, KRR")
    _s = "".join(cells[_if]["source"])
    _o1 = "    order = sorted(sc, key=lambda k: -sc[k])[:n]\n"
    _n1 = ("    _pco = PC_ONLY.get(globals().get('_PCV_MID'), {})  # ours (pcvote): PubChem-only candidates get engine 2's vote at their main rank\n"
           "    if _pco:\n"
           "        _ek = set(k for k in e_keys if k)\n"
           "        for r, s in enumerate(v_smis, 1):\n"
           "            if s in _pco:\n"
           "                k = chem.score_key(s) or s\n"
           "                if k not in _ek:\n"
           "                    sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r)\n"
           + _o1)
    _o2 = "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n"
    _n2 = "            _PCV_MID = str(mid)\n" + _o2
    assert _s.count(_o1) == 1 and _s.count(_o2) == 1, "fusion lines not found"
    cells[_if]["source"] = _s.replace(_o1, _n1).replace(_o2, _n2)
if pcfix:
    cells.append(code_cell(hook_pc_fix))

hook_guard_snap = """# ==== ours (yasunorim; pcguard): keep the post-fusion list ====
import shutil as _gsh
_gsh.copy('submission.csv', '/tmp/ours_after_fusion.csv')
print('PCGUARD snapshot after the fusion', flush=True)
"""
hook_guard = """# ==== ours (yasunorim; pcguard): per molecule, the post-fusion list when its top-1 is PubChem-only ====
# validation e3val5 (np-examples): the overlays after the fusion lift c1 0.879 -> 0.936 but sink class 3 0.752 -> 0.459;
# a class-3 molecule is one whose post-fusion top-1 came from the PubChem tier only, so only those keep the fused order.
import os as _gos, pandas as pd
from rdkit import Chem as _GCh, RDLogger as _GRL
_GRL.DisableLog('rdApp.*')
def _gkey(s):
    try:
        _m = _GCh.MolFromSmiles(s)
        return _GCh.MolToInchiKey(_m)[:14] if _m is not None else s
    except Exception:
        return s
try:
    assert _gos.path.isfile('/tmp/ours_after_fusion.csv'), 'snapshot missing'
    _fu = {str(a): b for a, b in pd.read_csv('/tmp/ours_after_fusion.csv', dtype=str, keep_default_na=False).itertuples(index=False)}
    _sub = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _out, _n = [], 0
    for _m, _s in zip(_sub.molecule_id, _sub.smiles):
        _f = _fu.get(str(_m), '')
        _ft = [x for x in _f.split(';') if x and x != 'CCO'][:1]
        _pk = set(_gkey(c) for c in PC_ONLY.get(str(_m), {}))
        if _ft and _gkey(_ft[0]) in _pk and _f != _s:
            _out.append((_m, _f)); _n += 1
        else:
            _out.append((_m, _s))
    _gs = pd.DataFrame(_out, columns=['molecule_id', 'smiles'])
    assert len(_gs) == len(_sub) and _gs.molecule_id.is_unique and (_gs.smiles.str.len() > 0).all()
    _gs.to_csv('submission.csv', index=False)
    print('PCGUARD molecules back to the fused list', _n, flush=True)
except Exception as _e:
    print('PCGUARD failed, submission unchanged:', repr(_e), flush=True)
"""
if pcguard:
    _if2 = find_cell("ALPHA, KRR")
    cells.insert(_if2 + 1, code_cell(hook_guard_snap))
    cells.append(code_cell(hook_guard))


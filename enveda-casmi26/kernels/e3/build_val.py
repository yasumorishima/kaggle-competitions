# Validation rig for build.py (val=N): run by exec() inside build.py, so it sees cells, find_cell, code_cell, val, valseed.
# The fake test replaces COMP right before the base's smoke switch; a snapshot of submission.csv is kept after every
# code cell from the base-list cell on; the last cell scores every snapshot against the held-out truth.
# No submission is made from a val build (the request must not say "submit").

VCFG = dict(N=val, SEED=valseed, LIB={"e180": "enveda-180", "np": "enveda-np-examples"}[valsrc])

hook_v = """# ==== ours (yasunorim; build.py val): VALIDATION RUN, not a submission ====
# fake test = N molecules of library OURS_VCFG['LIB'] held out of train.parquet (np-examples: unseen by the public FPNet):
#   c1 half: molecules that also have spectra in other libraries; only their enveda-180 rows leave train (class-1 sim)
#   c2 half: molecules only in enveda-180; all their rows leave train (no reference spectrum; the structure can stay in
#            the precomputed engine-1 pool, so c2 is optimistic against a hidden class-2/3 molecule)
# up to 3 enveda-180 spectra per molecule become the test rows (hidden test: 3.0 per molecule).
OURS_VCFG = __VCFG__
import pyarrow as pa, pyarrow.parquet as pq, numpy as np, pandas as pd
_vsrc = os.path.join(COMP, 'train.parquet')
_vpf = pq.ParquetFile(_vsrc)
_vm = _vpf.read(columns=['ingest_lib', 'inchikey14']).to_pandas()
_VLIB = OURS_VCFG['LIB']
_ve = set(_vm.inchikey14[_vm.ingest_lib == _VLIB])
_vo = set(_vm.inchikey14[_vm.ingest_lib != _VLIB])
del _vm
_vr = np.random.RandomState(OURS_VCFG['SEED'])
_n1 = OURS_VCFG['N'] // 2
if _VLIB == 'enveda-180':
    VAL_C1 = set(_vr.choice(sorted(_ve & _vo), _n1, replace=False))
    VAL_C2 = set(_vr.choice(sorted(_ve - _vo), OURS_VCFG['N'] - _n1, replace=False))
else:
    # np-examples: every molecule also has spectra in other libraries; c1 keeps those, c2 loses every spectrum
    _vp = list(_vr.permutation(sorted(_ve)))[:min(OURS_VCFG['N'], len(_ve))]
    VAL_C1, VAL_C2 = set(_vp[:len(_vp) // 2]), set(_vp[len(_vp) // 2:])
_vall = VAL_C1 | VAL_C2
VC = '/tmp/valcomp'
os.makedirs(VC, exist_ok=True)
_vw, _vtest = None, []
for _b in _vpf.iter_batches(batch_size=200000):
    _t = _b.to_pandas()
    _hit = _t.inchikey14.isin(_vall)
    _vtest.append(_t[_hit & (_t.ingest_lib == _VLIB)])
    _drop = _t.inchikey14.isin(VAL_C2) | (_t.inchikey14.isin(VAL_C1) & (_t.ingest_lib == _VLIB))
    _tb = pa.Table.from_pandas(_t[~_drop], schema=_b.schema, preserve_index=False)
    if _vw is None:
        _vw = pq.ParquetWriter(os.path.join(VC, 'train.parquet'), _b.schema)
    _vw.write_table(_tb)
_vw.close()
_vt = pd.concat(_vtest).sample(frac=1.0, random_state=OURS_VCFG['SEED']).groupby('inchikey14', sort=True).head(3)
_ids = {k: f'v_{i:04d}' for i, k in enumerate(sorted(_vall))}
_vt = _vt[_vt.inchikey14.isin(_ids)].copy()
_vt['molecule_id'] = _vt.inchikey14.map(_ids)
_vt['spectrum_id'] = [f'vs_{i:05d}' for i in range(len(_vt))]
_real = pq.read_schema(os.path.join(COMP, 'test.parquet'))
_vtt = pa.Table.from_pandas(_vt[list(_real.names)].reset_index(drop=True), preserve_index=False)
try:
    _vtt = _vtt.cast(_real)
except Exception as _e:
    print('VAL test schema cast failed (kept as is):', repr(_e), flush=True)
pq.write_table(_vtt, os.path.join(VC, 'test.parquet'))
_tru = _vt.drop_duplicates('molecule_id')[['molecule_id', 'inchikey14', 'normalized_smiles', 'molecular_formula']].copy()
_tru['cls'] = np.where(_tru.inchikey14.isin(VAL_C1), 'c1', 'c2')
_tru.to_csv('/kaggle/working/val_truth.csv', index=False)
pd.DataFrame({'molecule_id': sorted(_tru.molecule_id), 'smiles': ';'.join(['CCO'] * 25)}).to_csv(
    os.path.join(VC, 'sample_submission.csv'), index=False)
COMP = VC
print('VAL fake test', len(_tru), 'molecules', len(_vt), 'spectra | c1', int((_tru.cls == 'c1').sum()),
      'c2', int((_tru.cls == 'c2').sum()), '| train rows kept',
      pq.ParquetFile(os.path.join(VC, 'train.parquet')).metadata.num_rows, f'{time.time()-T0:.0f}s', flush=True)
""".replace("__VCFG__", repr(VCFG))

hook_snap = """# ==== ours (yasunorim; val): stage snapshot of submission.csv ====
import shutil as _vsh, os as _vos
_vp = '/kaggle/working/submission.csv'
if _vos.path.isfile(_vp):
    _VAL_STAGE = globals().get('_VAL_STAGE', 0) + 1
    _vsh.copy(_vp, f'/kaggle/working/val_stage{_VAL_STAGE:02d}_after_cell__CELL__.csv')
"""

hook_score = """# ==== ours (yasunorim; val): score every stage against the held-out truth ====
import glob as _g, json as _j, numpy as np, pandas as pd
from rdkit import Chem
def _vkey(s):
    try:
        if 'metric_identity' in globals():
            k = metric_identity(s)
            if k:
                return str(k)
    except Exception:
        pass
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToInchiKey(m)[:14] if m is not None else None
_tru = pd.read_csv('/kaggle/working/val_truth.csv')
_tk = {m: _vkey(s) for m, s in zip(_tru.molecule_id, _tru.normalized_smiles)}
_cls = dict(zip(_tru.molecule_id, _tru.cls))
_files = sorted(_g.glob('/kaggle/working/val_stage*.csv')) + ['/kaggle/working/submission.csv']
_res, _ranks = {}, {}
for _f in _files:
    try:
        _s = pd.read_csv(_f, dtype=str, keep_default_na=False)
    except Exception:
        continue
    if 'molecule_id' not in _s or 'smiles' not in _s:
        continue
    _r = {}
    for _m, _l in zip(_s.molecule_id, _s.smiles):
        if _m not in _tk:
            continue
        _ks = [_vkey(x) for x in str(_l).split(';')[:25]]
        _r[_m] = next((i + 1 for i, k in enumerate(_ks) if k is not None and k == _tk[_m]), 0)
    _nm = _f.split('/')[-1]
    _ranks[_nm] = _r
    _out = {}
    for _c in ('all', 'c1', 'c2'):
        _rs = [v for m, v in _r.items() if _c == 'all' or _cls[m] == _c]
        if _rs:
            _a = np.array(_rs)
            _out[_c] = dict(n=len(_a), mrr=round(float(np.mean(np.where(_a > 0, 1.0 / np.maximum(_a, 1), 0))), 4),
                            top1=round(float(np.mean(_a == 1)), 4), hit5=round(float(np.mean((_a > 0) & (_a <= 5))), 4),
                            hit25=round(float(np.mean(_a > 0)), 4))
    _res[_nm] = _out
    print('VALSCORE', _nm, _out, flush=True)
_rk = pd.DataFrame(_ranks)
_rk.index.name = 'molecule_id'
_rk = _rk.join(_tru.set_index('molecule_id')[['cls', 'molecular_formula', 'normalized_smiles']])
try:
    _pm = pd.read_parquet('work/pool_meta.parquet')
    print('pool_meta columns', list(_pm.columns)[:20], len(_pm), flush=True)
    _kc = next((c for c in _pm.columns if 'inchikey' in c.lower() or c.lower() in ('ik', 'ik14', 'key')), None)
    if _kc:
        _pk = set(_pm[_kc].astype(str).str[:14])
        _rk['in_pool1'] = [str(k)[:14] in _pk for k in _tru.set_index('molecule_id').loc[_rk.index, 'inchikey14']]
        for _c in ('c1', 'c2'):
            _x = _rk[_rk.cls == _c]
            print('VALPOOL', _c, 'truth in engine-1 pool', int(_x.in_pool1.sum()), 'of', len(_x), flush=True)
except Exception as _e:
    print('pool check failed', repr(_e), flush=True)
_rk.to_csv('/kaggle/working/val_ranks.csv')
_j.dump(_res, open('/kaggle/working/val_scores.json', 'w'), indent=1)
"""

_first = find_cell("'''all molecules -> base lists")
_code = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and i >= _first]
for _i in reversed(_code):
    cells.insert(_i + 1, code_cell(hook_snap.replace("__CELL__", str(_i))))
_iv = find_cell("'''SMOKE switch")
# the fake test must exist before any cell that records the input ids (a451: the E004 preflight, which the
# E004 postflight later asserts against; e3val1-1 stopped there because the preflight saw the real test)
for _k, _c in enumerate(cells[:_iv]):
    if _c["cell_type"] == "code" and "_e004_preflight" in "".join(_c["source"]):
        _iv = _k
        # the preflight asserts the real competition path; the fake test lives in /tmp/valcomp
        _psrc = "".join(_c["source"])
        _pold = "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}\n"
        assert _psrc.count(_pold) == 1, "E004 preflight path assert not found"
        _c["source"] = _psrc.replace(_pold, "    # ours (val): path assert dropped, COMP is the fake test\n")
        break
cells.insert(_iv, code_cell(hook_v))
# a val run never submits: a base acceptance check must not stop the run before the scoring cell
for _c in cells:
    _src = "".join(_c["source"]) if isinstance(_c["source"], list) else _c["source"]
    if _c["cell_type"] == "code" and "E004_REPORT = _e004_postflight(globals())" in _src:
        _src = _src.replace("E004_REPORT = _e004_postflight(globals())",
                            "try:\n    E004_REPORT = _e004_postflight(globals())\nexcept Exception as _ve:\n"
                            "    print('VAL: E004 postflight failed (ignored in a val run):', repr(_ve), flush=True)")
        _c["source"] = _src
cells.append(code_cell(hook_score))

# np-examples run: the engine-1 FP bank fpnet_full1 was trained on ALL train incl. np-examples (its dataset card),
# so fall back to the base's own fallback, FPNet A (fpnet_0, np fold held out in its training: traindata.py)
if valsrc == "np" and not valfull:
    _nfb = 0
    for _c in cells:
        _src = "".join(_c["source"]) if isinstance(_c["source"], list) else _c["source"]
        _old = "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)"
        if _c["cell_type"] == "code" and _old in _src:
            _c["source"] = _src.replace(_old, "_FULL = []  # ours (val np): fpnet_full1 saw np-examples; use FPNet A")
            _nfb += 1
    assert _nfb == 1, f"fpnet_full1 line found {_nfb} times"

# c3 run (valc3=1): the c2 half also leaves the candidate pools, so only the PubChem tier can still propose the truth.
# engine 1 + its PubChem worker read POOL_DIR -> a filtered copy; engine 2 drops the keys while building its own pool.
hook_c3 = """# ==== ours (yasunorim; val c3): remove the c2 truths from the candidate pools ====
import numpy as np, pandas as pd, os
VAL_C3 = set(VAL_C2) if __VALC3__ == 1 else set(sorted(VAL_C2)[::2])  # valc3=2: half of c2 keeps its pool entry (c2), half leaves (c3)
if __VALC3__ == 2:
    _tru['cls'] = np.where(_tru.inchikey14.isin(VAL_C3), 'c3', _tru.cls)
    _tru.to_csv('/kaggle/working/val_truth.csv', index=False)
    print('VALC3 classes', _tru.cls.value_counts().to_dict(), flush=True)
_POOL_DIR_ORIG = POOL_DIR
_c3d = '/tmp/valcomp/pool'
os.makedirs(_c3d, exist_ok=True)
_st = pd.read_parquet(os.path.join(POOL_DIR, 'train_structs.parquet'))
_c3keys = set(_st.key[_st.inchikey14.isin(VAL_C3)].astype(str)) if {'key', 'inchikey14'} <= set(_st.columns) else set()
_pm = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'))
_k = _pm.key.astype(str)
_dropm = (_k.str[:14].isin(VAL_C3) | _k.isin(_c3keys)).values
_keep = np.where(~_dropm)[0]
_pm.iloc[_keep].reset_index(drop=True).to_parquet(os.path.join(_c3d, 'pool_meta.parquet'), index=False)
np.save(os.path.join(_c3d, 'pool_fp.npy'), np.load(os.path.join(POOL_DIR, 'pool_fp.npy'), mmap_mode='r')[_keep])
if os.path.exists(os.path.join(POOL_DIR, 'pool_frag_off.npy')):
    _off = np.load(os.path.join(POOL_DIR, 'pool_frag_off.npy'))
    _fm = np.load(os.path.join(POOL_DIR, 'pool_frag_mass.npy'), mmap_mode='r')
    _len = np.diff(_off)[_keep]
    _noff = np.concatenate([[0], np.cumsum(_len)]).astype(_off.dtype)
    _pos = np.repeat(_off[_keep] - _noff[:-1], _len) + np.arange(_noff[-1])
    np.save(os.path.join(_c3d, 'pool_frag_mass.npy'), np.asarray(_fm[_pos]))
    np.save(os.path.join(_c3d, 'pool_frag_off.npy'), _noff)
for _f in os.listdir(POOL_DIR):
    if not os.path.exists(os.path.join(_c3d, _f)):
        os.symlink(os.path.join(POOL_DIR, _f), os.path.join(_c3d, _f))
POOL_DIR = _c3d
open('/tmp/valcomp/c3_keys.txt', 'w').write('\\n'.join(sorted(VAL_C3 | _c3keys | {_x[:14] for _x in _c3keys})))
_tm = _pm.key.astype(str).str[:14]
print('VALC3 engine-1 pool dropped', int(_dropm.sum()), 'of', len(_pm), '| c2 truths', len(VAL_C3),
      '| key sample', list(_k.iloc[:2]), '| struct keys', len(_c3keys), flush=True)
"""
if valc3:
    assert valsrc == "np", "valc3 needs valsrc=np"
    _iv2 = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "VALIDATION RUN, not a submission" in "".join(c["source"]))
    cells.insert(_iv2 + 1, code_cell(hook_c3.replace("__VALC3__", str(int(valc3)))))
    _old2 = '    good = np.isfinite(mass)\\n'
    _new2 = ('    good = np.isfinite(mass) & ~pd.Series(keys).astype(str).str[:14].isin(set(open("/tmp/valcomp/c3_keys.txt").read().split())'
             ' if os.path.exists("/tmp/valcomp/c3_keys.txt") else set()).values; log(f"VALC3 engine-2 pool kept {int(good.sum()):,} of {len(good):,}")\\n')
    _n2 = 0
    for _c in cells:
        _src = "".join(_c["source"]) if isinstance(_c["source"], list) else _c["source"]
        if _c["cell_type"] == "code" and _old2 in _src:
            _c["source"] = _src.replace(_old2, _new2)
            _n2 += 1
    assert _n2 == 1, f"engine-2 pool line found {_n2} times"
    # engine 1 + fe_v4 keep the original pool (fe_v4 res tables are aligned to it); its candidate windows skip the c3 rows.
    # The filtered POOL_DIR copy above is used only by the PubChem worker (cell before this one).
    hook_c3b = """# ==== ours (yasunorim; val c3): engine 1 keeps the original pool; its mass windows skip the c3 truths ====
POOL_DIR = _POOL_DIR_ORIG
_C3SET = set(open('/tmp/valcomp/c3_keys.txt').read().split())
"""
    hook_c3c = """# ==== ours (yasunorim; val c3): patch Pool.window so the c3 truths never become engine-1 candidates ====
import numpy as np, pandas as pd
from casmi import pool as _c3pool
if not getattr(_c3pool.Pool, '_c3_patched', False):
    _c3_orig_window = _c3pool.Pool.window
    def _c3_window(self, target, ppm, _o=_c3_orig_window):
        r = _o(self, target, ppm)
        m = getattr(self, '_c3mask', None)
        if m is None:
            k = pd.Series(self.key).astype(str)
            m = self._c3mask = (k.str[:14].isin(_C3SET) | k.isin(_C3SET)).values
            print('VALC3 engine-1 window mask', int(m.sum()), 'of', len(m), flush=True)
        return r[~m[r]]
    _c3pool.Pool.window = _c3_window
    _c3pool.Pool._c3_patched = True
"""
    _i16 = find_cell("'''v1 engine code")
    cells.insert(_i16, code_cell(hook_c3b))
    _i17 = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "E = Engine(L, P, tfp" in "".join(c["source"]))
    cells.insert(_i17, code_cell(hook_c3c))

# ---- ours (10-27): fold-restricted held-out molecules + fold-safe engine FP (valfold=k) ----
if valfold >= 0:
    assert valsrc == "e180" and base == "f465", "valfold is written for e180 molecules on the f465 base"
    _hv = next(c for c in cells if c["cell_type"] == "code" and "VALIDATION RUN, not a submission" in "".join(c["source"]))
    _src = "".join(_hv["source"])
    _old = "_vo = set(_vm.inchikey14[_vm.ingest_lib != _VLIB])\n"
    assert _src.count(_old) == 1
    _new = _old + (
        "import glob as _vg\n"
        "_vfp = sorted(_vg.glob('/kaggle/input/**/casmi26-identity-disjoint-validation-folds/structures.parquet', recursive=True), key=len)[0]\n"
        "_vfs = pd.read_parquet(_vfp, columns=['inchikey14', 'fold_identity'])\n"
        f"_vfk = set(_vfs.inchikey14[_vfs.fold_identity == {valfold}])\n"
        "_ve = _ve & _vfk\n"
        f"print('VAL fold {valfold}: enveda-180 molecules in the fold', len(_ve), '| folds file', _vfp, flush=True)\n")
    _hv["source"] = _src.replace(_old, _new)
    _nf = 0
    for _c in cells:
        _src = "".join(_c["source"]) if isinstance(_c["source"], list) else _c["source"]
        _old = "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)"
        if _c["cell_type"] == "code" and _old in _src:
            _c["source"] = _src.replace(_old, f"_FULL = sorted(glob.glob('/kaggle/input/**/fpnet_fold{valfold}.pt', recursive=True), key=len)[:1]  # ours (valfold): never saw fold {valfold}\nassert _FULL, 'fold-safe fpnet_fold{valfold}.pt not found'")
            _nf += 1
    assert _nf == 1, f"fpnet_full1 line found {_nf} times"

# ---- ours (10-27): dump the base ranker's input rows and stop after the base lists (rkdump=1) ----
hook_rk = """# ==== ours (yasunorim; val rkdump): ranker training rows from held-out molecules ====
import numpy as np, pandas as pd, json as _rj
_tk14 = dict(zip(_tru.molecule_id, _tru.inchikey14))
_rows, _X = [], []
for (_m, _keys, _XF, _names, _sc, _lm, _pid) in _RKD:
    _t = _tk14.get(_m)
    _k14 = np.array([str(k)[:14] for k in _keys])
    _lab = (_k14 == _t)
    _keep = np.union1d(np.argsort(-_sc, kind='stable')[:300], np.where(_lab)[0])
    _pos = {n: i for i, n in enumerate(_names)}
    _X.append(np.asarray(_XF[_keep][:, [_pos[f] for f in RK_FEATS]], np.float32))
    for _r, _i in enumerate(_keep):
        _rows.append((_m, str(_keys[_i]), int(_lab[_i]), float(_sc[_i]), float(_lm), int(_pid[_i]), len(_keys)))
_meta = pd.DataFrame(_rows, columns=['molecule_id', 'key', 'label', 'base_score', 'm_lib_max', 'm_pid', 'm_n_cand'])
_meta['m_pop'] = np.where(_meta.m_pid >= 0, POOL_POP[np.maximum(_meta.m_pid.values, 0)], np.nan)
_meta = _meta.join(pd.DataFrame(np.concatenate(_X), columns=RK_FEATS))
_meta = _meta.merge(_tru[['molecule_id', 'cls']], on='molecule_id', how='left')
_meta.to_parquet('/kaggle/working/rk_rows.parquet', index=False)
_g = _meta.groupby('molecule_id').label.max()
print('RKDUMP rows', len(_meta), 'molecules', _meta.molecule_id.nunique(), 'truth among dumped rows', int(_g.sum()),
      '| features', len(RK_FEATS), flush=True)
_rr = []
for _m, _d in _meta.groupby('molecule_id'):
    _o = _d.sort_values('base_score', ascending=False, kind='stable').drop_duplicates('key')
    _w = np.where(_o.label.values == 1)[0]
    _rr.append(1.0 / (_w[0] + 1) if len(_w) and _w[0] < 25 else 0.0)
print('RKDUMP base ranker MRR@25 (no pop re-rank, no simulators)', round(float(np.mean(_rr)), 4), 'n', len(_rr), flush=True)
"""
if rkdump:
    assert val > 0 and base == "f465", "rkdump needs a val run on the f465 base"
    _ib = find_cell("'''all molecules -> base lists")
    _bc = next(i for i, c in enumerate(cells) if i >= _ib and c["cell_type"] == "code"
               and "score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))" in "".join(c["source"]))
    _src = "".join(cells[_bc]["source"])
    _old = "                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))\n"
    assert _src.count(_old) == 1
    _src = _src.replace(_old, _old + "                _RKD.append((mid, list(C.key), np.concatenate([X, F], 1).astype(np.float32), list(FEATURES) + list(names), score, lib_max, np.asarray(C.pid)))  # ours (rkdump)\n")
    _src = "_RKD = []  # ours (rkdump)\n" + _src
    cells[_bc]["source"] = _src
    cells.insert(_bc + 1, code_cell(hook_rk))
    # nothing after the base lists is needed: later code cells become markdown (kept visible, not run)
    for _c in cells[_bc + 2:]:
        if _c["cell_type"] == "code":
            _c["cell_type"] = "markdown"
            _src = "".join(_c["source"]) if isinstance(_c["source"], list) else _c["source"]
            _c["source"] = "(not run in an rkdump build)\n\n```python\n" + _src + "\n```"
            _c.pop("outputs", None); _c.pop("execution_count", None)
if rkdump:   # the ranker rows need only engine 1: engine 2 (own process) and the PubChem channel are skipped
    _nsk = 0
    for _c in cells:
        _src = "".join(_c["source"]) if isinstance(_c["source"], list) else _c["source"]
        if _c["cell_type"] == "code" and (_src.startswith("'''Our engine (two-ranker + BIO + AFIX) in a subprocess")
                                          or _src.startswith("V17_PATCH = ")):
            _c["cell_type"] = "markdown"
            _c["source"] = "(skipped in an rkdump build: not needed for the engine-1 ranker rows)\n"
            _c.pop("outputs", None); _c.pop("execution_count", None)
            _nsk += 1
    assert _nsk == 2, f"rkdump skip cells found {_nsk}"

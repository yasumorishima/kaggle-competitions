# Validation rig for build.py (val=N): run by exec() inside build.py, so it sees cells, find_cell, code_cell, val, valseed.
# The fake test replaces COMP right before the base's smoke switch; a snapshot of submission.csv is kept after every
# code cell from the base-list cell on; the last cell scores every snapshot against the held-out truth.
# No submission is made from a val build (the request must not say "submit").

VCFG = dict(N=val, SEED=valseed)

hook_v = """# ==== ours (yasunorim; build.py val): VALIDATION RUN, not a submission ====
# fake test = N enveda-180 (timsTOF, like the hidden test) molecules held out of train.parquet:
#   c1 half: molecules that also have spectra in other libraries; only their enveda-180 rows leave train (class-1 sim)
#   c2 half: molecules only in enveda-180; all their rows leave train (no reference spectrum; the structure can stay in
#            the precomputed engine-1 pool, so c2 is optimistic against a hidden class-2/3 molecule)
# up to 3 enveda-180 spectra per molecule become the test rows (hidden test: 3.0 per molecule).
OURS_VCFG = __VCFG__
import pyarrow as pa, pyarrow.parquet as pq, numpy as np, pandas as pd
_vsrc = os.path.join(COMP, 'train.parquet')
_vpf = pq.ParquetFile(_vsrc)
_vm = _vpf.read(columns=['ingest_lib', 'inchikey14']).to_pandas()
_ve = set(_vm.inchikey14[_vm.ingest_lib == 'enveda-180'])
_vo = set(_vm.inchikey14[_vm.ingest_lib != 'enveda-180'])
del _vm
_vr = np.random.RandomState(OURS_VCFG['SEED'])
_n1 = OURS_VCFG['N'] // 2
VAL_C1 = set(_vr.choice(sorted(_ve & _vo), _n1, replace=False))
VAL_C2 = set(_vr.choice(sorted(_ve - _vo), OURS_VCFG['N'] - _n1, replace=False))
_vall = VAL_C1 | VAL_C2
VC = '/tmp/valcomp'
os.makedirs(VC, exist_ok=True)
_vw, _vtest = None, []
for _b in _vpf.iter_batches(batch_size=200000):
    _t = _b.to_pandas()
    _hit = _t.inchikey14.isin(_vall)
    _vtest.append(_t[_hit & (_t.ingest_lib == 'enveda-180')])
    _drop = _t.inchikey14.isin(VAL_C2) | (_t.inchikey14.isin(VAL_C1) & (_t.ingest_lib == 'enveda-180'))
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
cells.insert(_iv, code_cell(hook_v))
cells.append(code_cell(hook_score))

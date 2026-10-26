"""best_pool_fz of the PubChem gate for the NP panel, with and without the truth in the pool (v3 A+B logits).
usage: python np_poolfz.py <v3_code> <v3_models> <pool_dir> <train.parquet> <out.parquet> <probe_dir>"""
import os, sys, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
V3CODE, V3MOD, POOL, TRAIN, OUT, PROBE = sys.argv[1:7]
sys.path.insert(0, V3CODE); sys.path.insert(0, PROBE)
import torch
import probe_core2 as pc
from casmi import fpnet, chem
ts = pd.read_parquet(os.path.join(POOL, 'train_structs.parquet'))
np_keys = sorted(set(ts.loc[(ts.libmask.astype('int64') & (1 << 9)) != 0, 'key']))
alias = ts[ts.key.isin(np_keys)]; raw2key = dict(zip(alias.inchikey14, alias.key))
cols = ['ingest_lib', 'inchikey14', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']
pf = pq.ParquetFile(TRAIN); parts = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=cols).to_pandas(); parts.append(t[t.ingest_lib == 'enveda-np-examples'])
q = pd.concat(parts, ignore_index=True); q['key'] = q.inchikey14.map(raw2key)
q['nm'] = chem.neutral_mass(q.precursor_mz.values.astype(np.float64), q.adduct.values)
bank = fpnet.ModelBank(sorted(glob.glob(os.path.join(V3MOD, 'fpnet_*.pt'))), device='cuda' if torch.cuda.is_available() else 'cpu')
bits = np.load(os.path.join(POOL, 'fp_bits.npy')); nbits = len(bits)
pm = pd.read_parquet(os.path.join(POOL, 'pool_meta.parquet'), columns=['mass', 'key'])
pmass = pm.mass.values.astype(np.float64); pkey = pm.key.values
pfp = np.load(os.path.join(POOL, 'pool_fp.npy'), mmap_mode='r')
rows = []
for key, sub in q.groupby('key', sort=False):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    if not len(nms):
        continue
    target = float(np.median(nms)); spectra = []
    for r in sub.itertuples():
        ce = r.collision_energy_ev
        ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                            mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                            ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    z = pc.molecule_logits(bank, spectra[:16])
    if z is None:
        continue
    tol = target * 10e-6
    a = np.searchsorted(pmass, target - tol, 'left'); b = np.searchsorted(pmass, target + tol, 'right')
    if b <= a:
        tol = target * 30e-6; a = np.searchsorted(pmass, target - tol, 'left'); b = np.searchsorted(pmass, target + tol, 'right')
    if b > a:
        F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64); fz = F @ z.astype(np.float64)
        tr = pkey[a:b] == key
        rows.append(dict(key=key, n_pool=int(b - a), best_incl=float(fz.max()), best_excl=float(fz[~tr].max()) if (~tr).any() else float('nan'),
                         truth_fz=float(fz[tr].max()) if tr.any() else float('nan'), truth_pool_rank=int((fz > fz[tr].max()).sum()) + 1 if tr.any() else -1))
    else:
        rows.append(dict(key=key, n_pool=0, best_incl=float('nan'), best_excl=float('nan'), truth_fz=float('nan'), truth_pool_rank=-1))
d = pd.DataFrame(rows); d.to_parquet(OUT, index=False)
print(len(d), 'pool fz rank of truth: top1', (d.truth_pool_rank == 1).mean(), 'le5', ((d.truth_pool_rank > 0) & (d.truth_pool_rank <= 5)).mean(), 'median window', d.n_pool.median())

"""PubChem-channel development harness on the enveda-np-examples panel (conditional NP proxy, not clean CV).

For every NP-example molecule (scorer key) this builds the complete candidate table of its 10-ppm PubChem window:
partial ECFP4 score (pass 1), full FPNet A+B f.z (pass 2) for ALL rows, popularity/patent/NP flags, connectivity key,
and scorer keys for the rows that can matter. Variants are then replayed offline by np_eval.py.

usage: python np_tables.py <v3_code> <v3_models> <pool_dir> <tier_dir> <annot_dir> <train.parquet> <out_dir> <workers> <probe_dir>
"""
import json, os, sys, time
import numpy as np
import pandas as pd

V3CODE, V3MOD, POOL, TIER, ANN, TRAIN, OUT, NW, PROBE = sys.argv[1:10]
NW = int(NW)
sys.path.insert(0, V3CODE); sys.path.insert(0, PROBE)
PPM = 10.0
_W = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def init():
    from casmi import chem
    _W['chem'] = chem
    r = chem._rdkit()
    _W['Chem'] = r['Chem']; _W['gen'] = r['gen'].GetMorganGenerator(radius=2, fpSize=4096)
    _W['mass'] = np.load(os.path.join(TIER, 'pc_mass.npy'), mmap_mode='r')
    _W['off'] = np.load(os.path.join(TIER, 'pc_off.npy'), mmap_mode='r')
    _W['buf'] = np.load(os.path.join(TIER, 'pc_smiles.npy'), mmap_mode='r')
    bits = np.load(os.path.join(POOL, 'fp_bits.npy'))
    _W['bits'] = bits; _W['sel_e'] = np.where(bits < 4096)[0]; _W['raw_e'] = bits[_W['sel_e']]
    for n in ('pc_lsid', 'pc_lpmid', 'pc_lpatent', 'pc_np', 'pc_ik14', 'pc_cid'):
        _W[n] = np.load(os.path.join(ANN, n + '.npy'), mmap_mode='r')


def one(task):
    mid, target, z, truth_keys, truth_ik = task
    t0 = time.time()
    chem, Chem, gen = _W['chem'], _W['Chem'], _W['gen']
    tol = target * PPM * 1e-6
    a = int(np.searchsorted(_W['mass'], target - tol, 'left')); b = int(np.searchsorted(_W['mass'], target + tol, 'right'))
    n = b - a
    if n <= 0:
        np.savez_compressed(os.path.join(OUT, mid + '.npz'), n=0, a=a); return mid, 0, 0, time.time() - t0
    off = np.asarray(_W['off'][a:b + 1]).astype(np.int64); raw = bytes(_W['buf'][off[0]:off[-1]]); base = off[0]
    smis = [raw[off[i] - base:off[i + 1] - base].decode('ascii') for i in range(n)]
    bits, raw_e, z_e = _W['bits'], _W['raw_e'], z[_W['sel_e']].astype(np.float32)
    sc1 = np.full(n, -np.inf, np.float32); fz = np.full(n, -np.inf, np.float32)
    cache = {}
    for i, s in enumerate(smis):
        if s in cache:
            sc1[i], fz[i] = cache[s]; continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            cache[s] = (-np.inf, -np.inf); continue
        p = float(gen.GetFingerprintAsNumPy(m)[raw_e].astype(np.float32) @ z_e)
        fp = chem.raw_fingerprint(s)
        f = float(fp[bits].astype(np.float32) @ z) if fp is not None else -np.inf
        cache[s] = (p, f); sc1[i], fz[i] = p, f
    ik = np.asarray(_W['pc_ik14'][a:b])
    pop = np.asarray(_W['pc_lsid'][a:b], np.float32) + np.asarray(_W['pc_lpmid'][a:b], np.float32)
    # scorer keys where they can matter: strong rows by either score, popular rows, and every row sharing a truth connectivity
    need = set(np.argsort(-fz)[:4000].tolist()) | set(np.argsort(-sc1)[:1500].tolist()) | set(np.argsort(-pop)[:600].tolist())
    need |= set(np.where(np.isin(ik, np.array(sorted(truth_ik), 'S14')))[0].tolist())
    sk = np.zeros(n, 'S14'); kc = {}
    for i in need:
        s = smis[i]
        if s not in kc:
            try:
                kc[s] = (chem.score_key(s) or '').encode()
            except Exception:
                kc[s] = b''
        sk[i] = kc[s]
    is_truth = np.isin(sk, np.array(sorted(truth_keys), 'S14'))
    np.savez_compressed(os.path.join(OUT, mid + '.npz'), n=n, a=a, sc1=sc1, fz=fz, ik=ik, sk=sk, is_truth=is_truth,
                        lsid=np.asarray(_W['pc_lsid'][a:b]), lpmid=np.asarray(_W['pc_lpmid'][a:b]),
                        lpat=np.asarray(_W['pc_lpatent'][a:b]), npf=np.asarray(_W['pc_np'][a:b]), cid=np.asarray(_W['pc_cid'][a:b]))
    return mid, n, int(is_truth.sum()), time.time() - t0


def main():
    os.makedirs(OUT, exist_ok=True)
    import torch, glob
    import pyarrow.parquet as pq
    import probe_core2 as pc
    from casmi import fpnet, chem
    ts = pd.read_parquet(os.path.join(POOL, 'train_structs.parquet'))
    np_rows = (ts.libmask.astype('int64') & (1 << 9)) != 0
    np_keys = sorted(set(ts.loc[np_rows, 'key']))
    alias = ts[ts.key.isin(np_keys)]
    log('NP keys', len(np_keys), 'alias rows', len(alias), 'single-sid keys', int((alias.groupby('key').sid.nunique() == 1).sum()))
    raw2key = dict(zip(alias.inchikey14, alias.key))
    cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs',
            'ms2_normalized_intensities', 'collision_energy_ev']
    pf = pq.ParquetFile(TRAIN); parts = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=cols).to_pandas()
        parts.append(t[t.ingest_lib == 'enveda-np-examples'])
    q = pd.concat(parts, ignore_index=True)
    q['key'] = q.inchikey14.map(raw2key)
    assert q.key.notna().all()
    q['nm'] = chem.neutral_mass(q.precursor_mz.values.astype(np.float64), q.adduct.values)
    log('NP query spectra', len(q), 'molecules', q.key.nunique(), 'adducts', q.adduct.value_counts().head(6).to_dict())
    from rdkit import Chem
    std_of = {}
    for k, s in zip(alias.key, alias.smiles):
        m = Chem.MolFromSmiles(s)
        std_of.setdefault(k, set()).add(Chem.MolToInchiKey(m)[:14] if m is not None else '')
    models = sorted(glob.glob(os.path.join(V3MOD, 'fpnet_*.pt')))
    bank = fpnet.ModelBank(models, device='cuda' if torch.cuda.is_available() else 'cpu')
    tasks, meta = [], []
    for key, sub in q.groupby('key', sort=False):
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            meta.append(dict(key=key, status='no_mass')); continue
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                                ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        z = pc.molecule_logits(bank, spectra[:16])
        if z is None:
            meta.append(dict(key=key, status='no_logits')); continue
        true_mass = float(ts.loc[ts.key == key, 'mass'].iloc[0])
        tk = {key.encode()}
        tasks.append((key, target, z, tk, {x.encode() for x in std_of.get(key, set()) | {key}}))
        meta.append(dict(key=key, status='ok', target=target, true_mass=true_mass, ppm=(target - true_mass) / true_mass * 1e6,
                         n_spectra=int(len(sub)), n_sid=int(alias[alias.key == key].sid.nunique()),
                         adducts=';'.join(sorted(set(sub.adduct))), modes=';'.join(sorted(set(sub.ionization_mode)))))
    del bank
    log('tasks', len(tasks))
    import multiprocessing as mp
    res = {}
    with mp.get_context('fork').Pool(NW, initializer=init) as P:
        for k, (mid, n, nt, secs) in enumerate(P.imap_unordered(one, tasks, chunksize=1)):
            res[mid] = (n, nt, secs)
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                log('done', k + 1, len(tasks))
    for m in meta:
        if m['key'] in res:
            m['n_window'], m['n_truth_rows'], m['secs'] = res[m['key']]
    pd.DataFrame(meta).to_parquet(os.path.join(OUT, 'panel.parquet'), index=False)
    d = pd.DataFrame(meta)
    log(json.dumps(dict(molecules=len(d), ok=int((d.status == 'ok').sum()), truth_in_window=int((d.n_truth_rows > 0).sum()),
                        median_window=float(d.n_window.median()), max_window=int(d.n_window.max()), total_rows=int(d.n_window.sum()))))
    log('ALL_DONE')


if __name__ == '__main__':
    main()

"""ours (yasunorim): class-3 candidate generation from matched molecular pairs (MMP) mined from the train structures.

The base engine's generator (casmi/derive.py) applies about 48 hand-written transforms. Here the transforms are mined
from data: every train structure is cut at each acyclic single bond (rdMMPA, one cut); two structures that share the
larger side (the context) define a replacement x -> y of the smaller side (<= MAXV heavy atoms). A replacement seen
in >= MIN_COUNT contexts is a rule. For a molecule, the parents are the engine's strongest analogs; each rule whose
mass change matches target - mass(parent) is applied at every cut of the parent.

Measured offline (400 enveda-180 structures, their 10 nearest train structures as parents, rules mined without the
400): the true structure is generated for 65.7 % of them (11.0 % with casmi/derive.py), 29 products per molecule.

install(E, ...) wraps E.generate: the base products first, then the MMP products (best first by parent similarity x
log(1 + rule count)) up to MAX_NEW. Product records are built exactly like Engine.generate builds them.
"""
import time
from collections import defaultdict

import numpy as np

MAXV = 13
RULES = {}          # x -> list of (y, count, delta mass)
STATS = dict(molecules=0, parents=0, products=0, secs=0.0)
GEN_KEYS = {}       # target mass (rounded) -> set of keys produced by MMP for the last call (validation bookkeeping)


def _frag(s):
    from rdkit import Chem
    from rdkit.Chem import rdMMPA
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None or m.GetNumHeavyAtoms() > 80:
        return []
    try:
        fr = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
    except Exception:
        return []
    out = []
    for _core, ch in fr:
        p = ch.split('.')
        if len(p) != 2:
            continue
        a, b = p
        ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if ma is None or mb is None:
            continue
        na, nb = ma.GetNumHeavyAtoms() - 1, mb.GetNumHeavyAtoms() - 1
        if nb <= MAXV and na >= nb:
            out.append((a, b))
        if na <= MAXV and nb >= na:
            out.append((b, a))
    return out


def _init():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def _vmass(s):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    m = Chem.MolFromSmiles(s.replace('[*:1]', '[H]'))
    return Descriptors.ExactMolWt(m) if m is not None else None


def mine(train_parquet, workers=4, min_count=2, max_ctx=60, log=print):
    """Rules from the unique structures of train_parquet (normalized_smiles, one per inchikey14)."""
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    t0 = time.time()
    t = pq.read_table(train_parquet, columns=['normalized_smiles', 'inchikey14']).to_pandas()
    smis = t.drop_duplicates('inchikey14').normalized_smiles.dropna().tolist()
    del t
    with Pool(workers, initializer=_init) as p:
        R = p.map(_frag, smis, chunksize=500)
    ctx = defaultdict(set)
    for r in R:
        for c, v in r:
            ctx[c].add(v)
    del R
    cnt = defaultdict(int)
    for vs in ctx.values():
        if len(vs) < 2 or len(vs) > max_ctx:
            continue
        vs = list(vs)
        for x in vs:
            for y in vs:
                if x != y:
                    cnt[(x, y)] += 1
    del ctx
    vm = {}
    RULES.clear()
    for (x, y), c in cnt.items():
        if c < min_count:
            continue
        for z in (x, y):
            if z not in vm:
                vm[z] = _vmass(z)
        if vm[x] is None or vm[y] is None:
            continue
        RULES.setdefault(x, []).append((y, c, vm[y] - vm[x]))
    log(f'MMP rules: {sum(len(v) for v in RULES.values())} from {len(smis)} structures ({time.time()-t0:.0f}s)')
    return RULES


def products(parent_smiles, delta, tol):
    """{product SMILES: rule count} for the rules of parent_smiles whose mass change is delta +- tol."""
    from rdkit import Chem
    out = {}
    for c, x in _frag(parent_smiles):
        for y, cnt, dm in RULES.get(x, ()):
            if abs(dm - delta) > tol:
                continue
            try:
                m = Chem.molzip(Chem.MolFromSmiles(c), Chem.MolFromSmiles(y))
                Chem.SanitizeMol(m)
                s = Chem.MolToSmiles(m)
            except Exception:
                continue
            if cnt > out.get(s, 0):
                out[s] = cnt
    return out


CUR = {}           # spectra of the molecule being run (set by the E.run wrapper)
_TFP = {}


def _pred_logits(E, spectra):
    """FP logits of the molecule exactly as Engine.run computes z_avg (single + merged spectra)."""
    from casmi import chem, fpnet
    from casmi.spectra import merge_spectra
    spectra = spectra[:E.cfg.max_query_spectra]
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec'])
        ce_ok = s['ce'] is not None and not np.isnan(s['ce'])
        items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']),
                            ce=float(s['ce']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(int(s.get('ce_n', 1)), 8), mode=float(s['mode'])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
        prec = float(np.median([s['prec'] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
        adducts = [s['adduct'] for s in grp]
        ad = max(set(adducts), key=adducts.count)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    zs, _ = E.bank.logits_el(items_s)
    zm, _ = E.bank.logits_el(items_m)
    return 0.5 * (np.asarray(zs).mean(0) + np.asarray(zm).mean(0))


def fp_parents(E, spectra, target, k, max_shift=250.0, exclude=()):
    """ours: k train structures whose fingerprint best fits the molecule's predicted FP (score f.z / sqrt(bits)),
    within max_shift Da of the target. They become extra MMP parents (the spectral analog channel misses many)."""
    L, P = E.L, E.pool
    if 'fp' not in _TFP:
        _TFP['fp'] = np.unpackbits(E.train_fp, axis=1)[:, :P.nbits]
        _TFP['nb'] = np.sqrt(_TFP['fp'].sum(1).astype(np.float32) + 1.0)
    T, nb = _TFP['fp'], _TFP['nb']
    z = _pred_logits(E, spectra).astype(np.float32)
    m = np.asarray(L.struct_mass, np.float64)
    idx = np.where(np.abs(m - target) <= max_shift)[0]
    if len(idx) == 0:
        return []
    sc = np.empty(len(idx), np.float32)
    for a in range(0, len(idx), 20000):
        j = idx[a:a + 20000]
        sc[a:a + 20000] = (T[j].astype(np.float32) @ z) / nb[j]
    ex = set(exclude)
    out = []
    for t in np.argsort(-sc):
        sid = int(idx[t])
        if sid in ex:
            continue
        out.append(sid)
        if len(out) >= k:
            break
    return out


def install(E, max_new=60, n_parent=10, min_sim=0.3, keep_base=True, n_fp_parent=0, fp_parent_sim=0.3):
    """Wrap E.generate (call BEFORE pc_join.install, which wraps E.generate again). n_fp_parent > 0 also adds that
    many FP-retrieved parents (fp_parents) behind the spectral analogs, with similarity fp_parent_sim."""
    from casmi import chem, frag as fragmod
    gen0 = E.generate
    run0 = E.run

    def run(spectra, target, *a, **kw):
        CUR['spectra'] = spectra
        return run0(spectra, target, *a, **kw)
    E.run = run

    def generate(analogs, target, exclude_keys):
        t0 = time.time()
        base = gen0(analogs, target, exclude_keys) if keep_base else []
        seen = set(exclude_keys) | set(g['key'] for g in base)
        cfg, L, P = E.cfg, E.L, E.pool
        tol = max(cfg.gen_tol_da, target * cfg.ppm_win * 1e-6)
        parents = [a for a in analogs if a[1] >= min_sim][:n_parent]
        if n_fp_parent > 0 and E.bank is not None and CUR.get('spectra'):
            try:
                fps = fp_parents(E, CUR['spectra'], target, n_fp_parent, exclude=[a[0] for a in parents])
                parents = parents + [(sid, fp_parent_sim, 0.0, False) for sid in fps]
                STATS['fp_parents'] = STATS.get('fp_parents', 0) + len(fps)
            except Exception as e:
                STATS['fp_parent_errors'] = STATS.get('fp_parent_errors', 0) + 1
                STATS['fp_parent_error'] = repr(e)[:200]
        cand = {}
        for sid, sim, _shift, _np in parents:
            psmi = L.struct_smiles[sid]
            try:
                pr = products(psmi, target - float(L.struct_mass[sid]), tol)
            except Exception:
                pr = {}
            for s, cnt in pr.items():
                sc = float(sim) * np.log1p(cnt)
                if sc > cand.get(s, (0.0,))[0]:
                    cand[s] = (sc, sid, sim)
        new = []
        for s, (sc, sid, sim) in sorted(cand.items(), key=lambda kv: -kv[1][0]):
            std = chem.standardize_smiles(s)
            if std is None:
                continue
            k = chem.score_key(std)
            if k is None or k in seen:
                continue
            p = chem.mol_props(std)
            if p is None or abs(p[1] - target) > tol:
                continue
            fp = chem.raw_fingerprint(std)
            if fp is None:
                continue
            fp = fp[P.bits].astype(np.float32)
            pfp = np.unpackbits(E.train_fp[sid])[:P.nbits].astype(np.float32)
            inter = float((fp * pfp).sum()); tan = inter / (fp.sum() + pfp.sum() - inter + 1e-9)
            seen.add(k)
            new.append(dict(smiles=std, key=k, formula=p[0], mass=p[1], n_heavy=p[2], fp=fp,
                            frags=fragmod.fragments_for_smiles(std), sim=sim, steps=1, tan=tan, parent=sid))
            if len(new) >= max_new:
                break
        GEN_KEYS.clear(); GEN_KEYS['last'] = set(g['key'] for g in new)
        STATS['molecules'] += 1; STATS['parents'] += len(parents); STATS['products'] += len(new)
        STATS['secs'] += time.time() - t0
        return base + new

    E.generate = generate

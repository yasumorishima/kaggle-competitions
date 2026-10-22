"""ours_bt (yasunorim, 10-22): biotransformation neighbours of known structures for engine 2's candidate pool.

Engine 2 ranks the pool structures inside the test molecule's neutral-mass window. Natural products come in series
(the same core with one sugar more or less, one methyl more or less, a sulfate, an extra hydroxyl). A test molecule
whose structure is missing from every structure table is often one such step away from a table structure. For each
test target mass T and each step with mass change d, the pool structures at T - d get the step applied (RDKit
reactions, one site at a time); products that land within 10 ppm of T and are not in the pool yet become candidates.
Only engine 2 sees them (the public v14 lesson: an extra structure source helps in engine 2 alone, it hurt in engine 1).

Steps (in priority order, per target at most CAP products):
  sugar off (O-glycoside of a pyranose / furanose split, the aglycone keeps the OH), + hexose, - CH2 (aryl methyl
  ether -> phenol), + CH2 (phenol -> aryl methyl ether), + deoxyhexose, + glucuronic acid, + pentose, + SO3 on a
  phenol, + O on an aromatic CH, - O (phenol -> CH), + H2 on a non-aromatic C=C.
"""
import time
import numpy as np
import pandas as pd

PPM = 10.0
CAP = 40
BUDGET = 1200.0      # seconds for the whole step; past it the products found so far are kept
_SUG = {"glc": "C1OC(CO)C(O)C(O)C1O", "rha": "C1OC(C)C(O)C(O)C1O", "gla": "C1OC(C(=O)O)C(O)C(O)C1O", "pen": "C1OCC(O)C(O)C1O"}
# name, mass change, reaction SMARTS (None = the sugar split, done in code)
STEPS = [
    ("deglyc", None, None),
    ("+glc", 162.0528234, "[c:1][OX2H1:2]>>[c:1][O:2]" + _SUG["glc"]),
    ("-ch2", -14.0156501, "[c:1][OX2:2][CH3]>>[c:1][O:2]"),
    ("+ch2", 14.0156501, "[c:1][OX2H1:2]>>[c:1][O:2]C"),
    ("+rha", 146.0579088, "[c:1][OX2H1:2]>>[c:1][O:2]" + _SUG["rha"]),
    ("+gla", 176.0320880, "[c:1][OX2H1:2]>>[c:1][O:2]" + _SUG["gla"]),
    ("+pen", 132.0422587, "[c:1][OX2H1:2]>>[c:1][O:2]" + _SUG["pen"]),
    ("+so3", 79.9568148, "[c:1][OX2H1:2]>>[c:1][O:2]S(=O)(=O)O"),
    ("+o", 15.9949146, "[cH1:1]>>[c:1]O"),
    ("-o", -15.9949146, "[c:1][OX2H1]>>[c:1]"),
    ("+h2", 2.0156501, "[C;!a:1]=[C;!a:2]>>[C:1][C:2]"),
]
SUGAR_LOSS = [162.0528234, 146.0579088, 176.0320880, 132.0422587]   # dehydrated hexose, deoxyhexose, glucuronic acid, pentose
_R = {}


def _rx(sma):
    from rdkit.Chem import AllChem
    if sma not in _R:
        _R[sma] = AllChem.ReactionFromSmarts(sma)
    return _R[sma]


def exact_mass(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.CalcExactMolWt(m) if m is not None else np.nan
    except Exception:
        return np.nan


def _deglyc(m):
    """Aglycones of every O-glycosidic bond whose sugar side is a small ring with one O (pyranose / furanose)."""
    from rdkit import Chem
    out = []
    ri = m.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        at = [m.GetAtomWithIdx(i) for i in ring]
        if sum(a.GetSymbol() == "O" for a in at) != 1 or any(a.GetIsAromatic() for a in at):
            continue
        if any(a.GetSymbol() not in ("C", "O") for a in at):
            continue
        ro = next(a for a in at if a.GetSymbol() == "O")
        for c in ro.GetNeighbors():
            if c.GetIdx() not in ring:
                continue
            for nb in c.GetNeighbors():
                if nb.GetSymbol() != "O" or nb.GetIdx() in ring or nb.IsInRing() or nb.GetDegree() != 2:
                    continue
                rw = Chem.RWMol(m)
                rw.RemoveBond(c.GetIdx(), nb.GetIdx())
                frags = Chem.GetMolFrags(rw, asMols=False, sanitizeFrags=False)
                fa = next(f for f in frags if nb.GetIdx() in f)
                fs = next(f for f in frags if c.GetIdx() in f)
                if len(fs) > 14 or len(fa) < 6:      # the sugar side stays a single sugar; the aglycone is not tiny
                    continue
                for i in sorted(fs, reverse=True):
                    rw.RemoveAtom(i)
                try:
                    p = rw.GetMol()
                    for a in p.GetAtoms():
                        a.SetNoImplicit(False)
                    Chem.SanitizeMol(p)
                    out.append(Chem.MolToSmiles(p))
                except Exception:
                    pass
    return out


def _apply(step, smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    if step[0] == "deglyc":
        return _deglyc(m)
    out = []
    for ps in _rx(step[2]).RunReactants((m,), maxProducts=24):
        p = ps[0]
        try:
            Chem.SanitizeMol(p)
            out.append(Chem.MolToSmiles(p))
        except Exception:
            pass
    return out


def _target_job(args):
    """All products for one target mass, in step priority order (raw SMILES, mass-checked, at most 4 * CAP)."""
    T, jobs, t_end = args
    seen, out = set(), []
    for si, smi in jobs:
        if time.time() > t_end or len(out) >= 4 * CAP:
            break
        for p in _apply(STEPS[si], smi):
            if p in seen:
                continue
            seen.add(p)
            pm = exact_mass(p)
            if np.isfinite(pm) and abs(pm - T) <= T * PPM / 1e6:
                out.append(p)
    return out


def extend(tr, co_keys, co_smi, co_mass, test_path, nm_fn, canon_fn, log, workers=4):
    """New rows (ik, smi) for engine 2's pool: biotransformation neighbours of the pool / training structures."""
    from multiprocessing import Pool as MPool
    t0 = time.time()
    te = pd.read_parquet(test_path, columns=["molecule_id", "precursor_mz", "adduct"])
    nm = nm_fn(te.precursor_mz.values.astype(np.float64), te.adduct.astype(str).values)
    te = te.assign(nm=nm)[np.isfinite(nm)]
    T = te.groupby("molecule_id").nm.median().values
    with MPool(workers) as mp:
        tr_mass = np.array(mp.map(exact_mass, list(tr.smi), chunksize=2000))
    smi = np.concatenate([np.asarray(co_smi, dtype=object), tr.smi.values.astype(object)])
    ms = np.concatenate([np.asarray(co_mass, np.float64), tr_mass])
    ok = np.isfinite(ms)
    smi, ms = smi[ok], ms[ok]
    o = np.argsort(ms, kind="mergesort")
    smi, ms = smi[o], ms[o]

    def win(c, T_):
        a = np.searchsorted(ms, c - T_ * PPM / 1e6, "left")
        b = np.searchsorted(ms, c + T_ * PPM / 1e6, "right")
        return range(a, b)

    t_end = t0 + BUDGET
    tasks = []
    for t in T:
        jobs = []
        for si, st in enumerate(STEPS):
            if st[0] == "deglyc":
                for sl in SUGAR_LOSS:
                    jobs += [(si, smi[i]) for i in win(t + sl, t)]
            else:
                jobs += [(si, smi[i]) for i in win(t - st[1], t)]
        tasks.append((float(t), jobs[:600], t_end))
    log(f"ours_bt: {len(T)} targets, seeds per target median {int(np.median([len(j) for _, j, _ in tasks])) if tasks else 0}, "
        f"pool+train masses {len(ms):,} ({time.time() - t0:.0f}s)")
    with MPool(workers) as mp:
        prods = mp.map(_target_job, tasks, chunksize=1)
    flat = sorted({p for ps in prods for p in ps})
    with MPool(workers) as mp:
        keys = dict(zip(flat, mp.map(canon_fn, flat, chunksize=200)))
    have = set(co_keys) | set(tr.ik)
    rows, nt = [], 0
    for ps in prods:
        k_ = 0
        for p in ps:
            k = keys.get(p)
            if k is None or k in have:
                continue
            have.add(k); rows.append((k, p)); k_ += 1
            if k_ >= CAP:
                break
        nt += k_ > 0
    log(f"ours_bt: {len(rows):,} new structures for {nt} of {len(T)} targets ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows, columns=["ik", "smi"])

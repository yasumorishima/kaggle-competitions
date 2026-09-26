"""Fragment explanation (MetFrag-lite) as a ranking channel, measured per class.

For each candidate, every fragment that one or two bond cuts can make (a ring
opens with two cuts in the same ring) gives a neutral mass; charged as
fragment +/- proton with -2..+2 hydrogen shifts. A query peak is explained when
some fragment ion lies within TOL_PPM (or TOL_DA below 100 m/z). The channel is
the explained share of sqrt intensity, averaged over the molecule's spectra.
Precursor-region peaks (within 1.5 Da of the precursor) are ignored.

    python fragexp.py scores_analog_150_coco_np.pkl
"""
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from common import DATA, PROTON, load_peaks, mrr25

RDLogger.DisableLog("rdApp.*")
TOL_PPM, TOL_DA = 10.0, 0.005
H = 1.00782503207
MAX_BONDS = 80
W = [0.0, 0.1, 0.3, 1.0, 3.0]


def fragment_masses(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    m = Chem.AddHs(m)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    mass = np.array([a.GetMass() for a in m.GetAtoms()])
    # hydrogens ride with their heavy atom
    w = mass.copy()
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1:
            nb = a.GetNeighbors()[0].GetIdx()
            w[nb] += mass[a.GetIdx()]
            w[a.GetIdx()] = 0.0
    hmap = {i: k for k, i in enumerate(heavy)}
    adj = [[] for _ in heavy]
    bonds = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in hmap and j in hmap:
            bi = len(bonds)
            bonds.append((hmap[i], hmap[j]))
            adj[hmap[i]].append((hmap[j], bi))
            adj[hmap[j]].append((hmap[i], bi))
    hw = w[heavy]
    total = hw.sum()
    n = len(heavy)

    def comps(cut):
        seen = np.full(n, -1)
        out = []
        for s in range(n):
            if seen[s] >= 0:
                continue
            stack, acc = [s], 0.0
            seen[s] = len(out)
            while stack:
                u = stack.pop()
                acc += hw[u]
                for v, bi in adj[u]:
                    if bi not in cut and seen[v] < 0:
                        seen[v] = len(out)
                        stack.append(v)
            out.append(acc)
        return out

    ms = {round(total, 4)}
    nb = min(len(bonds), MAX_BONDS)
    for a in range(nb):
        for c in comps({a}):
            ms.add(round(c, 4))
        for b in range(a + 1, nb):
            cs = comps({a, b})
            if len(cs) > 1:
                for c in cs:
                    ms.add(round(c, 4))
    return np.array(sorted(ms))


def explain(fm, spectra, polarity):
    if fm is None:
        return 0.0
    shifts = np.arange(-2, 3) * H
    ions = (fm[:, None] + shifts[None, :] + (PROTON if polarity == "positive" else -PROTON)).ravel()
    ions.sort()
    vals = []
    for pk, prec in spectra:
        mz, it = pk[:, 0], np.sqrt(np.clip(pk[:, 1], 0, None))
        k = mz < prec - 1.5
        mz, it = mz[k], it[k]
        if not len(mz) or it.sum() <= 0:
            continue
        tol = np.maximum(mz * TOL_PPM * 1e-6, TOL_DA)
        j = np.clip(np.searchsorted(ions, mz), 1, len(ions) - 1)
        d = np.minimum(np.abs(ions[j] - mz), np.abs(ions[j - 1] - mz))
        vals.append(it[d <= tol].sum() / it.sum())
    return float(np.mean(vals)) if vals else 0.0


def work(args):
    ik, cands, smis, spectra, pol = args
    return ik, [explain(fragment_masses(s), spectra, pol) for s in smis]


def main():
    t0 = time.time()
    dump = pickle.load(open(DATA + "/" + sys.argv[1], "rb"))
    dump = [d for d in dump if d[1] != 3]
    S = pd.read_parquet(DATA + "/structures.parquet")
    coco = pd.read_parquet(DATA + "/coconut.parquet")
    smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
    meta = pd.read_parquet(DATA + "/train_meta.parquet",
                           columns=["ingest_lib", "inchikey14", "precursor_mz", "ionization_mode"])
    split = pd.read_parquet(DATA + "/split.parquet")
    e = meta[meta.ingest_lib == "enveda-np-examples"]
    e = e.assign(row=e.index.values).sample(frac=1, random_state=0).groupby("inchikey14").head(16)
    q = pd.concat([split[["row", "inchikey14"]], e[["row", "inchikey14"]]])
    q = q[q.inchikey14.isin(set(d[0] for d in dump))]
    spec = load_peaks(q.row.values, lambda a, b: np.column_stack([a, b]))
    jobs = []
    for ik, cls, cands, *_ in dump:
        rows = q.row[q.inchikey14 == ik].values
        pol = meta.ionization_mode.values[rows[0]]
        rows = [r for r in rows if meta.ionization_mode.values[r] == pol]
        jobs.append((ik, cands, [smi.get(c) for c in cands],
                     [(spec[r], meta.precursor_mz.values[r]) for r in rows], pol))
    print(f"jobs {len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    with Pool(4) as p:
        fx = dict(p.imap_unordered(work, jobs, chunksize=2))
    print(f"explained ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(fx, open(DATA + "/fragexp_" + sys.argv[1], "wb"))
    rows = []
    for ik, cls, cands, lib, ana, *_ in dump:
        f = np.asarray(fx[ik])
        lib, ana = np.asarray(lib), np.asarray(ana)
        gate = np.where(lib >= 0.95, lib + 1.0, 0.0)
        rows.append((cls, "frag only", mrr25([cands[i] for i in np.argsort(-f, kind="stable")], ik)))
        for w in W:
            sc = gate + ana + w * f
            rows.append((cls, f"w={w}", mrr25([cands[i] for i in np.argsort(-sc, kind="stable")], ik)))
            sc = gate + ana * (1 + w * f)
            rows.append((cls, f"x{w}", mrr25([cands[i] for i in np.argsort(-sc, kind="stable")], ik)))
    r = pd.DataFrame(rows, columns=["cls", "chan", "mrr"])
    print(r.pivot_table(index="chan", columns="cls", values="mrr", aggfunc="mean").round(3))


if __name__ == "__main__":
    main()

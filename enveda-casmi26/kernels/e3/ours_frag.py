"""ours_frag: fragment explanation (MetFrag-lite) score, self-contained port of enveda-casmi26/eval/fragexp.py.

For each candidate, every fragment that one or two bond cuts can make (a ring opens with two cuts in the same ring)
gives a neutral mass; charged as fragment +/- proton with -2..+2 hydrogen shifts. A query peak is explained when some
fragment ion lies within TOL_PPM (or TOL_DA below 100 m/z). The score is the explained share of sqrt intensity,
averaged over the molecule's spectra. Precursor-region peaks (within 1.5 Da of the precursor) are ignored.

fragment_masses / explain are the same code as eval/fragexp.py (no eval/common.py import; PROTON inlined).
explain_mixed is explain with a per-spectrum polarity (test molecules may mix modes); with one polarity it equals
explain exactly.
"""
import time

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
PROTON = 1.007276467
TOL_PPM, TOL_DA = 10.0, 0.005
H = 1.00782503207
MAX_BONDS = 80
# decoy correction (0 = off): the score becomes explained share minus its mean over the spectra shifted by these
# offsets (Da). A candidate with more distinct 1-2 cut fragment masses explains more peaks by chance; the shifted
# spectra measure that chance rate for the same candidate.
DECOY_OFFSETS = ()


def fragment_masses(smi, deadline=None):
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
        if deadline is not None and time.time() > deadline:
            raise TimeoutError("fragment_masses")
        for c in comps({a}):
            ms.add(round(c, 4))
        for b in range(a + 1, nb):
            cs = comps({a, b})
            if len(cs) > 1:
                for c in cs:
                    ms.add(round(c, 4))
    return np.array(sorted(ms))


def explain(fm, spectra, polarity):
    """spectra: [(peaks Nx2 [mz, intensity], precursor_mz)], polarity 'positive' / 'negative'."""
    return explain_mixed(fm, [(pk, prec, polarity) for pk, prec in spectra])


def explain_mixed(fm, spectra):
    """spectra: [(peaks Nx2, precursor_mz, polarity)]."""
    if fm is None:
        return 0.0
    shifts = np.arange(-2, 3) * H
    base = (fm[:, None] + shifts[None, :]).ravel()
    ions_by = {}
    vals = []
    for pk, prec, polarity in spectra:
        ions = ions_by.get(polarity)
        if ions is None:
            ions = np.sort(base + (PROTON if polarity == "positive" else -PROTON))
            ions_by[polarity] = ions
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


def polarity_of(adduct, mode=None):
    a = str(adduct or "").strip()
    if a.endswith("+"):
        return "positive"
    if a.endswith("-"):
        return "negative"
    return "negative" if str(mode).lower().startswith("neg") else "positive"


def score_list(smis, spectra, max_secs=3.0):
    """Fragment scores for one candidate list; None when the time cap is hit (the list is then left as is)."""
    dl = time.time() + max_secs
    out = []
    for s in smis:
        fm = fragment_masses(s, deadline=dl)
        v = explain_mixed(fm, spectra)
        if DECOY_OFFSETS:
            v -= float(np.mean([explain_mixed(fm, [(np.c_[pk[:, 0] + o, pk[:, 1]], prec + o, pol)
                                                   for pk, prec, pol in spectra]) for o in DECOY_OFFSETS]))
        out.append(v)
        if time.time() > dl:
            return None
    return out


def rerank(smis, frag, lam):
    """Order of indices by z(-log rank) + lam * z(frag); z within the list, a term with std 0 is skipped."""
    r = -np.log(np.arange(1, len(smis) + 1, dtype=np.float64))
    f = np.asarray(frag, np.float64)
    z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)  # noqa: E731
    sc = z(r) + lam * z(f)
    return list(np.argsort(-sc, kind="stable"))

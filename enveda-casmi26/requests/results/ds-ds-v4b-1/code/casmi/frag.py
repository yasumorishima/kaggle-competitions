"""In-silico fragmentation (MetFrag-lite) and subformula explanation scoring.

Fragments: connected components after removing 1 or 2 bonds (any bond, rings included, so two
ring-bond cleavages open rings) of the heavy-atom graph; masses include implicit hydrogens.
Hydrogen rearrangements are handled at scoring time by +-2 H shifts.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from . import chem

H_MASS = chem.MASS['H']
PROTON = chem.PROTON
ELECTRON = chem.ELECTRON


def mol_graph(smiles: str):
    """(atom masses incl. implicit H, bonds (E,2) int32) or None."""
    r = chem._rdkit()
    Chem = r['Chem']
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    n = m.GetNumAtoms()
    w = np.zeros(n, np.float64)
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        if sym not in chem.MASS:
            return None
        w[a.GetIdx()] = chem.MASS[sym] + a.GetTotalNumHs() * H_MASS
    bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()], np.int32).reshape(-1, 2)
    return w, bonds


@njit(cache=True)
def _components(n, bonds, drop1, drop2, comp):
    """Label components after dropping bonds drop1/drop2 (-1 = none). comp: int32[n] output."""
    for i in range(n):
        comp[i] = -1
    # adjacency via edge list scan (n is small)
    E = bonds.shape[0]
    ncomp = 0
    stack = np.empty(n, np.int64)
    for s in range(n):
        if comp[s] >= 0:
            continue
        top = 0
        stack[top] = s; top += 1
        comp[s] = ncomp
        while top > 0:
            top -= 1
            u = stack[top]
            for e in range(E):
                if e == drop1 or e == drop2:
                    continue
                a = bonds[e, 0]; b = bonds[e, 1]
                v = -1
                if a == u:
                    v = b
                elif b == u:
                    v = a
                if v >= 0 and comp[v] < 0:
                    comp[v] = ncomp
                    stack[top] = v; top += 1
        ncomp += 1
    return ncomp


@njit(cache=True)
def fragment_masses(w, bonds, max_depth, max_out):
    """All fragment masses (incl. parent) for up to `max_depth` (1 or 2) bond removals.
    Returns sorted unique masses (rounded to 1e-3 Da)."""
    n = len(w); E = bonds.shape[0]
    out = np.empty(max_out, np.float64)
    k = 0
    tot = 0.0
    for i in range(n):
        tot += w[i]
    out[k] = tot; k += 1
    comp = np.empty(n, np.int32)
    cm = np.empty(n, np.float64)
    if E > 0:
        for e1 in range(E):
            nc = _components(n, bonds, e1, -1, comp)
            if nc > 1:
                for c in range(nc):
                    cm[c] = 0.0
                for i in range(n):
                    cm[comp[i]] += w[i]
                for c in range(nc):
                    if k < max_out:
                        out[k] = cm[c]; k += 1
            if max_depth >= 2:
                for e2 in range(e1 + 1, E):
                    nc2 = _components(n, bonds, e1, e2, comp)
                    if nc2 > 1:
                        for c in range(nc2):
                            cm[c] = 0.0
                        for i in range(n):
                            cm[comp[i]] += w[i]
                        for c in range(nc2):
                            if k < max_out:
                                out[k] = cm[c]; k += 1
    res = out[:k]
    # round + unique
    r = np.empty(k, np.float64)
    for i in range(k):
        r[i] = np.floor(res[i] * 1000.0 + 0.5) / 1000.0
    r.sort()
    u = np.empty(k, np.float64)
    m = 0
    for i in range(k):
        if i == 0 or r[i] != r[i - 1]:
            u[m] = r[i]; m += 1
    return u[:m]


def fragments_for_smiles(smiles: str, max_depth: int = 2, max_out: int = 200000) -> np.ndarray:
    g = mol_graph(smiles)
    if g is None:
        return np.zeros(0, np.float64)
    w, bonds = g
    if len(bonds) == 0:
        return np.array([w.sum()])
    return fragment_masses(w, bonds, max_depth, max_out)


@njit(cache=True, fastmath=True)
def explain_peaks(frag, peak_mz, peak_w, ion_delta, hshift_max, tol_da, tol_ppm):
    """For each peak: is there a fragment mass f such that |peak - (f + ion_delta + s*H)| <= tol
    for some integer s in [-hshift_max, hshift_max]? Returns bool array."""
    n = len(peak_mz); m = len(frag)
    ok = np.zeros(n, np.bool_)
    if m == 0:
        return ok
    for i in range(n):
        p = peak_mz[i]
        t = tol_da
        pt = p * tol_ppm * 1e-6
        if pt > t:
            t = pt
        for s in range(-hshift_max, hshift_max + 1):
            target = p - ion_delta - s * H_MASS
            j = np.searchsorted(frag, target)
            if j < m and abs(frag[j] - target) <= t:
                ok[i] = True; break
            if j > 0 and abs(frag[j - 1] - target) <= t:
                ok[i] = True; break
    return ok


def ion_deltas(adduct: str) -> List[float]:
    """Mass offsets fragment(neutral) -> fragment ion, for the query adduct. Positive mode:
    protonated fragment (+H-e) plus, for metal adducts, the metal-cationised fragment."""
    n, z, d, sign = chem.ADDUCTS.get(adduct, chem.ADDUCTS['[M+H]+' if adduct.endswith('+') else '[M-H]-'])
    if sign > 0:
        out = [PROTON]
        if 'Na' in adduct:
            out.append(chem.MASS['Na'] - ELECTRON)
        if 'K' in adduct:
            out.append(chem.MASS['K'] - ELECTRON)
        if 'NH4' in adduct:
            out.append(chem.MASS['N'] + 4 * H_MASS - ELECTRON)
        return out
    else:
        out = [-PROTON]
        if 'Cl' in adduct:
            out.append(chem.MASS['Cl'] + ELECTRON)
        if 'CH2O2' in adduct or 'HCOO' in adduct:
            out.append(chem.MASS['C'] + 2 * H_MASS + 2 * chem.MASS['O'] - PROTON)
        return out


def frag_features(frag: np.ndarray, peaks: List[Tuple[np.ndarray, np.ndarray, str]],
                  tol_da=0.01, tol_ppm=15.0, hshift=2) -> np.ndarray:
    """peaks: list of (mz, intensity(base=1), adduct) cleaned query spectra.
    Returns [max_int_frac, mean_int_frac, max_cnt_frac, top10_frac, strict_int_frac(hshift 1)]"""
    if len(frag) == 0 or not peaks:
        return np.zeros(5, np.float32)
    ints, cnts, tops, stricts = [], [], [], []
    for mz, it, adduct in peaks:
        if len(mz) == 0:
            continue
        w = np.sqrt(np.asarray(it, np.float64)); tot = w.sum()
        if tot <= 0:
            continue
        ok = np.zeros(len(mz), bool); ok1 = np.zeros(len(mz), bool)
        for d in ion_deltas(adduct):
            ok |= explain_peaks(frag, np.asarray(mz, np.float64), w, d, hshift, tol_da, tol_ppm)
            ok1 |= explain_peaks(frag, np.asarray(mz, np.float64), w, d, 1, tol_da, tol_ppm)
        ints.append(w[ok].sum() / tot); cnts.append(ok.mean())
        top = np.argsort(-w)[:10]; tops.append(ok[top].mean())
        stricts.append(w[ok1].sum() / tot)
    if not ints:
        return np.zeros(5, np.float32)
    return np.array([max(ints), float(np.mean(ints)), max(cnts), max(tops), max(stricts)], np.float32)


# ----------------------------------------------------------------------------------------------
# Subformula explanation (formula-level evidence, independent of connectivity)
# ----------------------------------------------------------------------------------------------
_EL = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
_EL_MASS = np.array([chem.MASS[e] for e in _EL])


def formula_vec(f: str) -> Optional[np.ndarray]:
    c = chem.parse_formula(f)
    if c is None:
        return None
    if any(e not in _EL for e in c):
        return None
    return np.array([c.get(e, 0) for e in _EL], np.int64)


@njit(cache=True)
def subformula_masses(counts, masses, max_out):
    """Enumerate all subformulas (with valence-plausibility: H <= 2C + N + 2 + (P+... ) roughly,
    RDBE >= -1) and return their sorted masses."""
    nC, nH, nN, nO, nP, nS, nF, nCl, nBr, nI = counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7], counts[8], counts[9]
    out = np.empty(max_out, np.float64)
    k = 0
    hal_tot = nF + nCl + nBr + nI
    for c in range(nC + 1):
        for n in range(nN + 1):
            for o in range(nO + 1):
                for p in range(nP + 1):
                    for s in range(nS + 1):
                        for hal in range(hal_tot + 1):
                            base = c * masses[0] + n * masses[2] + o * masses[3] + p * masses[4] + s * masses[5]
                            # halogen mass: approximate by distributing over available halogens (enumerate combos only if few)
                            # simple approach: enumerate combos of (F,Cl,Br,I) with total hal
                            for f in range(min(nF, hal) + 1):
                                for cl in range(min(nCl, hal - f) + 1):
                                    for br in range(min(nBr, hal - f - cl) + 1):
                                        i_ = hal - f - cl - br
                                        if i_ > nI:
                                            continue
                                        hmax = 2 * c + n + 2 + p * 2 + 0
                                        hm = nH if nH < hmax else hmax
                                        for h in range(hm + 1):
                                            if c + n + o + p + s + hal + h == 0:
                                                continue
                                            # RDBE = C - (H+hal)/2 + (N+P)/2 + 1 >= -1
                                            rdbe2 = 2 * c - (h + hal) + (n + p) + 2
                                            if rdbe2 < -2:
                                                continue
                                            if k >= max_out:
                                                return out[:k]
                                            out[k] = base + h * masses[1] + f * masses[6] + cl * masses[7] + br * masses[8] + i_ * masses[9]
                                            k += 1
    r = out[:k]
    r.sort()
    return r


def subformula_features(fvec: np.ndarray, peaks: List[Tuple[np.ndarray, np.ndarray, str]],
                        tol_da=0.005, tol_ppm=10.0, max_out=2000000) -> np.ndarray:
    """[max_int_frac, mean_int_frac, cnt_frac, top10_frac] of peaks explainable as subformula ions."""
    if fvec is None or not peaks:
        return np.zeros(4, np.float32)
    sub = subformula_masses(fvec, _EL_MASS, max_out)
    ints, cnts, tops = [], [], []
    for mz, it, adduct in peaks:
        if len(mz) == 0:
            continue
        w = np.sqrt(np.asarray(it, np.float64)); tot = w.sum()
        if tot <= 0:
            continue
        ok = np.zeros(len(mz), bool)
        for d in ion_deltas(adduct):
            ok |= explain_peaks(sub, np.asarray(mz, np.float64), w, d, 0, tol_da, tol_ppm)
        ints.append(w[ok].sum() / tot); cnts.append(ok.mean())
        top = np.argsort(-w)[:10]; tops.append(ok[top].mean())
    if not ints:
        return np.zeros(4, np.float32)
    return np.array([max(ints), float(np.mean(ints)), max(cnts), max(tops)], np.float32)

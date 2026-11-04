"""ours_ae (yasunorim, 11-04): analog-edit candidate for rank 2.

Idea credited to heliosli's r3p overlay (public notebook heliosli/casmi26-0-481, cell 24): a test spectrum whose
modified-cosine analog in the training library sits exactly one natural-product modification away gives a candidate
structure = the analog edited by that modification. Our code, written from scratch, differs in:
  - all 10 test adducts (heliosli: [M+H]+, [M-H]-, [M+Na]+ only), matched within the same ionization mode and adduct;
  - his 19 modifications plus hydrogenation / dehydrogenation (H2) and NH (amine) edits;
  - the product is chosen by our fragment explanation of the query spectra (ours_frag: 1-2 bond cuts), not by a
    learned site model (heliosli's site model is a private dataset);
  - the library index is built by us from the competition train.parquet (eval/ae/build_lib.py).
Only one structure per molecule is added, at rank 2; rank 1 and the order of the rest are unchanged; products already
in the list or the pipeline's pools are skipped (they are class 1/2 answers the pipeline already ranks).
"""
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

MODS = {   # kind: (neutral mass delta, [(site SMARTS with :1, group SMILES attached at the site)])
    'CH2': (14.015650, [('[OX2H:1]', 'C'), ('[NX3;H1,H2;!$(NC=O):1]', 'C'), ('[c;H1:1]', 'C'), ('[CX4;H3:1]', 'C')]),
    'O': (15.994915, [('[c;H1:1]', 'O'), ('[CX4;H1,H2,H3:1]', 'O')]),
    'NH': (15.010899, [('[c;H1:1]', 'N'), ('[CX4;H1,H2,H3:1]', 'N')]),
    'CH2O': (30.010565, [('[c;H1:1]', 'OC')]),
    'C2H2O': (42.010565, [('[OX2H:1]', 'C(C)=O'), ('[NX3;H1,H2:1]', 'C(C)=O')]),
    'CO2': (43.989829, [('[c;H1:1]', 'C(=O)O')]),
    'C5H8': (68.062600, [('[c;H1:1]', 'CC=C(C)C'), ('[OX2H:1]', 'CC=C(C)C')]),
    'SO3': (79.956815, [('[OX2H:1]', 'S(=O)(=O)O')]),
    'C3H2O3': (86.000394, [('[OX2H:1]', 'C(=O)CC(=O)O')]),
    'C7H4O': (104.026215, [('[OX2H:1]', 'C(=O)c1ccccc1')]),
    'C5H8O4': (132.042259, [('[OX2H:1]', 'C1OCC(O)C(O)C1O')]),
    'C6H10O4': (146.057909, [('[OX2H:1]', 'C1OC(C)C(O)C(O)C1O')]),
    'C9H6O2': (146.036779, [('[OX2H:1]', 'C(=O)C=Cc1ccc(O)cc1')]),
    'C7H4O4': (152.010959, [('[OX2H:1]', 'C(=O)c1cc(O)c(O)c(O)c1')]),
    'C6H10O5': (162.052824, [('[OX2H:1]', 'C1OC(CO)C(O)C(O)C1O'), ('[c;H1:1]', 'C1OC(CO)C(O)C(O)C1O')]),
    'C9H6O3': (162.031694, [('[OX2H:1]', 'C(=O)C=Cc1ccc(O)c(O)c1')]),
    'C6H8O6': (176.032088, [('[OX2H:1]', 'C1OC(C(=O)O)C(O)C(O)C1O')]),
    'C10H8O3': (176.047344, [('[OX2H:1]', 'C(=O)C=Cc1ccc(O)c(OC)c1')]),
    'C11H10O4': (206.057909, [('[OX2H:1]', 'C(=O)C=Cc1cc(OC)c(O)c(OC)c1')]),
    'C12H20O10': (324.105647, [('[OX2H:1]', 'C1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O')]),
}
H2 = 2.015650
_RXN = {k: [AllChem.ReactionFromSmarts(f"{s}>>[*:1]{g}") for s, g in v[1]] for k, v in MODS.items()}
_HYD = AllChem.ReactionFromSmarts("[C:1]=[C:2]>>[C:1][C:2]")
_DEHYD = AllChem.ReactionFromSmarts("[CX4;!H0:1]-[CX4;!H0:2]>>[C:1]=[C:2]")
TARGETS = [(k, v[0], "add") for k, v in MODS.items()] + [(k, -v[0], "remove") for k, v in MODS.items()] + \
          [("H2", H2, "add"), ("H2", -H2, "remove")]

try:
    import numba

    _jit = numba.njit(cache=False)
except Exception:   # noqa: BLE001
    def _jit(f):
        return f


@_jit
def modcos(qm, qi, lm, li, shift, tol):
    """Greedy modified cosine: a peak pair matches unshifted or shifted by the precursor difference."""
    nq, nl = len(qm), len(lm)
    if nq == 0 or nl == 0:
        return 0.0
    cs = np.empty(nq * nl * 2)
    ci = np.empty(nq * nl * 2, np.int64)
    cj = np.empty(nq * nl * 2, np.int64)
    c = 0
    for i in range(nq):
        for j in range(nl):
            d = qm[i] - lm[j]
            if abs(d) <= tol or abs(d - shift) <= tol:
                cs[c] = qi[i] * li[j]
                ci[c] = i
                cj[c] = j
                c += 1
    if c == 0:
        return 0.0
    order = np.argsort(-cs[:c])
    ui = np.zeros(nq, np.bool_)
    uj = np.zeros(nl, np.bool_)
    num = 0.0
    for k in range(c):
        o = order[k]
        if ui[ci[o]] or uj[cj[o]]:
            continue
        ui[ci[o]] = True
        uj[cj[o]] = True
        num += cs[o]
    a = (qi * qi).sum()
    b = (li * li).sum()
    return num / np.sqrt(a * b) if a > 0 and b > 0 else 0.0


def _key(m):
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:   # noqa: BLE001
        return None


def edits(smiles, kind, direction):
    """Single edits of `smiles` by one modification; {inchikey14: smiles}."""
    m = Chem.MolFromSmiles(smiles)
    out = {}
    if m is None:
        return out
    if kind == "H2":
        rx = _HYD if direction == "add" else _DEHYD
        try:
            prods = rx.RunReactants((m,))
        except Exception:   # noqa: BLE001
            prods = ()
        for (p,) in prods[:300]:
            try:
                Chem.SanitizeMol(p)
                k = _key(p)
                if k and k not in out:
                    out[k] = Chem.MolToSmiles(p)
            except Exception:   # noqa: BLE001
                pass
        return out
    if direction == "add":
        for r in _RXN[kind]:
            try:
                prods = r.RunReactants((m,))
            except Exception:   # noqa: BLE001
                continue
            for (p,) in prods[:400]:
                try:
                    Chem.SanitizeMol(p)
                    k = _key(p)
                    if k and k not in out:
                        out[k] = Chem.MolToSmiles(p)
                except Exception:   # noqa: BLE001
                    pass
        return out
    # remove: cut one acyclic single bond between heavy atoms; keep the piece that is lighter by the delta
    mass = MODS[kind][0]
    mw = Descriptors.ExactMolWt(m)
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        try:
            frag = Chem.FragmentOnBonds(m, [b.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
            for piece in Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False):
                rw = Chem.RWMol(piece)
                for at in rw.GetAtoms():
                    if at.GetAtomicNum() == 0:
                        at.SetAtomicNum(1)
                        at.SetIsotope(0)
                p = rw.GetMol()
                Chem.SanitizeMol(p)
                p = Chem.RemoveHs(p)
                if abs((mw - Descriptors.ExactMolWt(p)) - mass) <= 0.002:
                    k = _key(p)
                    if k and k not in out:
                        out[k] = Chem.MolToSmiles(p)
        except Exception:   # noqa: BLE001
            pass
    return out


class Library:
    def __init__(self, npz, struct_smiles, groups):
        z = np.load(npz)
        self.mz, self.it, self.n, self.prec, self.grp, self.sidx = z["mz"], z["it"], z["n"], z["prec"], z["grp"], z["sidx"]
        self.smiles = list(struct_smiles)
        self.gid = {g: i for i, g in enumerate(groups)}
        self.by_g = {}
        for g in range(len(groups)):
            rows = np.flatnonzero(self.grp == g)
            o = np.argsort(self.prec[rows], kind="stable")
            self.by_g[g] = (rows[o], self.prec[rows[o]])
        self.mask = None   # optional boolean array over structures: True = hidden (simulation)

    def analogs(self, specs, tau, top=5, tol=0.01, ptol=0.005):
        """Best score per (structure, kind, direction) over all query spectra, >= tau, best first (at most `top`).
        Each item: (score, struct index, kind, direction, group, precursor, query mz, query int, lib row)."""
        hits = {}
        for g, prec, qm, qi in specs:
            gi = self.gid.get(g)
            if gi is None:
                continue
            rows, precs = self.by_g[gi]
            for kind, d, direction in TARGETS:
                lo, hi = np.searchsorted(precs, prec - d - ptol), np.searchsorted(precs, prec - d + ptol)
                for t in range(lo, hi):
                    r = rows[t]
                    si = int(self.sidx[r])
                    if self.mask is not None and self.mask[si]:
                        continue
                    nl = int(self.n[r])
                    s = modcos(qm, qi, self.mz[r, :nl].astype(np.float64), self.it[r, :nl].astype(np.float64),
                               float(prec - self.prec[r]), tol)
                    k = (si, kind, direction)
                    if s >= tau and (k not in hits or s > hits[k][0]):
                        hits[k] = (s, si, kind, direction, g, prec, qm, qi, r)
        return sorted(hits.values(), key=lambda x: -x[0])[:top]

    def best_analog(self, specs, tol=0.01, ptol=0.005):
        """specs: list of (group, precursor, mz, sqrt-int). Returns (score, struct index, kind, direction) or None."""
        best = None
        for g, prec, qm, qi in specs:
            gi = self.gid.get(g)
            if gi is None:
                continue
            rows, precs = self.by_g[gi]
            for kind, d, direction in TARGETS:
                lo, hi = np.searchsorted(precs, prec - d - ptol), np.searchsorted(precs, prec - d + ptol)
                for t in range(lo, hi):
                    r = rows[t]
                    si = int(self.sidx[r])
                    if self.mask is not None and self.mask[si]:
                        continue
                    nl = int(self.n[r])
                    s = modcos(qm, qi, self.mz[r, :nl].astype(np.float64), self.it[r, :nl].astype(np.float64),
                               float(prec - self.prec[r]), tol)
                    if best is None or s > best[0]:
                        best = (s, si, kind, direction)
        return best


def query_specs(rows, max_specs=4, k=32):
    """rows: iterable of (ionization_mode, adduct, precursor_mz, mzs, intensities) of one molecule."""
    out = []
    for mode, adduct, prec, mz, it in rows:
        mz = np.asarray(mz, np.float64)
        it = np.asarray(it, np.float64)
        m = (mz <= prec + 1.0) & (it > 0)
        if m.sum() < 3:
            continue
        mz, it = mz[m], it[m]
        o = np.argsort(-it)[:k]
        out.append((f"{mode}|{adduct}", float(prec), mz[o], np.sqrt(it[o] / it[o[0]])))
    out.sort(key=lambda x: -len(x[2]))
    return out[:max_specs]


def propose(lib, specs, frag_rows, frag_mod, exclude=(), tau=0.85, max_products=60, deadline=None):
    """One analog-edit structure for a molecule, or None. frag_rows: spectra for ours_frag.score_list.
    Returns (smiles, info dict)."""
    best = lib.best_analog(specs)
    if best is None or best[0] < tau:
        return None, dict(gate=0, modcos=None if best is None else round(float(best[0]), 3))
    s, si, kind, direction = best
    prods = edits(lib.smiles[si], kind, direction)
    info = dict(gate=1, modcos=round(float(s), 3), kind=kind, direction=direction, n_products=len(prods))
    cand = [(k, smi) for k, smi in prods.items() if k not in exclude][:max_products]
    info["n_new"] = len(cand)
    if not cand:
        return None, info
    if len(cand) == 1 or frag_mod is None:
        return cand[0][1], dict(info, keys=[c[0] for c in cand])
    left = 3.0 if deadline is None else max(0.2, min(3.0, deadline - time.time()))
    sc = frag_mod.score_list([c[1] for c in cand], frag_rows, max_secs=left)
    if sc is None:
        return cand[0][1], dict(info, keys=[c[0] for c in cand], frag="timeout")
    sc = np.nan_to_num(np.asarray(sc, float), nan=-1.0)
    i = int(np.argmax(sc))
    return cand[i][1], dict(info, keys=[c[0] for c in cand], pick=cand[i][0])


def shifted_peaks(qm, qi, lm, shift, tol=0.01):
    """Query peaks that match a library peak only after the precursor shift (they carry the modification)."""
    out = []
    for i in range(len(qm)):
        d = qm[i] - lm
        if np.any(np.abs(d - shift) <= tol) and not np.any(np.abs(d) <= tol):
            out.append(i)
    return np.array(out, int)


def propose_v(lib, specs, frag_rows, frag_mod, exclude=(), tau=0.95, top=5, mode="vote", deadline=None):
    """Variant pickers over several analogs. mode: 'vote' = sum of analog scores per product, ties by fragment
    explanation of all query peaks; 'shift' = best analog only, fragment explanation of the shifted query peaks;
    'voteshift' = vote, ties by the shifted-peak explanation."""
    an = lib.analogs(specs, tau, top=top)
    if not an:
        return None, dict(gate=0)
    votes, shiftset = {}, {}
    for s, si, kind, direction, g, prec, qm, qi, r in (an if mode != "shift" else an[:1]):
        nl = int(lib.n[r])
        sp = shifted_peaks(qm, qi, lib.mz[r, :nl].astype(np.float64), float(prec - lib.prec[r]))
        for k, smi in edits(lib.smiles[si], kind, direction).items():
            if k in exclude:
                continue
            v = votes.setdefault(k, [smi, 0.0, 0])
            v[1] += s
            v[2] += 1
            if len(sp):
                shiftset.setdefault(k, []).append((g, prec, qm[sp], qi[sp]))
    info = dict(gate=1, modcos=round(float(an[0][0]), 3), kind=an[0][2], direction=an[0][3], n_analogs=len(an), n_new=len(votes))
    if not votes:
        return None, info
    keys = list(votes)
    smis = [votes[k][0] for k in keys]
    if mode == "shift" or mode == "voteshift":
        pol = lambda g: "positive" if g.startswith("positive") else "negative"   # noqa: E731
        fr = []
        for k in keys:
            ss = shiftset.get(k, [])
            fm = frag_mod.fragment_masses(votes[k][0])
            fr.append(frag_mod.explain_mixed(fm, [(np.c_[m_, i_ ** 2], p_ + 10.0, pol(g_)) for g_, p_, m_, i_ in ss]) if ss else 0.0)
        fr = np.asarray(fr)
    else:
        sc = frag_mod.score_list(smis, frag_rows, max_secs=3.0)
        fr = np.zeros(len(keys)) if sc is None else np.nan_to_num(np.asarray(sc, float), nan=-1.0)
    if mode == "shift":
        i = int(np.argmax(fr))
    else:
        vs = np.array([votes[k][1] for k in keys])
        i = int(np.lexsort((-fr, -vs))[0])
    return smis[i], dict(info, keys=keys, pick=keys[i], votes=int(votes[keys[i]][2]))

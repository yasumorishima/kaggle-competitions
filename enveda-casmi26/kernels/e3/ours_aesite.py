"""ours_aesite (yasunorim, 11-07): features of each analog-edit product, for a learned pick among the products (ae v1).

ae v0 (ours_ae.propose_v) picks the product by analog votes, ties by our fragment explanation: in the 11-04 simulation the
answer is among the products for 25% of the gated molecules but picked for 11%. Here every product gets
  - its edit site on the analog (element, aromatic, ring, degree, Hs, hetero neighbours, carbonyl neighbour, symmetry
    class size, how central the site is), the edit kind / direction / reaction index,
  - its spectral support (votes, summed and best modified cosine of the analogs proposing it, fragment explanation of all
    query peaks and of the shifted peaks only, both also relative to the best product of the molecule),
  - the molecule's context (products, analogs, best cosine),
so a ranker fitted on simulated molecules (eval/ae/site.py) can pick among them.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

import ours_ae as _ae

KINDS = sorted(_ae.MODS) + ["H2"]
KIND_ID = {k: i for i, k in enumerate(KINDS)}
SITE_COLS = ["s_z", "s_arom", "s_ring", "s_ring6", "s_deg", "s_hs", "s_het_nb", "s_arom_nb", "s_co_nb", "s_sym",
             "s_central", "s_dist_o", "s_heavy", "s_frac_kept"]
COLS = ["kind", "direction", "rxn"] + SITE_COLS + [
    "votes", "vote_sum", "vote_max", "fr", "fr_rel", "fr_rank", "frs", "frs_rel", "n_new", "n_analogs", "best_cos",
    "n_same_kind"]


def _site_feats(m, idx, ranks, dmat, kept_frac=1.0):
    a = m.GetAtomWithIdx(int(idx))
    nb = list(a.GetNeighbors())
    co = 0
    for x in nb:
        if x.GetAtomicNum() == 6:
            for b in x.GetBonds():
                o = b.GetOtherAtom(x)
                if o.GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE:
                    co = 1
    ri = m.GetRingInfo()
    d = dmat[int(idx)]
    os_ = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).GetAtomicNum() == 8 and i != int(idx)]
    return [a.GetAtomicNum(), int(a.GetIsAromatic()), int(a.IsInRing()), int(ri.IsAtomInRingOfSize(int(idx), 6)), a.GetDegree(),
            a.GetTotalNumHs(), sum(x.GetAtomicNum() not in (1, 6) for x in nb), sum(x.GetIsAromatic() for x in nb), co,
            int(np.sum(ranks == ranks[int(idx)])), float(d.max() / max(dmat.max(), 1)), float(min((d[i] for i in os_), default=20)),
            m.GetNumHeavyAtoms(), kept_frac]


def edits_info(smiles, kind, direction):
    """{inchikey14: (smiles, rxn index, site feature list)} for the single edits of `smiles` (same products as ours_ae.edits)."""
    m = Chem.MolFromSmiles(smiles)
    out = {}
    if m is None:
        return out
    ranks = np.array(list(Chem.CanonicalRankAtoms(m, breakTies=False)))
    dmat = Chem.GetDistanceMatrix(m)
    if kind == "H2":
        rx = _ae._HYD if direction == "add" else _ae._DEHYD
        try:
            prods = rx.RunReactants((m,))
        except Exception:   # noqa: BLE001
            prods = ()
        for (p,) in prods[:300]:
            try:
                site = [a.GetIntProp("react_atom_idx") for a in p.GetAtoms() if a.HasProp("old_mapno") and a.GetIntProp("old_mapno") == 1]
                Chem.SanitizeMol(p)
                k = _ae._key(p)
                if k and k not in out:
                    out[k] = (Chem.MolToSmiles(p), 0, _site_feats(m, site[0] if site else 0, ranks, dmat))
            except Exception:   # noqa: BLE001
                pass
        return out
    if direction == "add":
        for ri, r in enumerate(_ae._RXN[kind]):
            try:
                prods = r.RunReactants((m,))
            except Exception:   # noqa: BLE001
                continue
            for (p,) in prods[:400]:
                try:
                    site = [a.GetIntProp("react_atom_idx") for a in p.GetAtoms() if a.HasProp("old_mapno") and a.GetIntProp("old_mapno") == 1]
                    Chem.SanitizeMol(p)
                    k = _ae._key(p)
                    if k and k not in out:
                        out[k] = (Chem.MolToSmiles(p), ri, _site_feats(m, site[0] if site else 0, ranks, dmat))
                except Exception:   # noqa: BLE001
                    pass
        return out
    mass = _ae.MODS[kind][0]
    mw = Descriptors.ExactMolWt(m)
    n0 = m.GetNumHeavyAtoms()
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        try:
            frag = Chem.FragmentOnBonds(m, [b.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
            amap = []
            pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=amap)
            for pi, piece in enumerate(pieces):
                rw = Chem.RWMol(piece)
                for at in rw.GetAtoms():
                    if at.GetAtomicNum() == 0:
                        at.SetAtomicNum(1)
                        at.SetIsotope(0)
                p = rw.GetMol()
                Chem.SanitizeMol(p)
                p = Chem.RemoveHs(p)
                if abs((mw - Descriptors.ExactMolWt(p)) - mass) <= 0.002:
                    k = _ae._key(p)
                    if k and k not in out:
                        ids = set(amap[pi])
                        site = b.GetBeginAtomIdx() if b.GetBeginAtomIdx() in ids else b.GetEndAtomIdx()
                        out[k] = (Chem.MolToSmiles(p), 0, _site_feats(m, site, ranks, dmat, p.GetNumHeavyAtoms() / max(n0, 1)))
        except Exception:   # noqa: BLE001
            pass
    return out


def product_rows(lib, specs, frag_rows, frag_mod, exclude=(), tau=0.95, top=5):
    """(rows, info): one feature row per new product (COLS order) and its smiles / key; None when the gate is closed."""
    an = lib.analogs(specs, tau, top=top)
    if not an:
        return None, dict(gate=0)
    prod = {}
    for s, si, kind, direction, g, prec, qm, qi, r in an:
        nl = int(lib.n[r])
        sp = _ae.shifted_peaks(qm, qi, lib.mz[r, :nl].astype(np.float64), float(prec - lib.prec[r]))
        for k, (smi, rxi, site) in edits_info(lib.smiles[si], kind, direction).items():
            if k in exclude:
                continue
            v = prod.get(k)
            if v is None:
                v = prod[k] = dict(smi=smi, kind=KIND_ID[kind], direction=int(direction == "add"), rxn=rxi, site=site,
                                   votes=0, vote_sum=0.0, vote_max=0.0, shift=[])
            v["votes"] += 1
            v["vote_sum"] += s
            v["vote_max"] = max(v["vote_max"], s)
            if len(sp):
                v["shift"].append((g, prec, qm[sp], qi[sp]))
    info = dict(gate=1, n_new=len(prod), n_analogs=len(an), modcos=round(float(an[0][0]), 3))
    if not prod:
        return None, info
    keys = list(prod)
    smis = [prod[k]["smi"] for k in keys]
    sc = frag_mod.score_list(smis, frag_rows, max_secs=3.0)
    fr = np.zeros(len(keys)) if sc is None else np.nan_to_num(np.asarray(sc, float), nan=-1.0)
    pol = lambda g: "positive" if g.startswith("positive") else "negative"   # noqa: E731
    frs = []
    for k in keys:
        ss = prod[k]["shift"]
        try:
            fm = frag_mod.fragment_masses(prod[k]["smi"])
            frs.append(frag_mod.explain_mixed(fm, [(np.c_[m_, i_ ** 2], p_ + 10.0, pol(g_)) for g_, p_, m_, i_ in ss]) if ss else 0.0)
        except Exception:   # noqa: BLE001
            frs.append(0.0)
    frs = np.asarray(frs, float)
    frank = (-fr).argsort().argsort()
    kc = {}
    for k in keys:
        kc[prod[k]["kind"]] = kc.get(prod[k]["kind"], 0) + 1
    rows = []
    for j, k in enumerate(keys):
        v = prod[k]
        rows.append([v["kind"], v["direction"], v["rxn"]] + list(v["site"]) + [
            v["votes"], v["vote_sum"], v["vote_max"], fr[j], fr[j] - fr.max(), int(frank[j]), frs[j], frs[j] - frs.max(),
            len(keys), len(an), float(an[0][0]), kc[v["kind"]]])
    return np.asarray(rows, float), dict(info, keys=keys, smiles=smis)

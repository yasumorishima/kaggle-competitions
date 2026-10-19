"""ours (yasunorim, 10-19): PubChemLite CC0 candidates into engine 1 (v4n), through the de novo injection route.

The v14/v15 base adds the PubChemLite CC0 structures (franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints)
to engine 2's pool only (public LB 0.451 -> 0.455). In the two-engine fusion a candidate with one engine's vote loses
to candidates both engines voted for (our e3val4: PubChem-only candidates lost 48 rank-1s in the fusion, gained 0),
so the PubChemLite structures that engine 1 never sees are held back. Here they also enter engine 1:

  pass 1  engine 1 runs as before; its top formulas are chosen as for de novo (ours_denovo.choose_formulas);
  lookup  PubChemLite structures whose neutral monoisotopic mass is within PPM of the formula's exact mass and whose
          formula (RDKit) equals it, not already among the pass-1 candidates; ordered by our fp2 log likelihood
          (Morgan r2 bits under our fp2 nets' bit probabilities, as for de novo); the top MAX_NEW are kept;
  pass 2  engine 1 runs again with them injected through pc_join (pool-like rows), so its features, ranker,
          ICEBERG / GLACIER and the fusion see them like any other candidate (and both engines can now vote).

install(ours_denovo) swaps ours_denovo.load (fp2 nets only, no dn1) and ours_denovo.propose; everything else
(formula choice, injection, time cap, fallbacks to the pass-1 result) is ours_denovo unchanged.
"""
import glob
import os
import pickle

import numpy as np

PPM = 5.0
MAX_SCAN = 400          # PubChemLite rows looked at per formula (mass window), before the formula check
STATS = dict(lookups=0, window=0, formula_ok=0, kept=0, no_data=0)
_D = {}
MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196, "S": 31.97207100, "P": 30.97376163,
        "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325, "B": 11.0093054,
        "Se": 79.9165213}


def _data(log=print):
    if "mass" not in _D:
        hits = sorted(glob.glob("/kaggle/input/**/pubchemlite_mass.npy", recursive=True))
        if not hits:
            raise FileNotFoundError("pubchemlite_mass.npy not mounted")
        d = os.path.dirname(hits[0])
        m = pickle.load(open(os.path.join(d, "pubchemlite_meta.pkl"), "rb"))
        _D.update(mass=np.load(hits[0]).astype(np.float64), smiles=np.asarray(m["smiles"], dtype=object),
                  keys=np.asarray(m["keys"], dtype=object))
        assert len(_D["mass"]) == len(_D["smiles"]) == len(_D["keys"])
        assert np.all(np.diff(_D["mass"]) >= 0), "PubChemLite mass not sorted"
        log(f"ours_pcl: PubChemLite {len(_D['mass']):,} structures from {d}")
    return _D


def install(dn, log=print):
    import ours_fp2

    def load(pattern=None, fp2_only=None, device=None, log=log, fp2_pattern="/kaggle/input/**/fp2.pt"):
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        fpp, fpn = ours_fp2.load(fp2_pattern, only=fp2_only, device=dev)
        dn._M.update(dev=dev, fp2=fpn, path="pubchemlite")
        _data(log)
        log(f"ours_pcl: fp2 nets {fpp} on {dev}")
        return dn._M

    def propose(spectra, forms, exclude_ik14=(), seed=0):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        D = _data()
        cands = {}
        for fi, (f, fc) in enumerate(forms):
            STATS["lookups"] += 1
            em = sum(int(n) * MASS[e] for e, n in zip(dn.FORM_ELEMS, fc) if n)
            tol = em * PPM * 1e-6
            a = int(np.searchsorted(D["mass"], em - tol, "left"))
            b = int(np.searchsorted(D["mass"], em + tol, "right"))
            b = min(b, a + MAX_SCAN)
            STATS["window"] += b - a
            for j in range(a, b):
                m = Chem.MolFromSmiles(str(D["smiles"][j]))
                if m is None:
                    continue
                fc2 = dn.form_counts(CalcMolFormula(m))
                if fc2 is None or not np.array_equal(fc2, fc):
                    continue
                STATS["formula_ok"] += 1
                try:
                    ik = Chem.MolToInchiKey(m)[:14]
                except Exception:
                    continue
                if not ik or ik in exclude_ik14 or ik in cands:
                    continue
                cands[ik] = [Chem.MolToSmiles(m), 1, 0.0, m, fi]
        if not cands:
            return [], False
        keys = list(cands)
        p = dn.fp2_probs(spectra)
        if p is None:
            return [(k, cands[k], float("nan")) for k in keys], False
        if "g2" not in dn._M:
            dn._M["g2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        B = np.stack([np.asarray(dn._M["g2"].GetFingerprintAsNumPy(cands[k][3]), np.float64) for k in keys])
        ll = B @ np.log(p) + (1 - B) @ np.log(1 - p)
        order = np.argsort(-ll, kind="stable")
        STATS["kept"] += min(len(keys), dn.MAX_NEW)
        return [(keys[i], cands[keys[i]], float(ll[i])) for i in order], True

    dn.load = load
    dn.propose = propose
    log(f"ours_pcl installed: PubChemLite into engine 1, ppm {PPM}, max new {dn.MAX_NEW}")

"""ours (yasunorim): natural-product-likeness prior for the final top-K list.

The hidden test is Enveda's natural products; the candidate windows mix natural products with synthetic isomers
(PubChem, the non-NP train libraries). Each candidate gets the Ertl NP-likeness score (Ertl, Roggo, Schuffenhauer,
J Chem Inf Model 48:68 (2008); the scorer and model shipped in RDKit Contrib/NP_Score, used as is). The list is
re-ranked by z(-log rank) + lam * z(NP score).
On train (enveda-np-examples, 171 molecules, the truth among 49 same-formula train isomers) the NP score alone
ranks the truth at MRR 0.441 (random 0.289); on enveda-180 structures (synthetic-like, NP score median -1.45) it
is flat (0.169 vs 0.168).
"""
import glob
import gzip
import math
import os
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_FS = None
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def load(log=print):
    global _FS
    cands = []
    try:
        from rdkit.Chem import RDConfig
        cands.append(os.path.join(RDConfig.RDContribDir, "NP_Score", "publicnp.model.gz"))
    except Exception:
        pass
    cands += glob.glob("/usr/local/lib/python3*/*-packages/rdkit/Contrib/NP_Score/publicnp.model.gz")
    cands += glob.glob("/kaggle/input/**/publicnp.model.gz", recursive=True)
    cands += glob.glob("/usr/**/NP_Score/publicnp.model.gz", recursive=True)
    cands += glob.glob("/opt/**/NP_Score/publicnp.model.gz", recursive=True)
    for p in cands:
        if os.path.exists(p):
            _FS = pickle.load(gzip.open(p))
            log(f"NP model {p} ({len(_FS)} fragments)")
            return True
    raise FileNotFoundError("publicnp.model.gz not found")


def score_smiles(smi):
    """Ertl NP-likeness (same arithmetic as Contrib/NP_Score/npscorer.scoreMol); None when unparsable."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    bits = _GEN.GetSparseCountFingerprint(m).GetNonzeroElements()
    if not bits:
        return None
    score = sum(_FS.get(b, 0) for b in bits) / float(m.GetNumAtoms())
    if score > 4:
        score = 4. + math.log10(score - 4. + 1.)
    elif score < -4:
        score = -4. - math.log10(-score - 4. + 1.)
    return score


def rerank(smis, lam):
    """Order of indices by z(-log rank) + lam * z(NP score); unparsable candidates get the list's lowest score."""
    v = np.array([np.nan if x is None else x for x in map(score_smiles, smis)], np.float64)
    if np.all(np.isnan(v)):
        return list(range(len(smis)))
    v = np.where(np.isnan(v), np.nanmin(v), v)
    r = -np.log(np.arange(1, len(smis) + 1, dtype=np.float64))
    z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)  # noqa: E731
    return list(np.argsort(-(z(r) + lam * z(v)), kind="stable"))

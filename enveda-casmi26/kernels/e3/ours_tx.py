"""ours (yasunorim): test-set chemotype prior (transductive).

Natural products come in families: the hidden test (Enveda's natural products) holds many molecules that share a
scaffold with other test molecules. Each molecule's final top-K candidates get the mean of their top-3 Morgan r2
Tanimoto similarities to the other molecules' current rank-1 structures, and the list is re-ranked by
z(-log rank) + lam * z(that similarity).
On train (enveda-np-examples, 171 molecules, the truth among 49 same-formula train isomers, anchors = the other
molecules' truths) the similarity alone ranks the truth at MRR 0.533 (random 0.289); with 60 % of the anchors
replaced by a wrong same-formula isomer, 0.465. On enveda-180 (no families) it is flat (0.207 vs 0.189).
"""
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_C = {}


def fp(smi):
    if smi not in _C:
        m = Chem.MolFromSmiles(smi)
        _C[smi] = _GEN.GetFingerprint(m) if m is not None else None
    return _C[smi]


def anchors(lists):
    """{molecule: rank-1 fingerprint} from {molecule: [smiles, ...]}."""
    out = {}
    for m, l in lists.items():
        if l and fp(l[0]) is not None:
            out[m] = fp(l[0])
    return out


def score_list(mol, smis, anc, top=3):
    """Mean of the top-`top` Tanimoto similarities of each candidate to the other molecules' anchors (nan if none)."""
    A = [a for k, a in anc.items() if k != mol]
    out = []
    for s in smis:
        f = fp(s)
        if f is None or not A:
            out.append(np.nan); continue
        v = np.sort(np.asarray(DataStructs.BulkTanimotoSimilarity(f, A)))
        out.append(float(v[-top:].mean()))
    return np.array(out, np.float64)


def rerank(sc, lam):
    v = np.asarray(sc, np.float64)
    n = len(v)
    if n < 2 or np.all(np.isnan(v)):
        return list(range(n))
    v = np.where(np.isnan(v), np.nanmin(v), v)
    r = -np.log(np.arange(1, n + 1, dtype=np.float64))
    z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)  # noqa: E731
    return list(np.argsort(-(z(r) + lam * z(v)), kind="stable"))

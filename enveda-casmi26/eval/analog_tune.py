"""Re-score the analog channel from an analog.py dump with other settings.

Uses the saved hits (per query spectrum: top analogs with their hybrid
similarity), so nothing is searched again. Varies the fingerprint used for
candidate-analog similarity, the power on the spectral similarity, how many
analogs count, and max vs. sum-of-top-k aggregation. Library gate 0.95 as b2.

    python analog_tune.py scores_analog_150_coco_np.pkl
"""
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from common import DATA, mrr25

RDLogger.DisableLog("rdApp.*")
GENS = {
    "morgan2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint,
    "morgan1": rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048).GetFingerprint,
    "morgan3c": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096).GetCountFingerprint,
    "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=6, fpSize=4096).GetFingerprint,
    "apair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096).GetCountFingerprint,
    "maccs": MACCSkeys.GenMACCSKeys,
}
POWS = [1.0, 3.0, 6.0]
NS = [10, 30, 100]
AGG = ["max", "top3", "soft"]


def sim(f, fs):
    if isinstance(f, DataStructs.ExplicitBitVect):
        return np.array(DataStructs.BulkTanimotoSimilarity(f, fs))
    return np.array(DataStructs.BulkTanimotoSimilarity(f, fs))


def main():
    dump = [d for d in pickle.load(open(DATA + "/" + sys.argv[1], "rb")) if d[1] != 3]
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else list(GENS)
    S = pd.read_parquet(DATA + "/structures.parquet")
    coco = pd.read_parquet(DATA + "/coconut.parquet")
    smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
    mols = {}

    def mol(ik):
        if ik not in mols:
            s = smi.get(ik)
            mols[ik] = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return mols[ik]

    rows = []
    for fam in fams:
        g = GENS[fam]
        cache = {}

        def fp(ik):
            if ik not in cache:
                m = mol(ik)
                cache[ik] = g(m) if m is not None else None
            return cache[ik]

        for ik, cls, cands, lib, _, hits in dump:
            lib = np.asarray(lib)
            gate = np.where(lib >= 0.95, lib + 1.0, 0.0)
            cf = [fp(c) for c in cands]
            okc = [i for i, f in enumerate(cf) if f is not None]
            best = {}
            for sh in hits:
                for rank, (s, a) in enumerate(sh):
                    if s > best.get(a, (0, 99))[0]:
                        best[a] = (s, rank)
            items = sorted(((s, r, a) for a, (s, r) in best.items() if fp(a) is not None), key=lambda x: -x[0])
            T = np.zeros((len(items), len(cands)))
            for k, (_, _, a) in enumerate(items):
                if okc:
                    T[k, okc] = sim(fp(a), [cf[i] for i in okc])
            sv = np.array([x[0] for x in items])
            for p in POWS:
                for n in NS:
                    w = sv[:n] ** p
                    M = w[:, None] * T[:n] if len(w) else np.zeros((1, len(cands)))
                    for agg in AGG:
                        if agg == "max":
                            a = M.max(0)
                        elif agg == "top3":
                            a = np.sort(M, 0)[-3:].sum(0)
                        else:
                            a = (w[:, None] * T[:n] ** 4).sum(0) / (w.sum() + 1e-9) if len(w) else M.max(0)
                        sc = gate + a
                        rows.append((fam, p, n, agg, cls, mrr25([cands[i] for i in np.argsort(-sc, kind="stable")], ik)))
        r = pd.DataFrame(rows, columns=["fp", "pow", "n", "agg", "cls", "mrr"])
        t = r[r.fp == fam].pivot_table(index=["fp", "pow", "n", "agg"], columns="cls", values="mrr", aggfunc="mean").round(3)
        print(t.sort_values(4, ascending=False).head(8), flush=True)
    r = pd.DataFrame(rows, columns=["fp", "pow", "n", "agg", "cls", "mrr"])
    t = r.pivot_table(index=["fp", "pow", "n", "agg"], columns="cls", values="mrr", aggfunc="mean").round(3)
    print("=== best on class 4")
    print(t.sort_values(4, ascending=False).head(15))


if __name__ == "__main__":
    main()

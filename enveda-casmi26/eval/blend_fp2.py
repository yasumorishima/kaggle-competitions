"""Blend b3's analog score with fp2's fingerprint log-likelihood on class 4.

fp2dump (Kaggle) prints one line per class-4 molecule:
    FPLL <ik14> <cand>:<ll> <cand>:<ll> ...
and fp2all also "FPLLC ..." for the class 1-3 molecules (held out of its training), so with
its log every class is scored (per class and the mean over classes).
This joins them with the analog hits saved by `analog.py 150 coco np`, rebuilds b3's
channel (Morgan r3 counts, POW 2, 400 analogs, top-3 sum, library gate 0.8) and ranks by

    gate + analog + w * exp((ll - max ll) / T)

for a grid of w and T, plus FP alone and analog alone. MRR@25 on class 4.

    python blend_fp2.py [fp2dump_log.txt]
"""
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import DATA, mrr25

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
POW, N_KEEP, TOP_K, GATE = 2.0, 400, 3, 0.8
WS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
TS = [10.0, 25.0, 50.0, 100.0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA + "/fp2dump_log.txt"
    fpll = {}
    for line in open(path):
        if line.startswith(("FPLL ", "FPLLC ")):
            parts = line.split()
            fpll[parts[1]] = {c: float(v) for c, v in (x.rsplit(":", 1) for x in parts[2:])}
    dump = [d for d in pickle.load(open(DATA + "/scores_analog_150_coco_np.pkl", "rb")) ]
    S = pd.read_parquet(DATA + "/structures.parquet")
    coco = pd.read_parquet(DATA + "/coconut.parquet")
    smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
    cache = {}

    def fp(ik):
        if ik not in cache:
            s = smi.get(ik)
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            cache[ik] = GEN.GetCountFingerprint(m) if m is not None else None
        return cache[ik]

    rows, n_join = [], 0
    for ik, cls, cands, lib, _, hits in dump:
        if ik not in fpll:
            continue
        n_join += 1
        lib = np.asarray(lib)
        gate = np.where(lib >= GATE, lib + 1.0, 0.0)
        cf = [fp(c) for c in cands]
        okc = [i for i, f in enumerate(cf) if f is not None]
        best = {}
        for sh in hits:
            for s, a in sh:
                best[a] = max(best.get(a, 0.0), s)
        items = sorted(((s, a) for a, s in best.items() if fp(a) is not None), key=lambda x: -x[0])[:N_KEEP]
        M = np.zeros((max(len(items), 1), len(cands)))
        for k, (s, a) in enumerate(items):
            if okc:
                M[k, okc] = (s ** POW) * np.array(DataStructs.BulkTanimotoSimilarity(fp(a), [cf[i] for i in okc]))
        ana = np.sort(M, 0)[-TOP_K:].sum(0)
        ll = np.array([fpll[ik].get(c, -1e9) for c in cands])
        rank = lambda sc: [cands[i] for i in np.argsort(-sc, kind="stable")]  # noqa: E731
        rows.append(("analog (b3)", mrr25(rank(gate + ana), ik)))
        rows.append(("fp2 alone", mrr25(rank(ll), ik)))
        rows = [(r[0], r[1], cls) if len(r) == 2 else r for r in rows]
        for T in TS:
            z = np.exp((ll - ll.max()) / T)
            for w in WS[1:]:
                rows.append((f"T{T:g} w{w:g}", mrr25(rank(gate + ana + w * z), ik), cls))
        # rank fusion: reciprocal ranks of the two channels
        ra = np.argsort(np.argsort(-(gate + ana), kind="stable"))
        rf = np.argsort(np.argsort(-ll, kind="stable"))
        for k in (5, 20, 60):
            rows.append((f"rrf k{k}", mrr25(rank(1 / (k + ra) + 1 / (k + rf)), ik), cls))
    r = pd.DataFrame(rows, columns=["method", "mrr", "cls"]).pivot_table("mrr", "method", "cls")
    r["mean"] = r.mean(1)
    print(f"molecules joined: {n_join}")
    print(r.sort_values("mean", ascending=False).round(3).head(25).to_string())


if __name__ == "__main__":
    main()

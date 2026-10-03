"""Popularity prior (own PubChem table, datasets/pubchem-pop) on top of the LB-best b5 fusion (tp0.7_0.5); copied from blend_trainpen.py.

The hidden test's class 2 (structure in PubChem/COCONUT, no public spectra) and class 3
(not in PubChem) answers are never train structures that carry library spectra. So a
candidate that has spectra in the library, but none of them match (lib below the gate),
is less likely to be the answer than a candidate without spectra. This scales the
non-gated part of such candidates by F and reports MRR@25 per class, on top of the
LB-best b5 fusion (gate 0.8 + analog + 0.2 * exp((ll - max ll) / 50)).

    python blend_trainpen.py [fp2all_log.txt]

"has spectra" here = a train structure that the split did not strip (class 2/3 answers
and the class-4 natural products lose all their spectra; class 1 keeps other libraries').
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
POW, N_KEEP, TOP_K, GATE, T, W = 2.0, 400, 3, 0.8, 50.0, 0.2
FS = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
LIB_FLOORS = [0.0, 0.3, 0.5]   # only demote when the candidate's best own-spectrum match is below this ... or any
LAS = [(0.3, 0.5), (1.0, 0.5), (0.3, 0.65), (1.0, 0.65)]   # H3: + A * lib for L <= lib < gate, on top of F 0.5 below 0.5


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA + "/fp2all_log.txt"
    fpll = {}
    for line in open(path):
        if line.startswith(("FPLL ", "FPLLC ")):
            parts = line.split()
            fpll[parts[1]] = {c: float(v) for c, v in (x.rsplit(":", 1) for x in parts[2:])}
    dump = pickle.load(open(DATA + "/scores_analog_150_coco_np.pkl", "rb"))
    S = pd.read_parquet(DATA + "/structures.parquet")
    coco = pd.read_parquet(DATA + "/coconut.parquet")
    smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
    meta = pd.read_parquet(DATA + "/train_meta.parquet", columns=["inchikey14", "ingest_lib"])
    split = pd.read_parquet(DATA + "/split.parquet")
    stripped = set(split.inchikey14[split.cls.isin([2, 3])]) | set(meta.inchikey14[meta.ingest_lib == "enveda-np-examples"])
    has_spec = set(meta.inchikey14) - stripped
    del meta
    cache = {}
    pop = pd.read_parquet(DATA + "/pubchem_pop.parquet")
    pop = dict(zip(pop.inchikey14, np.log1p(pop.n_pmid.values)))
    cov = []

    def fp(ik):
        if ik not in cache:
            s = smi.get(ik)
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            cache[ik] = GEN.GetCountFingerprint(m) if m is not None else None
        return cache[ik]

    rows, n_join, share = [], 0, []
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
        base = ana + W * np.exp((ll - ll.max()) / T)
        hs = np.array([c in has_spec for c in cands])
        share.append((cls, hs.mean(), ik in has_spec))
        rank = lambda sc: [cands[i] for i in np.argsort(-sc, kind="stable")]  # noqa: E731
        tp = np.where(hs & (lib < 0.5), 0.7 * base, base)
        pv = np.array([pop.get(c, 0.0) for c in cands])
        cov.append((cls, (pv > 0).mean(), pop.get(ik, 0.0) > 0, pv.max(), pop.get(ik, 0.0)))
        for mu in [0, 0.02, 0.05, 0.1, 0.2, 0.4]:
            rows.append((f"tp0.7_0.5+pop{mu:g}", mrr25(rank(gate + tp + mu * pv), ik), cls))
        for mu in [0.1, 0.3]:   # relative: scale by the molecule's best non-gate score
            rows.append((f"tp0.7_0.5+poprel{mu:g}", mrr25(rank(gate + tp + mu * tp.max() * pv / max(pv.max(), 1e-9)), ik), cls))
    r = pd.DataFrame(rows, columns=["method", "mrr", "cls"]).pivot_table("mrr", "method", "cls")
    r["mean"] = r.mean(1)
    print(f"molecules joined: {n_join}")
    sh = pd.DataFrame(share, columns=["cls", "cand_with_spec", "answer_with_spec"]).groupby("cls").mean()
    print(sh.round(3).to_string())
    print(r.round(3).to_string())
    print(pd.DataFrame(cov, columns=["cls", "cand_with_pop", "answer_with_pop", "max_logpop", "answer_logpop"]).groupby("cls").mean().round(3).to_string())


if __name__ == "__main__":
    main()

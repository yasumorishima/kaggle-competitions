"""Enamine tier as extra candidates: where do the answers rank, and what do fixed slots cost?

The hidden test's class 2 (in PubChem, no public spectra) looks like Enamine screening
compounds (92% of enveda-180 is in PubChem's Enamine deposit). Proxy: the split's class 3
(enveda-180 structures stripped from the library AND the candidate pool) whose structure is
in the Enamine tier. Enamine candidates in the +-10 ppm window (around the answer's mass) are
scored by the analog channel only (sum of the top 3 sim^2 * Tanimoto) -- fp2 weights are not
available in the cloud, the kernel adds them.

Reports, per class:
  - in-window rate of the answer in the Enamine tier, Enamine window size
  - MRR of the answer among the Enamine-only candidates (analog ranking)
  - MRR@25 of the existing ranking (gate + analog + fp2) with the top-k Enamine-only
    candidates put at fixed slots, only when the best library match is below the gate

    python enamine_slot.py [fp2all_log.txt]
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
PPM = 10.0
SLOTS = {"none": [], "s25": [24], "s20-25": [19, 21, 23, 24], "s10-25": [9, 14, 19, 24],
         "s5-25": [4, 9, 14, 19, 24], "s2-6": [1, 3, 5], "s2-10": [1, 3, 5, 7, 9], "s3-11": [2, 5, 8, 10]}


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
    E = pd.read_parquet(DATA + "/enamine_tier.parquet").sort_values("fM").reset_index(drop=True)
    emass, eik, esmi = E.fM.values, E.inchikey14.values, E.smiles.values
    epos = {k: i for i, k in enumerate(eik)}
    meta = pd.read_parquet(DATA + "/train_meta.parquet", columns=["inchikey14", "ingest_lib"])
    split = pd.read_parquet(DATA + "/split.parquet")
    stripped = set(split.inchikey14[split.cls.isin([2, 3])]) | set(meta.inchikey14[meta.ingest_lib == "enveda-np-examples"])
    has_spec = set(meta.inchikey14) - stripped
    del meta
    cache = {}

    def fp(ik, s=None):
        if ik not in cache:
            s = s if s is not None else smi.get(ik)
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            cache[ik] = GEN.GetCountFingerprint(m) if m is not None else None
        return cache[ik]

    rows, stats = [], []
    for n, (ik, cls, cands, lib, _, hits) in enumerate(dump):
        lib = np.asarray(lib)
        # mass of the answer: from the Enamine table, else from the structure
        if ik in epos:
            m0 = emass[epos[ik]]
        else:
            mol = Chem.MolFromSmiles(smi.get(ik, "")) if isinstance(smi.get(ik), str) else None
            if mol is None:
                continue
            from rdkit.Chem.Descriptors import ExactMolWt
            m0 = ExactMolWt(mol)
        lo, hi = np.searchsorted(emass, [m0 * (1 - PPM * 1e-6), m0 * (1 + PPM * 1e-6)])
        cset = set(cands)
        ecand = [(eik[i], esmi[i]) for i in range(lo, hi) if eik[i] not in cset]
        best = {}
        for sh in hits:
            for s, a in sh:
                best[a] = max(best.get(a, 0.0), s)
        items = sorted(((s, a) for a, s in best.items() if fp(a) is not None), key=lambda x: -x[0])[:N_KEEP]
        afp = [fp(a) for _, a in items]
        aw = np.array([s ** POW for s, _ in items])

        def analog(fps):
            ok = [i for i, f in enumerate(fps) if f is not None]
            M = np.zeros((max(len(items), 1), len(fps)))
            for k in range(len(items)):
                if ok:
                    M[k, ok] = aw[k] * np.array(DataStructs.BulkTanimotoSimilarity(afp[k], [fps[i] for i in ok]))
            return np.sort(M, 0)[-TOP_K:].sum(0)

        # existing ranking: gate + analog + fp2 (when available), tp0.7_0.5 demotion
        ana = analog([fp(c) for c in cands])
        ll = np.array([fpll.get(ik, {}).get(c, -1e9) for c in cands]) if ik in fpll else None
        base = ana + (W * np.exp((ll - ll.max()) / T) if ll is not None else 0.0)
        hs = np.array([c in has_spec for c in cands])
        base = np.where(hs & (lib < 0.5), 0.7 * base, base)
        sc = np.where(lib >= GATE, 1000.0 + lib, 0.0) + base
        ranked = [cands[i] for i in np.argsort(-sc, kind="stable")]
        # Enamine-only candidates by analog
        e_ana = analog([fp(k, s) for k, s in ecand]) if ecand else np.zeros(0)
        e_rank = [ecand[i][0] for i in np.argsort(-e_ana, kind="stable")]
        in_e = ik in set(e_rank)
        e_mrr = mrr25(e_rank, ik)
        gated = bool((lib >= GATE).any())
        stats.append((cls, ik in epos, in_e, len(ecand), len(cands), e_mrr, gated, ik in cset))
        for name, sl in SLOTS.items():
            for only_weak in (True, False):
                if not sl and not only_weak:
                    continue
                out = list(ranked)
                if sl and not (only_weak and gated):
                    ins = iter(e_rank)
                    for p in sl:
                        k = next(ins, None)
                        if k is None:
                            break
                        out.insert(p, k)
                rows.append((name + ("" if only_weak else "/all"), mrr25(out, ik), cls))
        if n % 50 == 0:
            print(n, file=sys.stderr, flush=True)
    st = pd.DataFrame(stats, columns=["cls", "in_tier", "in_window", "n_enam", "n_pool", "e_mrr", "gated", "in_pool"])
    print(st.groupby("cls").agg(mols=("cls", "size"), in_tier=("in_tier", "mean"), in_window=("in_window", "mean"),
                                n_enam=("n_enam", "median"), n_pool=("n_pool", "median"), e_mrr=("e_mrr", "mean"),
                                gated=("gated", "mean"), in_pool=("in_pool", "mean")).round(3).to_string())
    r = pd.DataFrame(rows, columns=["method", "mrr", "cls"]).pivot_table("mrr", "method", "cls")
    print(r.round(3).to_string())
    st.to_csv(DATA + "/enamine_slot_stats.csv", index=False)


if __name__ == "__main__":
    main()

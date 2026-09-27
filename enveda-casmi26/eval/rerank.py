"""Plan step 2: a learned re-ranker over the analog and fp2 channels (class 4).

Features per candidate (all computable at test time, none of them knows where the truth
came from -- `intrain` and `lib` leaked on class 4 and are left out):
  ana3, ana1           top-3 sum / best of sim**2 * Tanimoto over <=400 analogs (b3's channel)
  ana3_rel, ana_rank   the same relative to the molecule's best candidate
  ll, ll_rank          fp2 log-likelihood (relative to the best) and its rank
  n                    candidates in the window
The library gate (lib >= 0.8 first) stays outside the model, as on the LB.

    python rerank.py            # grouped 5-fold CV x 3, then fits on all and writes
                                # ../kernels/b5/lgb_model.txt
"""
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import DATA, mrr25

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
FEATS = ["ana3", "ana1", "ana3_rel", "ana_rank", "ll", "ll_rank", "n"]
PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30, verbose=-1)


def features(cands, ana_matrix, ll):
    """ana_matrix: (analogs, candidates) of sim**2 * Tanimoto; ll: fp2 log-likelihood per candidate."""
    Ms = np.sort(ana_matrix, 0)
    ana3, ana1 = Ms[-3:].sum(0), Ms[-1]
    ll = np.asarray(ll, float)
    good = ll > -1e8
    ll = np.where(good, ll, (ll[good].min() - 50) if good.any() else 0.0)
    return pd.DataFrame({"cand": cands, "ana3": ana3, "ana1": ana1, "ana3_rel": ana3 - ana3.max(),
                         "ana_rank": np.argsort(np.argsort(-ana3)), "ll": ll - ll.max(),
                         "ll_rank": np.argsort(np.argsort(-ll)), "n": len(cands)})


def main():
    fpll = {}
    for line in open(DATA + "/fp2dump_log.txt"):
        if line.startswith("FPLL "):
            p = line.split()
            fpll[p[1]] = {c: float(v) for c, v in (x.rsplit(":", 1) for x in p[2:])}
    dump = [d for d in pickle.load(open(DATA + "/scores_analog_150_coco_np.pkl", "rb")) if d[1] == 4]
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

    rows = []
    for g, (ik, _, cands, _, _, hits) in enumerate(dump):
        if ik not in fpll:
            continue
        cf = [fp(c) for c in cands]
        okc = [i for i, f in enumerate(cf) if f is not None]
        best = {}
        for sh in hits:
            for s, a in sh:
                best[a] = max(best.get(a, 0.0), s)
        items = sorted(((s, a) for a, s in best.items() if fp(a) is not None), key=lambda x: -x[0])[:400]
        M = np.zeros((max(len(items), 1), len(cands)))
        for k, (s, a) in enumerate(items):
            if okc:
                M[k, okc] = (s ** 2) * np.array(DataStructs.BulkTanimotoSimilarity(fp(a), [cf[i] for i in okc]))
        f = features(cands, M, [fpll[ik].get(c, -1e9) for c in cands])
        rows.append(f.assign(g=g, ik=ik, y=(f.cand == ik).astype(int)))
    D = pd.concat(rows, ignore_index=True).sort_values("g", kind="stable")

    for seed in range(3):
        gids = D.g.unique()
        np.random.default_rng(seed).shuffle(gids)
        res = {"lgb": [], "analog": [], "fp2": []}
        for fold in np.array_split(gids, 5):
            te, tr = D[D.g.isin(fold)], D[~D.g.isin(fold)]
            m = lgb.LGBMRanker(**PARAMS).fit(tr[FEATS], tr.y, group=tr.groupby("g", sort=False).size().values)
            te = te.assign(p=m.predict(te[FEATS]))
            for _, x in te.groupby("g"):
                ik = x.ik.iloc[0]
                for name, col in (("lgb", "p"), ("analog", "ana3"), ("fp2", "ll")):
                    res[name].append(mrr25(list(x.sort_values(col, ascending=False, kind="stable").cand), ik))
        print("CV seed", seed, {k: round(float(np.mean(v)), 3) for k, v in res.items()})

    m = lgb.LGBMRanker(**PARAMS).fit(D[FEATS], D.y, group=D.groupby("g", sort=False).size().values)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "b5", "lgb_model.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m.booster_.save_model(out)
    print("wrote", os.path.normpath(out), dict(zip(FEATS, m.feature_importances_)))


if __name__ == "__main__":
    main()

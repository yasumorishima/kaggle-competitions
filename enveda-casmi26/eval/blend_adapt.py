"""fp2 fusion variants on every class with fp2all (class 1-3 held out): plain, x(1-s1), x(1-s1)^2, rel (W x best analog).

    cd eval && python blend_adapt.py      # needs DATA/fp2all_log.txt
"""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from blend_fp2 import *  # noqa
from common import DATA, mrr25
fpll = {}
for line in open(DATA + "/fp2all_log.txt"):
    if line.startswith(("FPLL ", "FPLLC ")):
        p = line.split(); fpll[p[1]] = {c: float(v) for c, v in (x.rsplit(":", 1) for x in p[2:])}
dump = pickle.load(open(DATA + "/scores_analog_150_coco_np.pkl", "rb"))
S = pd.read_parquet(DATA + "/structures.parquet"); coco = pd.read_parquet(DATA + "/coconut.parquet")
smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
cache = {}
def fp(ik):
    if ik not in cache:
        s = smi.get(ik); m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        cache[ik] = GEN.GetCountFingerprint(m) if m is not None else None
    return cache[ik]
rows = []
for ik, cls, cands, lib, _, hits in dump:
    if ik not in fpll: continue
    lib = np.asarray(lib); gate = np.where(lib >= GATE, lib + 1.0, 0.0)
    cf = [fp(c) for c in cands]; okc = [i for i, f in enumerate(cf) if f is not None]
    best = {}
    for sh in hits:
        for s, a in sh: best[a] = max(best.get(a, 0.0), s)
    items = sorted(((s, a) for a, s in best.items() if fp(a) is not None), key=lambda x: -x[0])[:N_KEEP]
    M = np.zeros((max(len(items), 1), len(cands)))
    for k, (s, a) in enumerate(items):
        if okc: M[k, okc] = (s ** POW) * np.array(DataStructs.BulkTanimotoSimilarity(fp(a), [cf[i] for i in okc]))
    ana = np.sort(M, 0)[-TOP_K:].sum(0)
    ll = np.array([fpll[ik].get(c, -1e9) for c in cands])
    rank = lambda sc: [cands[i] for i in np.argsort(-sc, kind="stable")]
    amax = ana.max() if len(ana) else 0
    s1 = items[0][0] if items else 0.0
    for T in (25, 50, 100):
        z = np.exp((ll - ll.max()) / T)
        for w in (0.1, 0.2, 0.3, 0.5):
            rows.append((f"T{T} w{w}", mrr25(rank(gate + ana + w * z), ik), cls))
            rows.append((f"T{T} w{w} x(1-s1)", mrr25(rank(gate + ana + w * (1 - s1) * z), ik), cls))
            rows.append((f"T{T} w{w} x(1-s1)^2", mrr25(rank(gate + ana + w * (1 - s1) ** 2 * z), ik), cls))
            rows.append((f"T{T} w{w} rel", mrr25(rank(gate + ana + w * max(amax, 1e-9) * z), ik), cls))
    rows.append(("analog", mrr25(rank(gate + ana), ik), cls))
r = pd.DataFrame(rows, columns=["m", "mrr", "cls"]).pivot_table("mrr", "m", "cls")
r["mean"] = r.mean(1)
print(r.sort_values(2, ascending=False).round(3).head(30).to_string())
print(r.loc[["analog"]].round(3))

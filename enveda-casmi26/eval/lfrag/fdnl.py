import sys, random, numpy as np, pyarrow.parquet as pq, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/home/user/kaggle-competitions/enveda-casmi26/kernels/e3"); import ours_frag as F
sys.path.insert(0, "pub"); import frag_rescore_451 as R5
import ours_nl as NL
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OFF = [-9, -5, -3, 3, 5, 9]
meta = pq.read_table("enveda/train_meta.parquet", columns=["ingest_lib","inchikey14","normalized_smiles","molecular_formula"]).to_pandas()
pool = meta.drop_duplicates("inchikey14"); byf = pool.groupby("molecular_formula")
rows = []
for b in pq.ParquetFile("enveda/train.parquet").iter_batches(batch_size=200000, columns=["ingest_lib","inchikey14","normalized_smiles","molecular_formula","precursor_mz","adduct","ms2_mzs","ms2_normalized_intensities"]):
    t = b.to_pandas(); rows.append(t[t.ingest_lib == "enveda-np-examples"])
t = pd.concat(rows)
jobs = []; rng = random.Random(0)
for ik, g in t.groupby("inchikey14"):
    f = g.molecular_formula.iloc[0]
    if f not in byf.groups: continue
    c = byf.get_group(f); c = c[c.inchikey14 != ik]
    if len(c) < 3: continue
    cands = list(c.normalized_smiles.sample(min(NC - 1, len(c)), random_state=0)) + [g.normalized_smiles.iloc[0]]
    rng.shuffle(cands); truth = cands.index(g.normalized_smiles.iloc[0])
    sp = [(np.c_[np.asarray(r.ms2_mzs), np.asarray(r.ms2_normalized_intensities)], r.precursor_mz, F.polarity_of(r.adduct), r.adduct) for r in g.head(3).itertuples()]
    jobs.append((cands, truth, sp))
def run(j):
    cands, truth, sp = j
    mk = lambda o: [dict(mz=pk[:, 0] + o, it=pk[:, 1], adduct=a, prec=prec + o, mode=pol) for pk, prec, pol, a in sp]
    raw = np.array([v if v is not None else 0.0 for v in R5.score_molecule(mk(0), cands, cache={})])
    seen = NL.observed([(pk, prec) for pk, prec, pol, a in sp])
    nl = np.array(NL.score_list(cands, seen), float)
    return truth, raw, nl
def rr(sc, t): return 1 / (1 + (sc > sc[t]).sum() + 0.5 * ((sc == sc[t]).sum() - 1))
with Pool() as p: R = p.map(run, jobs)
z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
res = {k: [] for k in ("frag", "nl", "frag+nl0.5", "frag+nl0.25", "frag+nl1")}
nseen = 0
for t_, raw, nl in R:
    nseen += nl.std() > 0
    res["frag"].append(rr(raw, t_)); res["nl"].append(rr(nl, t_))
    for l in (0.25, 0.5, 1):
        res[f"frag+nl{l}"].append(rr(z(raw) + l * nl, t_))
print("molecules", len(R), "nl informative", nseen)
for k, v in res.items(): print(k, "MRR %.3f" % np.mean(v))

import sys, random, numpy as np, pyarrow.parquet as pq, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/home/user/kaggle-competitions/enveda-casmi26/kernels/e3"); import ours_frag as F
sys.path.insert(0, "pub"); import frag_rescore_451 as R5
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
    ad = lambda pol: "[M+H]+" if pol > 0 else "[M-H]-"
    mk = lambda o: [dict(mz=pk[:, 0] + o, it=pk[:, 1], adduct=a, prec=prec + o, mode=pol) for pk, prec, pol, a in sp]
    cache = {}
    raw = np.array([v if v is not None else 0.0 for v in R5.score_molecule(mk(0), cands, cache=cache)])
    dec = np.mean([[v if v is not None else 0.0 for v in R5.score_molecule(mk(o), cands, cache=cache)] for o in OFF], axis=0)
    return truth, raw, dec
def rr(sc, t): return 1 / (1 + (sc > sc[t]).sum() + 0.5 * ((sc == sc[t]).sum() - 1))
with Pool() as p: R = p.map(run, jobs)
res = {k: [] for k in ("raw", "dec", "corr", "ratio", "rand")}
for t_, raw, dec in R:
    res["raw"].append(rr(raw, t_)); res["dec"].append(rr(-dec, t_))
    res["corr"].append(rr(raw - dec, t_)); res["ratio"].append(rr((raw + 1e-3) / (dec + 1e-3), t_))
    res["rand"].append(np.mean(1 / np.arange(1, len(raw) + 1)))
print("molecules", len(R), "cands", NC)
for k, v in res.items(): print(k, "MRR %.3f" % np.mean(v))

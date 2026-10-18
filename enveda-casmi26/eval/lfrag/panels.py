import sys, random, numpy as np, pyarrow.parquet as pq, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/tmp/claude-0/-home-user-kaggle-competitions/e14dc4c1-b1cc-5b07-bb7b-e4b5776308f1/scratchpad/lf"); import feat
NC = 20
E = "/tmp/claude-0/-home-user-kaggle-competitions/e14dc4c1-b1cc-5b07-bb7b-e4b5776308f1/scratchpad/enveda/"
meta = pq.read_table(E + "train_meta.parquet", columns=["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula"]).to_pandas()
pool = meta.drop_duplicates("inchikey14"); byf = {f: g for f, g in pool.groupby("molecular_formula")}
rng = random.Random(0)
def pick(lib, n):
    ks = meta[meta.ingest_lib == lib].drop_duplicates("inchikey14")
    ks = ks[ks.molecular_formula.map(lambda f: f in byf and len(byf[f]) >= 4)]
    return set(ks.inchikey14.sample(min(n, len(ks)), random_state=1))
want = {"np": pick("enveda-np-examples", 10**6), "e180": pick("enveda-180", 3500)}
oth = set()
for lib in ("gnps", "mona", "massbank", "riken", "msdial", "pluskal_ms2"):
    oth |= pick(lib, 250)
want["oth"] = oth - want["np"] - want["e180"]
libof = {"np": "enveda-np-examples", "e180": "enveda-180"}
allk = set().union(*want.values())
rows = []
for b in pq.ParquetFile(E + "train.parquet").iter_batches(batch_size=200000, columns=["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]):
    t = b.to_pandas(); rows.append(t[t.inchikey14.isin(allk)])
t = pd.concat(rows)
jobs = []
for name, ks in want.items():
    sub = t[t.inchikey14.isin(ks)]
    if name in libof: sub = sub[sub.ingest_lib == libof[name]]
    for ik, g in sub.groupby("inchikey14"):
        c = byf[g.molecular_formula.iloc[0]]; c = c[c.inchikey14 != ik]
        cands = list(c.normalized_smiles.sample(min(NC - 1, len(c)), random_state=0)) + [g.normalized_smiles.iloc[0]]
        rng.shuffle(cands)
        sp = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float), adduct=r.adduct,
                   prec=float(r.precursor_mz), mode=-1 if str(r.adduct).endswith("-") else 1) for r in g.head(3).itertuples()]
        jobs.append((name, cands, cands.index(g.normalized_smiles.iloc[0]), sp))
def run(j):
    name, cands, truth, sp = j; cache = {}
    F = [feat.cand_feats(s, sp, cache) for s in cands]
    ok = [f is not None for f in F]
    F = np.array([f if f is not None else np.zeros(feat.NF) for f in F])
    return name, F, truth, np.array(ok)
if __name__ == "__main__":
    print({k: len(v) for k, v in want.items()}, "jobs", len(jobs), flush=True)
    with Pool() as p: R = p.map(run, jobs, chunksize=8)
    import pickle; pickle.dump(R, open(E + "../lf/panels.pkl", "wb"))
    print("done", len(R))

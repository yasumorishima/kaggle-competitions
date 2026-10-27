"""Leak-free ranker check on the rows of an rkdump val run (build.py val=N valfold=k rkdump=1).

Run by the enveda workflow's "output" action on the downloaded kernel output (rk_rows.parquet comes from train
molecules, so it stays out of the repo). Prints numbers only.
Compares, on the same held-out molecules, MRR@25 of
  base      : the public ranker's score (fitted on rows where FPNet and the library had seen the molecule)
  base+pop  : the base list's popularity re-rank, z(score) + 0.15 pop (what the pipeline does next)
  new       : LightGBM lambdarank fitted on these rows, 5-fold CV by molecule (features = the ranker's own inputs)
  new+base  : the same with the base score as one more feature
  new+pop   : the same with pop and lib_max added
usage: python rk_report.py <dir with rk_rows.parquet>
"""
import os, sys
import numpy as np, pandas as pd
import lightgbm as lgb

d = pd.read_parquet(os.path.join(sys.argv[1], "rk_rows.parquet"))
d = d.drop_duplicates(["molecule_id", "key"]).reset_index(drop=True)
mols = np.array(sorted(d.molecule_id.unique()))
cls = d.groupby("molecule_id").cls.first()
meta = {"molecule_id", "key", "label", "base_score", "m_lib_max", "m_pid", "m_n_cand", "m_pop", "cls"}
F = [c for c in d.columns if c not in meta]
print("rows", len(d), "molecules", len(mols), "truth in rows", int(d.groupby("molecule_id").label.max().sum()),
      "features", len(F), "| c1", int((cls == "c1").sum()), "c2", int((cls == "c2").sum()))
d["pop0"] = d["m_pop"].fillna(0.0)
g = d.groupby("molecule_id").base_score
d["z_base"] = (d.base_score - g.transform("mean")) / (g.transform("std") + 1e-9)


def mrr(score):
    s = d.assign(s=score).sort_values(["molecule_id", "s"], ascending=[True, False], kind="stable")
    s["r"] = s.groupby("molecule_id").cumcount() + 1
    t = s[s.label == 1].groupby("molecule_id").r.min().reindex(mols)
    rr = np.where(t.values <= 25, 1.0 / t.values, 0.0)
    rr = np.nan_to_num(rr)
    out = {"all": rr.mean(), "top1": float(np.mean(t.values == 1))}
    for c in ("c1", "c2"):
        m = (cls.reindex(mols) == c).values
        out[c] = rr[m].mean() if m.any() else float("nan")
    return out, pd.Series(rr, index=mols)


def cv(cols, seed=0, rounds=400):
    rng = np.random.RandomState(seed)
    fold = pd.Series(rng.permutation(len(mols)) % 5, index=mols)
    pred = np.zeros(len(d))
    for k in range(5):
        te = d.molecule_id.map(fold).values == k
        tr = ~te
        dt = d[tr].sort_values("molecule_id", kind="stable")
        grp = dt.groupby("molecule_id", sort=True).size().values
        m = lgb.train(dict(objective="lambdarank", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
                           bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambdarank_truncation_level=25,
                           verbose=-1, seed=seed), lgb.Dataset(dt[cols].values.astype(np.float32), dt.label.values, group=grp),
                      num_boost_round=rounds)
        pred[te] = m.predict(d.loc[te, cols].values.astype(np.float32))
    return pred


res = {}
res["base"], rb = mrr(d.base_score.values)
res["base+pop"], _ = mrr(d.z_base.values + 0.15 * d.pop0.values)
for name, cols in (("new", F), ("new+base", F + ["base_score"]), ("new+pop", F + ["pop0"]),
                   ("new+base+pop", F + ["base_score", "pop0"])):
    p = np.mean([cv(cols, seed=s) for s in (0, 1)], axis=0)
    res[name], rn = mrr(p)
    gp = d.assign(p=p).groupby("molecule_id").p
    zp = (p - gp.transform("mean").values) / (gp.transform("std").values + 1e-9)
    res[name + " >pop"], rnp = mrr(zp + 0.15 * d.pop0.values)
    w, l = int((rnp > rb + 1e-9).sum()), int((rnp < rb - 1e-9).sum())
    print(f"{name:14s} vs base (after pop): molecules up {w}, down {l}", flush=True)
for k, v in res.items():
    print(f"RK {k:20s} MRR@25 all {v['all']:.4f}  c1 {v['c1']:.4f}  c2 {v['c2']:.4f}  top1 {v['top1']:.4f}")
# which inputs the new ranker leans on (one fit on all rows)
dt = d.sort_values("molecule_id", kind="stable")
m = lgb.train(dict(objective="lambdarank", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, bagging_fraction=0.8,
                   bagging_freq=1, feature_fraction=0.7, verbose=-1, seed=0),
              lgb.Dataset(dt[F].values.astype(np.float32), dt.label.values, group=dt.groupby("molecule_id", sort=True).size().values),
              num_boost_round=400)
imp = pd.Series(m.feature_importance("gain"), index=F).sort_values(ascending=False)
print("top gain features:", ", ".join(f"{k} {v / imp.sum():.3f}" for k, v in imp.head(15).items()))

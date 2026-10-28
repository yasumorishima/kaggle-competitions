"""Why the leak-free ranker refit lost on the LB (e3f465rk-1 0.415, rm-1 0.424 vs 0.467) although the CV said +0.05.

Reads rk_rows.parquet of e3rk0 (800 fold-0 molecules held out of train; c1 keeps other-library spectra, c2 none).
Prints numbers only:
  1. per feature: MRR@25 when candidates are ranked by that feature alone (best of both directions), for c1 and c2.
     A learned feature that ranks the truth first on c2 (no spectrum of the molecule anywhere in the library) is a
     sign that its model saw the molecule in training - on the hidden test it cannot.
  2. the same grouped by feature prefix.
  3. the refit without every feature whose single-feature c2 MRR is above a cut (0.5 / 0.3), CV by molecule.
usage: python rk_diag.py <dir with rk_rows.parquet>
"""
import os, sys
import numpy as np, pandas as pd
import lightgbm as lgb

d = pd.read_parquet(os.path.join(sys.argv[1], "rk_rows.parquet"))
d = d.drop_duplicates(["molecule_id", "key"]).reset_index(drop=True)
d = d.sort_values("molecule_id", kind="stable").reset_index(drop=True)
mols = np.array(sorted(d.molecule_id.unique()))
cls = d.groupby("molecule_id").cls.first().reindex(mols)
meta = {"molecule_id", "key", "label", "base_score", "m_lib_max", "m_pid", "m_n_cand", "m_pop", "cls"}
F = [c for c in d.columns if c not in meta]
print("rows", len(d), "molecules", len(mols), "features", len(F), "c1", int((cls == "c1").sum()), "c2", int((cls == "c2").sum()))
print("candidates per molecule: median", int(d.groupby("molecule_id").size().median()))
mid = d.molecule_id.values
start = np.r_[0, np.flatnonzero(mid[1:] != mid[:-1]) + 1, len(d)]
lab = d.label.values


def rr_of(score):
    out = np.zeros(len(mols))
    for j in range(len(mols)):
        a, b = start[j], start[j + 1]
        s, l = score[a:b], lab[a:b]
        if l.max() < 1:
            continue
        t = s[l == 1].max()
        r = 1 + int((s > t).sum()) + 0.5 * int((s == t).sum() - 1)   # ties count half
        out[j] = 1.0 / r if r <= 25 else 0.0
    return out


c1m, c2m = (cls == "c1").values, (cls == "c2").values
rows = []
for f in F + ["base_score"]:
    x = d[f].astype(float).fillna(-1e9).values
    best = None
    for sgn in (1, -1):
        rr = rr_of(sgn * x)
        cand = (rr[c2m].mean(), rr[c1m].mean(), sgn)
        if best is None or cand[0] + cand[1] > best[0] + best[1]:
            best = cand
    rows.append((f, best[0], best[1], best[2], float(np.mean(d[f].notna()))))
t = pd.DataFrame(rows, columns=["feature", "c2", "c1", "sign", "nonnull"]).sort_values("c2", ascending=False)
pd.set_option("display.width", 250)
print("single-feature MRR@25 (top 45 by c2):")
print(t.head(45).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
t["group"] = t.feature.str.split("__").str[0]
g = t.groupby("group").agg(n=("feature", "size"), c2_max=("c2", "max"), c1_max=("c1", "max"))
print("by group:\n" + g.sort_values("c2_max", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))


def cv(cols, seed=0, rounds=400):
    rng = np.random.RandomState(seed)
    fold = pd.Series(rng.permutation(len(mols)) % 5, index=mols)
    pred = np.zeros(len(d))
    for k in range(5):
        te = d.molecule_id.map(fold).values == k
        dt = d[~te]
        m = lgb.train(dict(objective="lambdarank", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
                           bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambdarank_truncation_level=25,
                           verbose=-1, seed=seed),
                      lgb.Dataset(dt[cols].values.astype(np.float32), dt.label.values,
                                  group=dt.groupby("molecule_id", sort=True).size().values), num_boost_round=rounds)
        pred[te] = m.predict(d.loc[te, cols].values.astype(np.float32))
    return pred


rb = rr_of(d.base_score.values)
print(f"base ranker: all {rb.mean():.3f} c1 {rb[c1m].mean():.3f} c2 {rb[c2m].mean():.3f}")
for cut in (0.5, 0.3):
    keep = [f for f in F if t.set_index("feature").loc[f, "c2"] <= cut]
    rr = rr_of(cv(keep))
    print(f"refit without single-feature c2 > {cut}: {len(keep)} features, all {rr.mean():.3f} c1 {rr[c1m].mean():.3f} "
          f"c2 {rr[c2m].mean():.3f}; up {int((rr > rb + 1e-9).sum())} down {int((rr < rb - 1e-9).sum())}", flush=True)

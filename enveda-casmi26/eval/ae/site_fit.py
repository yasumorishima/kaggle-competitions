"""ae v1 fit (ours, 11-07): learned pick among analog-edit products (rows from site.py), against the ae v0 vote rule.

Per molecule (gated, with products), v0 picks by (vote_sum desc, fr desc) and inserts when n_new <= 4 or votes >= 2.
v1: LightGBM binary on the product rows, 5-fold by molecule; pick = argmax p; insert when p >= threshold.
Reports, over the gated molecules with products: hit rate of the pick (no insertion rule), and for insertion rules the
inserted count, hits and the expected MRR change per 1,000 such molecules (+0.5 per hit at rank 2, about -0.025 per
insertion for answers further down).
Usage: python site_fit.py ROWS_PARQUET [ROWS_PARQUET ...] [--save MODEL_TXT]
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 3)[0] + "/kernels/e3")
import ours_aesite   # noqa: E402

import os
PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("AE_LR", 0.03)), num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
ROUNDS = int(os.environ.get("AE_ROUNDS", 400))   # 11-07: the kernel ships 120 rounds at 0.1 (notebook size limit)


def v0_pick(d):
    return d.sort_values(["vote_sum", "fr"], ascending=[False, False]).index[0]


def ev(n_ins, hits, n_mol):
    return (0.5 * hits - 0.025 * n_ins) / n_mol * 1000


def main(paths, save=None):
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    mols = df.mol.unique()
    rng = np.random.default_rng(0)
    fold = dict(zip(mols, rng.integers(0, 5, len(mols))))
    df["fold"] = df.mol.map(fold)
    X = df[ours_aesite.COLS].to_numpy(float)
    df["p"] = np.nan
    for f in range(5):
        tr, te = df.fold != f, df.fold == f
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], df.label[tr]), ROUNDS)
        df.loc[te, "p"] = m.predict(X[te])
    n_mol = len(mols)
    orc = df.groupby("mol").label.max().sum()
    print(f"molecules with products {n_mol}  oracle {orc} ({orc / n_mol:.3f})  rows {len(df)}  positives {df.label.sum()}")
    g = df.groupby("mol", sort=False)
    v0 = df.loc[[v0_pick(d) for _, d in g]]
    v1 = df.loc[g.p.idxmax().values]
    print(f"pick hit (all with products): v0 {v0.label.mean():.3f}  v1 {v1.label.mean():.3f}")
    v0g = v0[(v0.n_new <= 4) | (v0.votes >= 2)]
    print(f"v0 rule (n_new <= 4 or votes >= 2): inserted {len(v0g)}  hits {v0g.label.sum()}  hit {v0g.label.mean():.3f}  "
          f"EV/1000 mol {ev(len(v0g), v0g.label.sum(), n_mol):+.1f}")
    for th in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        s = v1[v1.p >= th]
        print(f"v1 p >= {th:.2f}: inserted {len(s):4d}  hits {s.label.sum():4d}  hit {s.label.mean() if len(s) else 0:.3f}  "
              f"EV/1000 mol {ev(len(s), s.label.sum(), n_mol):+.1f}")
    s = v1[(v1.n_new <= 4) | (v1.votes >= 2)]
    print(f"v1 pick + v0 rule: inserted {len(s)}  hits {s.label.sum()}  hit {s.label.mean():.3f}  EV/1000 {ev(len(s), s.label.sum(), n_mol):+.1f}")
    # calibration
    v1 = v1.assign(b=pd.cut(v1.p, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]))
    print(v1.groupby("b", observed=True).label.agg(["size", "mean"]).to_string())
    m = lgb.train(PARAMS, lgb.Dataset(X, df.label), ROUNDS)
    imp = sorted(zip(m.feature_importance("gain"), ours_aesite.COLS), reverse=True)
    print("gain:", ", ".join(f"{c} {v:.0f}" for v, c in imp[:15]))
    if save:
        m.save_model(save)
        print("saved", save)


if __name__ == "__main__":
    a = sys.argv[1:]
    sv = None
    if "--save" in a:
        i = a.index("--save")
        sv = a[i + 1]
        a = a[:i] + a[i + 2:]
    main(a, sv)

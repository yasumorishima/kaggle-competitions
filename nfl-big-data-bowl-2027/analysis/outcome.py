"""ML pillar 2, first form: does drill tracking add to stopwatch + draft slot when predicting a game outcome?

Target (WR): the player's mean separation at pass forward over his routes, adjusted for route type (separation minus
the route's league mean among these rookies), players with >= MIN_ROUTES routes.
Feature sets: base = combine stopwatch / size (forty, ten split, three cone, shuttle, vertical, broad, height, weight)
+ draft overall pick; drill = per-player cut fingerprint from the WR drills (mean residual loss / decel / entry speed
against the population curve, by angle bin) + peak drill speed. Models: ridge (standardised, median-imputed) and a small
LightGBM, repeated 5-fold CV by player; reported: R^2 out of fold and the uplift (drill + base minus base) with a player
bootstrap 90% interval on the paired squared-error difference.
Usage: python outcome.py DATA_DIR CUTS_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MIN_ROUTES = 40


def target_wr(data):
    pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "route_ran", "separation_at_pass_forward"])
    pp = pp.dropna(subset=["separation_at_pass_forward", "route_ran"])
    pp["adj"] = pp.separation_at_pass_forward - pp.groupby("route_ran").separation_at_pass_forward.transform("mean")
    g = pp.groupby("nfl_id").agg(sep=("adj", "mean"), n=("adj", "size"))
    return g[g.n >= MIN_ROUTES]


def drill_feats(cd, drills_prefix="SKILL_DRILLS_WR"):
    d = pd.read_parquet(os.path.join(cd, "drill_cuts.parquet"))
    d = d[d.drill_type == drills_prefix].copy()
    d["ab"] = pd.cut(d.angle, [30, 60, 90, 180])
    pop = d.groupby("ab", observed=True)[["loss", "decel", "v_in", "v_min"]].median()
    d = d.join(pop, on="ab", rsuffix="_pop")
    for k in ("loss", "decel", "v_in", "v_min"):
        d["r_" + k] = d[k] - d[k + "_pop"]
    f = d.groupby(["nfl_id", "ab"], observed=True)[["r_loss", "r_decel", "r_v_in"]].mean().unstack()
    f.columns = [f"{a}_{b.left:.0f}" for a, b in f.columns]
    f["v_peak"] = d.groupby("nfl_id").v_out.max()
    f["n_cuts"] = d.groupby("nfl_id").size()
    return f


def cv(X, y, model, reps=10):
    P = np.zeros((reps, len(y)))
    for r in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            m = model(); m.fit(X.iloc[tr], y.iloc[tr]); P[r, te] = m.predict(X.iloc[te])
    return P.mean(0)


def main(data, cd):
    y = target_wr(data).sep
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    base = cr[["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height", "combine_weight"]].join(pl.draft_overall_pick)
    dr = drill_feats(cd)
    ids = y.index.intersection(pl.index[pl.nfl_position == "WR"]).intersection(dr.index)
    y = y.loc[ids]; B = base.reindex(ids); D = B.join(dr.reindex(ids))
    print("WR players", len(ids), "target sd", round(y.std(), 3))
    ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    out = {}
    for name, X in (("base", B), ("base+drill", D), ("drill", dr.reindex(ids))):
        p = cv(X, y, ridge); out[name] = p
        print(f"{name:11s} ridge R2 {1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum():+.3f}")
    try:
        import lightgbm as lgb
        gbm = lambda: lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        for name, X in (("base", B), ("base+drill", D)):
            p = cv(X, y, gbm, reps=5)
            print(f"{name:11s} lgbm  R2 {1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum():+.3f}")
    except ImportError:
        print("lightgbm not installed")
    e0, e1 = (y - out["base"]) ** 2, (y - out["base+drill"]) ** 2
    rng = np.random.default_rng(0); sst = ((y - y.mean()) ** 2).sum()
    bs = [((e0.values[i] - e1.values[i]).sum() / ((y.values[i] - y.values[i].mean()) ** 2).sum()) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    print(f"uplift R2 (ridge) {(e0.sum() - e1.sum()) / sst:+.3f}  90% CI [{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Pillar 2 prep (11-09 milestone): do the traits that TRANSLATE (trait_translation.py, 10-31) add predictive value for a
game outcome over the stopwatch + draft-position baseline?

Traits (drill side only, exactly as in trait_translation.py; mean over 40-yd attempts):
  WR  D_vmax40  40-yd tracking top speed
  CB  D_v10_40  40-yd tracking speed 1.0 s after the start of motion
Outcomes (the data hold no opponents or ball, so no "targeted defender" outcome exists for a CB):
  WR  sep   PRIMARY. separation_at_pass_forward minus the route type's mean, player mean over >= 40 routes (outcome.py)
      tpr   targets per route run (player_play.target over routes with route_ran), >= 40 routes
      yprr  receiving yards per route run, >= 40 routes
  CB  brk   PRIMARY. break on the throw: speed 0.8 s after pass_forward minus speed at pass_forward, 80th pct over pass
            plays (db_outcome.py; cached in CACHE_DIR/db_game.parquet), >= 30 plays
      tkl   (tackles + 0.5 assists) per defensive play, >= 100 plays (db_outcome.py)
      snap  speed 1.0 s after his own start after the snap (db_outcome.py) -- the game side of the translated trait itself,
            shown for reference only (not an outcome)
Baselines: full = 8 stopwatch / size measures (forty, ten split, three cone, shuttle, vertical, broad, height, weight) +
log(draft overall pick); small = forty, ten split, weight, log(pick) (n is 47-90, so 9 features overfit).
Model: ridge (median impute, standardise, RidgeCV), 5-fold CV by player repeated 20 times (out-of-fold prediction averaged
over repeats); R2 out of fold. Increment = R2(base + trait) - R2(base) with a player bootstrap 90% interval on the
paired squared-error difference (2,000 resamples). Also Spearman trait -> outcome and the partial after the small base.
Usage: python pillar2.py DATA_DIR CACHE_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_outcome import DEF_POS, game_targets   # noqa: E402
from outcome import cv                          # noqa: E402
from trait_translation import drill_table, drill_traits, partial, rho   # noqa: E402

FULL = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height",
        "combine_weight", "log_pick"]
SMALL = ["forty", "ten_yd_split", "combine_weight", "log_pick"]
REPS = 20


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))


def wr_outcomes(data):
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["nfl_id", "route_ran", "separation_at_pass_forward", "target", "rec_yards"])
    r = pp.dropna(subset=["route_ran"]).copy()
    r["tgt"] = r.target.astype(str).str.upper().isin(["TRUE", "1", "1.0"])
    o = r.groupby("nfl_id").agg(n_routes=("route_ran", "size"), tpr=("tgt", "mean"),
                                yprr=("rec_yards", lambda v: v.fillna(0).sum()))
    o["yprr"] = o.yprr / o.n_routes
    s = r.dropna(subset=["separation_at_pass_forward"]).copy()
    s["adj"] = s.separation_at_pass_forward - s.groupby("route_ran").separation_at_pass_forward.transform("mean")
    g = s.groupby("nfl_id").adj.agg(["mean", "size"])
    o["sep"] = g["mean"].where(g["size"] >= 40)
    return o[o.n_routes >= 40]


def cb_outcomes(data, cache, pl):
    dbs = set(pl.index[pl.nfl_position.isin(sorted(DEF_POS))])
    gt = game_targets(data, dbs, cache)
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["nfl_id", "team_abbr", "possession_team", "tackle", "assist"])
    pp = pp[(pp.team_abbr != pp.possession_team) & pp.nfl_id.isin(dbs)]
    tk = pp.groupby("nfl_id").agg(tk=("tackle", "sum"), asst=("assist", "sum"), n=("tackle", "size"))
    o = gt.join(tk, how="outer")
    o["tkl"] = ((o.tk + 0.5 * o.asst) / o.n).where(o.n >= 100)
    o["brk"] = o.brk.where(o.n_brk >= 30)
    o["snap"] = o.snap.where(o.n_snap >= 50)
    return o


def test(y, B, T, label):
    y = y.astype(float)
    sst = ((y - y.mean()) ** 2).sum()
    p0, p1, pt = cv(B, y, ridge, REPS), cv(B.join(T), y, ridge, REPS), cv(T, y, ridge, REPS)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        bs.append((e0.values[i] - e1.values[i]).sum() / ((y.values[i] - y.values[i].mean()) ** 2).sum())
    lo, hi = np.percentile(bs, [5, 95])
    inc = (e0.sum() - e1.sum()) / sst
    print(f"  {label:34s} n {len(y):3d}  base R2 {1 - e0.sum() / sst:+.3f}  base+trait {1 - e1.sum() / sst:+.3f}  "
          f"trait only {1 - ((y - pt) ** 2).sum() / sst:+.3f}  increment {inc:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return inc, lo, hi


def run(pos_label, ids, trait, outcomes, O, base, D):
    print(f"\n== {pos_label}: trait {trait}")
    for oc in outcomes:
        k = O.reindex(ids)[oc].dropna()
        k = k[k.index.isin(D.index[D[trait].notna()])]
        if len(k) < 20:
            print(f"  {oc}: n {len(k)} < 20, skipped")
            continue
        t = D.loc[k.index, trait]
        bb = base.reindex(k.index)
        ok = bb[SMALL].notna().all(axis=1)
        pr = partial(t[ok].values, k[ok].values, bb.loc[ok, SMALL].values)
        print(f"  [{oc}] Spearman trait -> outcome {rho(t.values, k.values):+.2f} (n {len(k)});  partial | small base "
              f"{pr:+.2f} (n {int(ok.sum())});  outcome sd {k.std():.3f}")
        for nm, cols in (("full base (8 stopwatch + log pick)", FULL), ("small base (forty, 10yd, wt, log pick)", SMALL)):
            test(k, bb[cols], D.loc[k.index, [trait]], f"{oc} | {nm}")


def main(data, cache):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    base = cr[FULL[:-1]].join(np.log(pl.draft_overall_pick).rename("log_pick"), how="outer")
    D, _ = drill_traits(drill_table(data))
    print("PILLAR 2 PREP: translated drill traits -> game outcomes, increment over stopwatch + draft position")
    print(f"ridge, 5-fold CV by player x {REPS} repeats; increment = R2(base+trait) - R2(base), player bootstrap 90% CI")
    print("undrafted players: log_pick missing -> median-imputed inside each fold "
          f"(missing pick among players: {int(pl.draft_overall_pick.isna().sum())})")
    wr = pl.index[pl.nfl_position == "WR"]
    run("WR", wr, "D_vmax40", ["sep", "tpr", "yprr"], wr_outcomes(data), base, D)
    cb = pl.index[pl.nfl_position.isin(["CB", "DB"])]
    run("CB", cb, "D_v10_40", ["brk", "tkl", "snap"], cb_outcomes(data, cache, pl), base, D)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

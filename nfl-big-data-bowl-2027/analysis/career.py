"""Does the forty-yard tracking curve add to stopwatch + draft slot for career playing time (per season)?

Target: log1p(snaps on the player's side of the ball per season since his draft; seasons = 2026 - draft_year).
Base: forty, ten split, three cone, shuttle, vertical, broad, height, weight, log(draft overall pick).
Tracking: the best forty attempt's speed 0.5 / 1.0 / 1.5 s after first movement, peak 0.3 s acceleration, time to 5 yd
(burst.curve). Ridge, repeated 5-fold CV by player, paired bootstrap 90% interval of the R^2 uplift.
Usage: python career.py DATA_DIR POS[,POS...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from burst import curve


def forty_curves(data):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & (c.drill_type == "FORTY_YARD_DASH")]
    rows = []
    for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):
        g = g.sort_values("time"); r = curve(g.x.values, g.y.values)
        if r:
            r["nfl_id"] = pid; rows.append(r)
    return pd.DataFrame(rows).sort_values("t10").groupby("nfl_id").first()


def cvpred(X, y, reps=20):
    P = np.zeros((reps, len(y)))
    for r in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
            m.fit(X.iloc[tr], y.iloc[tr]); P[r, te] = m.predict(X.iloc[te])
    return P.mean(0)


def main(data, poss):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    cs = pd.read_csv(os.path.join(data, "player_career_successes.csv")).set_index("nfl_id")
    fc = forty_curves(data)
    ids = pl.index[pl.nfl_position.isin(poss)].intersection(fc.index)
    off = pl.loc[ids, "nfl_position"].isin(["WR", "TE", "T", "G", "C", "RB", "FB"])
    snaps = np.where(off, cs.reindex(ids).career_offensive_snaps, cs.reindex(ids).career_defensive_snaps)
    seasons = 2026 - pl.loc[ids, "draft_year"]
    y = pd.Series(np.log1p(snaps / seasons.values), index=ids)
    base = cr.reindex(ids)[["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height", "combine_weight"]]
    base["log_pick"] = np.log(pl.loc[ids, "draft_overall_pick"])
    trk = fc.reindex(ids)[["v0.5", "v1.0", "v1.5", "amax", "t5"]]
    sst = ((y - y.mean()) ** 2).sum()
    p0 = cvpred(base, y); p1 = cvpred(base.join(trk), y)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    rng = np.random.default_rng(0)
    bs = [(e0.values[i] - e1.values[i]).sum() / ((y.values[i] - y.values[i].mean()) ** 2).sum()
          for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    print(f"{','.join(poss)} n {len(y)}  base R2 {1 - e0.sum() / sst:+.3f}  base+forty-curve R2 {1 - e1.sum() / sst:+.3f}  "
          f"uplift {(e0.sum() - e1.sum()) / sst:+.3f}  90% CI [{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))

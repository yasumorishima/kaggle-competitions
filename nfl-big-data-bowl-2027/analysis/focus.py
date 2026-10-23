"""Pillar 2, focused: one or two drill features per position beside stopwatch + draft slot.

DL / edge: run-the-hoop active duration (dl_drills.py: rho with position-adjusted get-off, uplift +0.35 alone).
CB: forty-yard peak 0.3 s acceleration (db_outcome.py: snap uplift +0.084 with the whole curve).
For each: ridge and a small GBM (sklearn HistGradientBoosting), repeated 5-fold CV by player, uplift over the base with a
player bootstrap 90% interval (db_outcome.test for ridge; the same bootstrap for the GBM).
Usage: python focus.py DATA_DIR CACHE_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_outcome import forty, game_targets, test, DEF_POS      # noqa: E402
from dl_drills import feats, DRILLS, SHORT                      # noqa: E402
from outcome import cv                                         # noqa: E402


def gbm_test(y, B, C, label):
    from sklearn.ensemble import HistGradientBoostingRegressor
    g = lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, max_leaf_nodes=7, min_samples_leaf=10)  # noqa: E731
    sst = ((y - y.mean()) ** 2).sum()
    p0, p1 = cv(B, y, g, reps=10), cv(B.join(C), y, g, reps=10)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        bs.append((e0.values[i] - e1.values[i]).sum() / ((y.values[i] - y.values[i].mean()) ** 2).sum())
    print(f"{label} [GBM]: n {len(y)}  base R2 {1 - e0.sum() / sst:+.3f}  base+feat {1 - e1.sum() / sst:+.3f}  uplift "
          f"{(e0.sum() - e1.sum()) / sst:+.3f} [{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]")


def base_table(data):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    return pl, cr[["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height",
                   "combine_weight"]].join(pl.draft_overall_pick)


def dl(data, pl, base):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & (c.drill_name == "RUN_THE_HOOP_DRILL")]
    rows = []
    for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):
        g = g.sort_values("time")
        f = feats(g.x.values, g.y.values)
        if f:
            f["nfl_id"] = pid; rows.append(f)
    d = pd.DataFrame(rows).sort_values("dur").groupby("nfl_id").first().add_prefix("hoop_")
    pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "player_get_off"]).dropna()
    t = pp.groupby("nfl_id").player_get_off.agg(["mean", "size"])
    t = t[t["size"] >= 30]
    pos = pl.nfl_position.reindex(t.index)
    t = t[pos.isin(["DE", "DT", "NT", "OLB"])]
    t["adj"] = t["mean"] - t.groupby(pos.reindex(t.index))["mean"].transform("mean")
    ids = t.index.intersection(d.index).intersection(base.index)
    y = t.loc[ids, "adj"]
    for cols in (["hoop_dur"], ["hoop_dur", "hoop_amax"], ["hoop_dur", "hoop_vmax"]):
        C = d.reindex(ids)[cols]
        test(y, base.reindex(ids), C, "DL get-off + " + "+".join(cols))
        gbm_test(y, base.reindex(ids), C, "DL get-off + " + "+".join(cols))


def cb(data, out, pl, base):
    dr = forty(data)
    gt = game_targets(data, set(pl.index[pl.nfl_position.isin(sorted(DEF_POS))]), out)
    ids = pl.index[pl.nfl_position == "CB"].intersection(dr.index).intersection(base.index)
    k = gt.reindex(ids)
    y = k["snap"][k["n_snap"] >= 50].dropna()
    for cols in (["amax"], ["amax", "r05"]):
        C = dr.reindex(y.index)[cols]
        test(y, base.reindex(y.index), C, "CB snap + " + "+".join(cols))
        gbm_test(y, base.reindex(y.index), C, "CB snap + " + "+".join(cols))


if __name__ == "__main__":
    pl, base = base_table(sys.argv[1])
    dl(sys.argv[1], pl, base)
    cb(sys.argv[1], sys.argv[2], pl, base)

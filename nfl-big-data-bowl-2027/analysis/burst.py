"""Straight-line burst: the forty-yard dash tracking curve vs the first second after the snap in games.

Drill: per forty attempt, speed curve from the start of motion (first frame with speed > 0.5 yd/s): speed at 0.5 / 1.0 /
1.5 s, peak acceleration over 0.3 s, time to 5 / 10 yd; per player the best attempt (fastest 10 yd).
Game: per play from the snap, the player's speed 0.5 / 1.0 / 1.5 s after his own first movement (speed > 0.5 yd/s within
2 s of the snap); per player the 80th percentile over plays (his intended burst, not every jog).
Prints Spearman drill vs game for every position group with >= 20 players, against the stopwatch ten-yard split.
Usage: python burst.py DATA_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HZ = 10


def curve(x, y):
    sp = np.hypot(np.gradient(x), np.gradient(y)) * HZ
    mv = np.nonzero(sp > 0.5)[0]
    if not len(mv):
        return None
    s0 = mv[0]
    s = sp[s0:]
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x[s0:]), np.diff(y[s0:])))])
    out = {f"v{t}": s[int(t * HZ)] if len(s) > t * HZ else np.nan for t in (0.5, 1.0, 1.5)}
    for yd in (5, 10):
        k = np.nonzero(d >= yd)[0]
        out[f"t{yd}"] = k[0] / HZ if len(k) else np.nan
    out["amax"] = np.nanmax((s[3:] - s[:-3]) / 0.3) if len(s) > 3 else np.nan
    return out


def main(data):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id").nfl_position
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & (c.drill_type == "FORTY_YARD_DASH")]
    rows = []
    for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):
        g = g.sort_values("time")
        r = curve(g.x.values, g.y.values)
        if r:
            r["nfl_id"] = pid; rows.append(r)
    dr = pd.DataFrame(rows).sort_values("t10").groupby("nfl_id").first()
    print("forty players", len(dr))
    keep = set(dr.index)
    G = []
    for yr in (2023, 2024, 2025):
        for ch in pd.read_csv(os.path.join(data, f"game_tracking_{yr}.csv"), chunksize=3_000_000,
                              usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]):
            G.append(ch[ch.nfl_id.isin(keep)])
    g = pd.concat(G, ignore_index=True).sort_values(["game_id", "play_id", "nfl_id", "time"])
    grows = []
    for (gid, play, pid), t in g.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        ev = t.event.fillna("").values
        sn = np.nonzero(np.isin(ev, ["ball_snap", "snap_direct"]))[0]
        if not len(sn):
            continue
        x, y = t.x.values[sn[0]:sn[0] + 40], t.y.values[sn[0]:sn[0] + 40]
        if len(x) < 25:
            continue
        r = curve(x, y)
        if r:
            r["nfl_id"] = pid; grows.append(r)
    gm = pd.DataFrame(grows).groupby("nfl_id").quantile(0.8)
    n = pd.DataFrame(grows).groupby("nfl_id").size()
    gm = gm[n.reindex(gm.index) >= 50]
    j = dr.join(gm, lsuffix="_d", rsuffix="_g", how="inner").join(cr.ten_yd_split)
    j["grp"] = j.index.map(lambda p: {"WR": "WR", "CB": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "TE": "TE",
                                      "DE": "DL", "DT": "DL", "NT": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB",
                                      "T": "OL", "G": "OL", "C": "OL"}.get(pl.get(p), "other"))
    j["pos"] = j.index.map(lambda p: pl.get(p))
    j = j.join(cr.combine_weight)
    for pos in ("DE", "DT", "OLB", "WR", "CB"):
        k = j[j.pos == pos]
        if len(k) >= 15:
            res = k[["v1.5_d", "v1.5_g", "combine_weight", "ten_yd_split"]].dropna()
            # partial out weight: rank residuals
            def resid(a, b):
                A = np.vstack([np.ones(len(b)), b]).T
                return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
            pr = spearmanr(resid(res["v1.5_d"].values, res.combine_weight.values), resid(res["v1.5_g"].values, res.combine_weight.values))[0]
            print(f"  {pos} n {len(k)}  v1.5 drill-game {spearmanr(k['v1.5_d'], k['v1.5_g'])[0]:+.2f}  partial(weight) {pr:+.2f}  ten-split vs game v1.5 {spearmanr(res.ten_yd_split, res['v1.5_g'])[0]:+.2f}")
    for grp, k in j.groupby("grp"):
        if len(k) < 20:
            continue
        msg = [f"{grp} n {len(k)}"]
        for v in ("v0.5", "v1.0", "v1.5", "amax"):
            msg.append(f"{v} {spearmanr(k[v + '_d'], k[v + '_g'])[0]:+.2f}")
        msg.append(f"| ten-split vs game v1.0 {spearmanr(k.ten_yd_split, k['v1.0_g'], nan_policy='omit')[0]:+.2f}")
        print("  ".join(msg))


if __name__ == "__main__":
    main(sys.argv[1])

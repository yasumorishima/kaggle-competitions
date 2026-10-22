"""ML pillar 2 for DBs: does the forty-yard dash tracking curve add to stopwatch + draft slot for a game outcome?

The game tracking holds only the rookies themselves (no ball, no opponents), so the outcomes come from the DB's own motion
and the play table:
  snap  : speed 1.0 s after his own first movement after the snap (80th percentile over plays; as burst.py)
  brk   : the break on the throw = speed 0.8 s after pass_forward minus speed at pass_forward (80th percentile over the
          pass plays he is on; a DB reading the throw and driving on the ball)
  tkl   : (tackles + 0.5 assists) per play on defence (player_play)
Feature sets: base = stopwatch / size (forty, ten split, three cone, shuttle, vertical, broad, height, weight) + draft
overall pick; curve = the forty tracking curve (speed at 0.5 / 1.0 / 1.5 s from first motion, peak 0.3 s acceleration,
time to 5 / 10 yd; best attempt) and two shape ratios (v0.5 / v1.5, v1.0 / v1.5).
Ridge, repeated 5-fold CV by player; uplift = base+curve minus base with a player bootstrap 90% interval.
Usage: python db_outcome.py DATA_DIR [OUT_DIR]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burst import curve          # noqa: E402
from outcome import cv           # noqa: E402

MIN_P = 30
DEF_POS = {"CB", "SS", "FS", "DB", "S"}
HZ = 10


def game_targets(data, ids, out):
    f = os.path.join(out, "db_game.parquet") if out else None
    if f and os.path.exists(f):
        return pd.read_parquet(f)
    G = []
    for yr in (2023, 2024, 2025):
        for ch in pd.read_csv(os.path.join(data, f"game_tracking_{yr}.csv"), chunksize=3_000_000,
                              usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]):
            G.append(ch[ch.nfl_id.isin(ids)])
    g = pd.concat(G, ignore_index=True).sort_values(["game_id", "play_id", "nfl_id", "time"])
    rows = []
    for (gid, play, pid), t in g.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        ev = t.event.fillna("").values
        x, y = t.x.values, t.y.values
        sp = np.hypot(np.gradient(x), np.gradient(y)) * HZ if len(x) > 2 else np.zeros(len(x))
        r = {"nfl_id": pid, "snap": np.nan, "brk": np.nan}
        sn = np.nonzero(np.isin(ev, ["ball_snap", "snap_direct"]))[0]
        if len(sn) and len(x) - sn[0] >= 25:
            c = curve(x[sn[0]:sn[0] + 40], y[sn[0]:sn[0] + 40])
            if c:
                r["snap"] = c["v1.0"]
        pf = np.nonzero(ev == "pass_forward")[0]
        if len(pf) and pf[0] + 8 < len(sp):
            r["brk"] = sp[pf[0] + 8] - sp[pf[0]]
        rows.append(r)
    d = pd.DataFrame(rows)
    q = d.groupby("nfl_id")[["snap", "brk"]].quantile(0.8)
    n = d.groupby("nfl_id")[["snap", "brk"]].count().add_prefix("n_")
    q = q.join(n)
    if f:
        q.to_parquet(f)
    return q


def forty(data):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & (c.drill_type == "FORTY_YARD_DASH")]
    rows = []
    for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):
        g = g.sort_values("time")
        r = curve(g.x.values, g.y.values)
        if r:
            r["nfl_id"] = pid
            rows.append(r)
    d = pd.DataFrame(rows).sort_values("t10").groupby("nfl_id").first()
    d["r05"] = d["v0.5"] / d["v1.5"]
    d["r10"] = d["v1.0"] / d["v1.5"]
    return d


def test(y, B, C, label):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))  # noqa: E731
    sst = ((y - y.mean()) ** 2).sum()
    p0, p1 = cv(B, y, ridge, reps=20), cv(B.join(C), y, ridge, reps=20)
    pc = cv(C, y, ridge, reps=20)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        bs.append((e0.values[i] - e1.values[i]).sum() / ((y.values[i] - y.values[i].mean()) ** 2).sum())
    print(f"{label}: n {len(y)}  base R2 {1 - e0.sum() / sst:+.3f}  base+curve {1 - e1.sum() / sst:+.3f}  curve only "
          f"{1 - ((y - pc) ** 2).sum() / sst:+.3f}  uplift {(e0.sum() - e1.sum()) / sst:+.3f} "
          f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]")


def main(data, out=None):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    base = cr[["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height",
               "combine_weight"]].join(pl.draft_overall_pick)
    dr = forty(data)[["v0.5", "v1.0", "v1.5", "amax", "t5", "t10", "r05", "r10"]]
    dbs = set(pl.index[pl.nfl_position.isin(sorted(DEF_POS))])
    gt = game_targets(data, dbs, out)
    pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "team_abbr", "possession_team", "tackle", "assist"])
    pp = pp[(pp.team_abbr != pp.possession_team) & pp.nfl_id.isin(dbs)]
    tk = pp.groupby("nfl_id").agg(tk=("tackle", "sum"), asst=("assist", "sum"), n=("tackle", "size"))
    tk["tkl"] = (tk.tk + 0.5 * tk.asst) / tk.n
    y_all = gt.join(tk[["tkl", "n"]], how="outer")
    print("DBs with game data", len(gt), "with plays", len(tk))
    for pos_set, lab in ((["CB"], "CB"), (sorted(DEF_POS), "DB")):
        ids = pl.index[pl.nfl_position.isin(pos_set)].intersection(dr.index).intersection(base.index)
        for tcol, ncol, mn in (("snap", "n_snap", 50), ("brk", "n_brk", MIN_P), ("tkl", "n", 100)):
            k = y_all.reindex(ids)
            y = k[tcol][k[ncol] >= mn].dropna()
            if len(y) >= 20:
                test(y, base.reindex(y.index), dr.reindex(y.index), f"{lab} {tcol}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

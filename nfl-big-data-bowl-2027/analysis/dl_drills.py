"""DL drills by type -> game get-off: does the position-drill tracking add to stopwatch + draft slot?

Drill features, per player and drill_name (best attempt = the shortest active duration): active duration (frames with
speed > 0.5 yd/s, first to last), speed 0.5 / 1.0 s after first motion, peak 0.3 s acceleration, peak speed.
Drills: the seven SKILL_DRILLS_DL drills with >= 100 players (pass rush, front-start wave + lateral reaction, four bag,
run and club, body control, run the hoop).
Target: mean player_get_off over the player's plays (player_play, >= 30 plays; seconds from snap, lower = faster),
residual on position (DE / DT / NT / OLB means removed).
Ridge, repeated 5-fold CV by player, uplift (base + drill minus base) with a player bootstrap 90% interval; also the
Spearman of each drill feature with the target, beside the ten-yard split's.
Usage: python dl_drills.py DATA_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_outcome import test      # noqa: E402

HZ = 10
DRILLS = ["PASS_RUSH_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "FOUR_BAG_AGILITY_DRILL",
          "RUN_AND_CLUB_DRILL", "BODY_CONTROL_DRILL", "RUN_THE_HOOP_DRILL"]
SHORT = {"PASS_RUSH_DRILL": "rush", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION": "wave", "FOUR_BAG_AGILITY_DRILL": "bag4",
         "RUN_AND_CLUB_DRILL": "club", "BODY_CONTROL_DRILL": "body", "RUN_THE_HOOP_DRILL": "hoop"}


def feats(x, y):
    sp = np.hypot(np.gradient(x), np.gradient(y)) * HZ
    mv = np.nonzero(sp > 0.5)[0]
    if len(mv) < 5:
        return None
    a, b = mv[0], mv[-1]
    s = sp[a:b + 1]
    return dict(dur=(b - a) / HZ, v05=s[5] if len(s) > 5 else np.nan, v10=s[10] if len(s) > 10 else np.nan,
                amax=np.nanmax((s[3:] - s[:-3]) / 0.3) if len(s) > 3 else np.nan, vmax=s.max())


def main(data):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(DRILLS)]
    rows = []
    for (pid, dn, eid, att), g in c.groupby(["nfl_id", "drill_name", "event_id", "attempt"]):
        g = g.sort_values("time")
        f = feats(g.x.values, g.y.values)
        if f:
            f.update(nfl_id=pid, drill=SHORT[dn]); rows.append(f)
    d = pd.DataFrame(rows).sort_values("dur").groupby(["nfl_id", "drill"]).first().unstack()
    d.columns = [f"{b}_{a}" for a, b in d.columns]
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    base = cr[["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height",
               "combine_weight"]].join(pl.draft_overall_pick)
    pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "player_get_off"]).dropna()
    t = pp.groupby("nfl_id").player_get_off.agg(["mean", "size"])
    t = t[t["size"] >= 30]
    pos = pl.nfl_position.reindex(t.index)
    t = t[pos.isin(["DE", "DT", "NT", "OLB"])]
    t["adj"] = t["mean"] - t.groupby(pos.reindex(t.index))["mean"].transform("mean")
    ids = t.index.intersection(d.index).intersection(base.index)
    y = t.loc[ids, "adj"]
    print("DL / edge players with get-off and drills", len(ids), "target sd", round(y.std(), 4))
    for col in sorted(d.columns):
        k = d.loc[ids, col]
        if k.notna().sum() >= 25:
            print(f"  {col:12s} rho {spearmanr(k, y, nan_policy='omit')[0]:+.2f}  n {k.notna().sum()}")
    print(f"  ten_yd_split rho {spearmanr(base.loc[ids, 'ten_yd_split'], y, nan_policy='omit')[0]:+.2f}")
    test(y, base.reindex(ids), d.reindex(ids), "DL get-off, all drills")
    for dr in SHORT.values():
        cols = [c_ for c_ in d.columns if c_.startswith(dr + "_")]
        test(y, base.reindex(ids), d.reindex(ids)[cols], f"DL get-off, {dr}")


if __name__ == "__main__":
    main(sys.argv[1])

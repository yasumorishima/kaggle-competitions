"""Scene-matched whole-route segments for ML pillar 1 (drill <-> game), WR and DB.

Instead of +-1 s around a cut apex (segs.py), each segment is the whole movement from its start:
  drill: the whole attempt from the first frame the player moves (speed >= 1 yd/s, minus 0.3 s)
  game : from the snap, up to the pass (or a sack / run / handoff), at most T frames
Same transform on both sides (shape only, no field position, no drill layout): origin at the start position, the
initial heading (mean velocity over +0.3..+0.8 s) rotated onto +y, mirrored so the net lateral movement goes to +x.
Channels: x, y (yd), speed (yd/s); T = 40 frames (4 s), padded with the last value; `n` = number of real frames.

Scene labels used for matching:
  drill: drill_name (WR route drills, DB back-pedal drills)
  game : route_ran (WR), lined_up_position, team_coverage_man_zone, coverage_assignment, and for DBs `back` = mean
         gap between facing `o` and the heading over the first 1.0 s of movement (> 110 deg = opens in a back-pedal).
Usage: python route_segs.py DATA_DIR OUT_DIR   (writes route_segs.npz + route_meta.parquet into OUT_DIR)
"""
import os
import sys

import numpy as np
import pandas as pd

from cuts import HZ, _smooth

T = 40
WR_DRILLS = ["SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "GO_ROUTE_RIGHT",
             "RED_ZONE_FADE_RIGHT", "POST_CORNER_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT",
             "SLOT_STRIKE_ROUTE_LEFT", "SPEED_OUT_LEFT", "SPEED_OUT_ROUTE_LEFT"]
DB_DRILLS = ["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BOX_DRILL",
             "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2", "W_DRILL"]
DBPOS = {"CB", "SS", "FS", "DB"}


def canon(x, y, start, o=None):
    """Return (3, T) segment, real length, back-pedal gap (deg) or None."""
    xs, ys = _smooth(np.asarray(x, float)), _smooth(np.asarray(y, float))
    vx, vy = np.gradient(xs) * HZ, np.gradient(ys) * HZ
    sp = np.hypot(vx, vy)
    seg = slice(start, min(len(xs), start + T))
    n = seg.stop - start
    if n < 15:
        return None
    u = np.array([vx[start + 3:start + 9].mean(), vy[start + 3:start + 9].mean()])
    if np.hypot(*u) < 0.8:
        return None
    th = np.arctan2(u[0], u[1])
    c, s = np.cos(th), np.sin(th)
    px, py = xs[seg] - xs[start], ys[seg] - ys[start]
    rx, ry = c * px - s * py, s * px + c * py
    if rx[-1] < 0:
        rx = -rx
    out = np.stack([rx, ry, sp[seg]])
    if n < T:
        out = np.concatenate([out, np.repeat(out[:, -1:], T - n, 1)], 1)
    back = np.nan
    if o is not None:
        a, b = start + 1, min(len(xs), start + 11)
        hd = np.degrees(np.arctan2(vx[a:b], vy[a:b])) % 360
        gap = np.abs((np.asarray(o, float)[a:b] - hd + 180) % 360 - 180)
        m = sp[a:b] > 1.0
        back = float(gap[m].mean()) if m.any() else np.nan
    return out.astype(np.float32), n, back


def drills(data):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(WR_DRILLS + DB_DRILLS)].sort_values(
        ["event_id", "nfl_id", "drill_name", "attempt", "time"])
    X, meta = [], []
    for (eid, pid, dn, att), g in c.groupby(["event_id", "nfl_id", "drill_name", "attempt"], sort=False):
        xs, ys = _smooth(g.x.values.astype(float)), _smooth(g.y.values.astype(float))
        sp = np.hypot(np.gradient(xs), np.gradient(ys)) * HZ
        mv = np.nonzero(sp >= 1.0)[0]
        if not len(mv):
            continue
        r = canon(g.x.values, g.y.values, max(0, mv[0] - 3))
        if r is None:
            continue
        X.append(r[0]); meta.append(dict(nfl_id=pid, game=0, scene=dn, n=r[1], src=f"{eid}_{att}"))
    return X, meta


def games(data):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id").nfl_position
    keep = set(pl.index[pl.isin(DBPOS | {"WR"})])
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["game_id", "play_id", "nfl_id", "route_ran", "lined_up_position", "pass_result",
                              "team_coverage_man_zone", "coverage_assignment", "play_nullified_by_penalty"])
    pp = pp[pp.nfl_id.isin(keep) & pp.pass_result.notna()].set_index(["game_id", "play_id", "nfl_id"])
    X, meta = [], []
    ends = {"pass_forward", "pass_shovel", "qb_sack", "qb_strip_sack", "run", "handoff"}
    for yr in (2023, 2024, 2025):
        parts = []
        for ch in pd.read_csv(os.path.join(data, f"game_tracking_{yr}.csv"), chunksize=4_000_000,
                              usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "o", "event"]):
            parts.append(ch[ch.nfl_id.isin(keep)])
        g = pd.concat(parts, ignore_index=True).sort_values(["game_id", "play_id", "nfl_id", "time"])
        key = g[["game_id", "play_id", "nfl_id"]].values
        brk = np.r_[0, np.nonzero((np.diff(key, axis=0) != 0).any(1))[0] + 1, len(g)]
        x, y, o, ev = g.x.values, g.y.values, g.o.values, g.event.fillna("").values
        for i0, i1 in zip(brk[:-1], brk[1:]):
            k = tuple(key[i0])
            if k not in pp.index:
                continue
            e = ev[i0:i1]
            sn = np.nonzero((e == "ball_snap") | (e == "snap_direct"))[0]
            if not len(sn):
                continue
            s0 = sn[0]
            en = [j for j in np.nonzero(np.isin(e, list(ends)))[0] if j > s0]
            stop = min(en[0] + 1 if en else i1 - i0, s0 + T)
            r = canon(x[i0:i0 + stop], y[i0:i0 + stop], s0, o[i0:i0 + stop])
            if r is None:
                continue
            row = pp.loc[k]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            X.append(r[0]); meta.append(dict(nfl_id=k[2], game=1, scene=str(row.route_ran), n=r[1],
                                             src=f"{k[0]}_{k[1]}", pos=row.lined_up_position,
                                             cov=row.team_coverage_man_zone, assign=row.coverage_assignment,
                                             back=r[2], season=yr))
        print("year", yr, "game segments", len(X), flush=True)
    return X, meta


def main(data, out):
    os.makedirs(out, exist_ok=True)
    Xd, md = drills(data)
    print("drill segments", len(Xd), flush=True)
    Xg, mg = games(data)
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id").nfl_position
    m = pd.DataFrame(md + mg)
    m["grp"] = m.nfl_id.map(lambda p: "WR" if pl.get(p) == "WR" else ("DB" if pl.get(p) in DBPOS else "other"))
    np.save(os.path.join(out, "route_X.npy"), np.stack(Xd + Xg))
    m.to_parquet(os.path.join(out, "route_meta.parquet"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

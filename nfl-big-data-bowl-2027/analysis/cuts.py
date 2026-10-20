"""Cut Fingerprint: detect changes of direction (cuts) in 10 Hz tracking with ONE definition for drills and games.

A cut is a frame where the direction of travel turns by at least MIN_ANGLE degrees between the half second before it
and the half second after it, entered at speed >= MIN_VIN yd/s. Heading comes from the smoothed position track (not the
`dir` column, so the combine and game feeds share one convention). For each cut:
  angle    turn between the mean velocity in [-0.6 s, -0.2 s] and [+0.2 s, +0.6 s] around the apex (deg, 0-180)
  v_in     peak speed in the 1.0 s before the apex (yd/s)
  v_min    lowest speed in +-0.4 s around the candidate frame (the apex)
  decel    (v_in - v_min) / time from the v_in frame to the apex (yd/s^2)
  t_rec    time after the apex until speed is back to 0.9 * v_in (s; NaN if not within 2.0 s)
  v_out    peak speed in the 1.0 s after the apex
  loss     1 - v_min / v_in   (share of entry speed given up in the cut)
Usage: python cuts.py DATA_DIR OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

HZ = 10
MIN_ANGLE = 30.0
MIN_VIN = 3.0
SUPPRESS = 8          # frames: one cut per 0.8 s (the sharpest)


def _smooth(a, k=3):
    if len(a) < k:
        return a
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def detect(x, y, t0_index=0):
    """x, y: positions (yd) at 10 Hz for one continuous track. Returns a list of dicts (frame indices are local)."""
    n = len(x)
    if n < 2 * HZ:
        return []
    xs, ys = _smooth(np.asarray(x, float)), _smooth(np.asarray(y, float))
    vx, vy = np.gradient(xs) * HZ, np.gradient(ys) * HZ
    sp = np.hypot(vx, vy)
    out, cand = [], []
    for i in range(6, n - 6):
        a, b = slice(i - 6, i - 1), slice(i + 2, i + 7)
        u, w = np.array([vx[a].mean(), vy[a].mean()]), np.array([vx[b].mean(), vy[b].mean()])
        nu, nw = np.hypot(*u), np.hypot(*w)
        if nu < 1.0 or nw < 1.0:
            continue
        ang = np.degrees(np.arccos(np.clip(u @ w / (nu * nw), -1, 1)))
        if ang < MIN_ANGLE:
            continue
        j0 = max(0, i - HZ)
        vin = sp[j0:i + 1].max()
        if vin < MIN_VIN:
            continue
        cand.append((ang, i))
    taken = np.zeros(n, bool)
    for ang, i in sorted(cand, reverse=True):
        if taken[max(0, i - SUPPRESS):i + SUPPRESS + 1].any():
            continue
        taken[i] = True
        lo, hi = max(0, i - 4), min(n, i + 5)
        apex = lo + int(np.argmin(sp[lo:hi]))
        j0 = max(0, apex - HZ)
        jin = j0 + int(np.argmax(sp[j0:apex + 1]))
        vin, vmin = sp[jin], sp[apex]
        if vin < MIN_VIN:
            continue
        rec = np.nonzero(sp[apex:min(n, apex + 2 * HZ + 1)] >= 0.9 * vin)[0]
        out.append(dict(frame=t0_index + apex, angle=ang, v_in=vin, v_min=vmin,
                        decel=(vin - vmin) / max(apex - jin, 1) * HZ,
                        t_rec=rec[0] / HZ if len(rec) else np.nan,
                        v_out=sp[apex:min(n, apex + HZ + 1)].max(), loss=1 - vmin / vin))
    return sorted(out, key=lambda d: d["frame"])


def combine_cuts(data):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[c.entity_type == "PLAYER"]
    rows = []
    for (eid, pid, drill, att), g in c.groupby(["event_id", "nfl_id", "drill_name", "attempt"], sort=False):
        g = g.sort_values("time")
        for d in detect(g.x.values, g.y.values):
            d.update(nfl_id=pid, source="drill", drill=drill, drill_type=g.drill_type.iat[0], event_id=eid, attempt=att)
            rows.append(d)
    return pd.DataFrame(rows)


def game_cuts(data, years=(2023, 2024, 2025), positions=("WR", "CB", "SS", "FS", "DB"), max_after_snap=5.0):
    """Cuts from the snap until the pass is thrown (or max_after_snap s): the route / coverage phase."""
    pl = pd.read_csv(os.path.join(data, "players.csv"))
    keep = set(pl.nfl_id[pl.nfl_position.isin(positions)])
    rows = []
    for yr in years:
        for ch in pd.read_csv(os.path.join(data, f"game_tracking_{yr}.csv"), chunksize=3_000_000,
                              usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "o", "event"]):
            ch = ch[ch.nfl_id.isin(keep)]
            rows.append(ch)
    g = pd.concat(rows, ignore_index=True)
    out = []
    for (gid, play, pid), t in g.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        t = t.sort_values("time")
        ev = t.event.fillna("").values
        snap = np.nonzero(np.isin(ev, ["ball_snap", "snap_direct"]))[0]
        if not len(snap):
            continue
        s0 = snap[0]
        end = np.nonzero(np.isin(ev, ["pass_forward", "pass_shovel", "qb_sack", "qb_strip_sack", "run", "handoff"]))[0]
        end = [e for e in end if e > s0]
        s1 = min(end[0] if end else len(t) - 1, s0 + int(max_after_snap * HZ))
        lo = max(0, s0 - 6)
        seg = t.iloc[lo:s1 + 7]
        sx, sy, so = seg.x.values, seg.y.values, seg.o.values
        for d in detect(sx, sy):
            if not (s0 <= lo + d["frame"] <= s1):
                continue
            # backpedal entry: facing vs direction of travel over the 0.6 s before the apex (o and the heading share
            # the NGS convention: degrees clockwise from +y; checked: dir vs atan2(dx, dy) median gap 0.7 deg)
            a, b = max(1, d["frame"] - 6), max(2, d["frame"])
            hd = np.degrees(np.arctan2(np.diff(sx[a - 1:b]), np.diff(sy[a - 1:b]))) % 360
            gap = np.abs((so[a:b] - hd + 180) % 360 - 180)
            d["back_in"] = float(np.nanmean(gap)) if len(gap) else np.nan
            d.update(nfl_id=pid, source="game", game_id=gid, play_id=play, t_after_snap=(lo + d["frame"] - s0) / HZ)
            out.append(d)
    return pd.DataFrame(out)


if __name__ == "__main__":
    data, outd = sys.argv[1], sys.argv[2]
    os.makedirs(outd, exist_ok=True)
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    if which in ("both", "drill"):
        dc = combine_cuts(data)
        dc.to_parquet(os.path.join(outd, "drill_cuts.parquet"))
        print("drill cuts", len(dc), "players", dc.nfl_id.nunique(), flush=True)
    if which in ("both", "game"):
        gc = game_cuts(data)
        gc.to_parquet(os.path.join(outd, "game_cuts.parquet"))
        print("game cuts", len(gc), "players", gc.nfl_id.nunique(), flush=True)

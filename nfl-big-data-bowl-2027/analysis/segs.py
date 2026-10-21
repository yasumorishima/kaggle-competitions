"""Cut windows for the embedding model (ML pillar 1): +-1.0 s around each cut apex from cuts.py, in a cut-centred frame.

Each window (21 frames) is translated so the apex is the origin, rotated so the entry velocity (mean over -0.6..-0.2 s)
points along +y, and mirrored so the turn goes to the right (+x). Channels: x, y (yd), speed (yd/s). Drill and game
windows use the same transform, so a model sees only the shape of the cut, never the field position or the drill layout.
Usage: python segs.py DATA_DIR CUTS_DIR   (writes segs.npz into CUTS_DIR)
"""
import os
import sys

import numpy as np
import pandas as pd

from cuts import HZ, _smooth

W = 10   # frames each side


def window(x, y, apex):
    n = len(x)
    if apex - W < 0 or apex + W >= n:
        return None
    xs, ys = _smooth(np.asarray(x, float)), _smooth(np.asarray(y, float))
    px, py = xs[apex - W:apex + W + 1] - xs[apex], ys[apex - W:apex + W + 1] - ys[apex]
    vx, vy = np.gradient(xs) * HZ, np.gradient(ys) * HZ
    u = np.array([vx[apex - 6:apex - 1].mean(), vy[apex - 6:apex - 1].mean()])
    if np.hypot(*u) < 0.5:
        return None
    th = np.arctan2(u[0], u[1])                      # rotate entry heading onto +y
    c, s = np.cos(th), np.sin(th)
    rx, ry = c * px - s * py, s * px + c * py
    if rx[-1] < 0:                                   # mirror: exit to the right
        rx = -rx
    sp = np.hypot(vx, vy)[apex - W:apex + W + 1]
    return np.stack([rx, ry, sp]).astype(np.float32)


def main(data, cd):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id").nfl_position
    keep_pos = {"WR", "CB", "SS", "FS", "DB"}
    X, meta = [], []
    d = pd.read_parquet(os.path.join(cd, "drill_cuts.parquet"))
    d = d[d.nfl_id.map(pl).isin(keep_pos) & d.drill_type.isin(["SKILL_DRILLS_WR", "SKILL_DRILLS_DB"])]
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & c.nfl_id.isin(set(d.nfl_id))]
    tracks = {k: g.sort_values("time") for k, g in c.groupby(["event_id", "nfl_id", "drill_name", "attempt"], sort=False)}
    for r in d.itertuples():
        g = tracks.get((r.event_id, r.nfl_id, r.drill, r.attempt))
        if g is None:
            continue
        w = window(g.x.values, g.y.values, int(r.frame))
        if w is not None:
            X.append(w); meta.append((r.nfl_id, 0, r.angle, r.drill))
    print("drill windows", len(X), flush=True)
    gc = pd.read_parquet(os.path.join(cd, "game_cuts.parquet"))
    want = set(zip(gc.game_id, gc.play_id, gc.nfl_id))
    for yr in (2023, 2024, 2025):
        parts = []
        for ch in pd.read_csv(os.path.join(data, f"game_tracking_{yr}.csv"), chunksize=3_000_000,
                              usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]):
            ch = ch[ch.nfl_id.isin(set(gc.nfl_id))]
            parts.append(ch)
        g = pd.concat(parts, ignore_index=True)
        cuts = gc.set_index(["game_id", "play_id", "nfl_id"])
        for k, t in g.groupby(["game_id", "play_id", "nfl_id"], sort=False):
            if k not in want:
                continue
            t = t.sort_values("time")
            ev = t.event.fillna("").values
            snap = np.nonzero(np.isin(ev, ["ball_snap", "snap_direct"]))[0]
            if not len(snap):
                continue
            rows = cuts.loc[[k]]
            for rr in rows.itertuples():
                apex = int(snap[0] + round(rr.t_after_snap * HZ))
                w = window(t.x.values, t.y.values, apex)
                if w is not None:
                    X.append(w); meta.append((k[2], 1, rr.angle, f"{k[0]}_{k[1]}"))
        print("year", yr, "windows", len(X), flush=True)
    m = pd.DataFrame(meta, columns=["nfl_id", "game", "angle", "src"])
    np.savez_compressed(os.path.join(cd, "segs.npz"), X=np.stack(X), nfl_id=m.nfl_id.values, game=m.game.values,
                        angle=m.angle.values, src=m.src.values.astype(str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

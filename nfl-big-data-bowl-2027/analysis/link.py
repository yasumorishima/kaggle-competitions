"""Link each player's drill cuts to his game cuts (same definition, cuts.py).

Fingerprint per player and source: within angle bins, the speed kept through the cut (1 - loss) and the entry speed, and
a single summary = mean residual of loss against the population curve loss(angle) (negative = keeps more speed than
an average player at the same angle). Prints coverage (cuts per player), the monotonic check, and drill-vs-game
correlations (Spearman, with a player bootstrap 90% interval).
Usage: python link.py DATA_DIR CUTS_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BINS = [30, 45, 60, 90, 120, 180]


def residuals(c):
    c = c.copy()
    c["ab"] = pd.cut(c.angle, BINS)
    curve = c.groupby("ab", observed=True)[["loss", "v_in", "decel"]].median()
    c = c.join(curve, on="ab", rsuffix="_pop")
    c["r_loss"] = c.loss - c.loss_pop
    c["r_vin"] = c.v_in - c.v_in_pop
    c["r_decel"] = c.decel - c.decel_pop
    return c


def fingerprint(c, min_cuts):
    g = c.groupby("nfl_id").agg(n=("loss", "size"), r_loss=("r_loss", "mean"), r_vin=("r_vin", "mean"),
                                r_decel=("r_decel", "mean"), sharp=("angle", lambda a: (a >= 90).mean()))
    return g[g.n >= min_cuts]


def boot_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    r = [spearmanr(a[i], b[i])[0] for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return np.nanpercentile(r, [5, 95])


if __name__ == "__main__":
    data, cd = sys.argv[1], sys.argv[2]
    pl = pd.read_csv(os.path.join(data, "players.csv"))
    pos = pl.set_index("nfl_id").nfl_position
    d = pd.read_parquet(os.path.join(cd, "drill_cuts.parquet"))
    d = d[d.drill_type.isin(["SKILL_DRILLS_WR", "SKILL_DRILLS_DB"])]
    g = pd.read_parquet(os.path.join(cd, "game_cuts.parquet"))
    d, g = residuals(d), residuals(g)
    for name, c in (("drill", d), ("game", g)):
        print(name, "angle bins (median loss, v_in, n):")
        print(c.groupby("ab", observed=True).agg(loss=("loss", "median"), v_in=("v_in", "median"), n=("loss", "size")).round(2).to_string())
    for grp, poss in (("WR", ["WR"]), ("DB", ["CB", "SS", "FS", "DB"])):
        fd = fingerprint(d[d.nfl_id.map(pos).isin(poss)], 10)
        fg = fingerprint(g[g.nfl_id.map(pos).isin(poss)], 30)
        j = fd.join(fg, lsuffix="_d", rsuffix="_g", how="inner")
        print(f"\n{grp}: players with >=10 drill cuts {len(fd)}, >=30 game cuts {len(fg)}, both {len(j)}; "
              f"median game cuts {fg.n.median() if len(fg) else 0}")
        for k in ("r_loss", "r_vin", "r_decel", "sharp"):
            if len(j) >= 10:
                r = spearmanr(j[k + "_d"], j[k + "_g"])[0]
                lo, hi = boot_ci(j[k + "_d"], j[k + "_g"])
                print(f"  {k}: drill vs game Spearman {r:+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]")

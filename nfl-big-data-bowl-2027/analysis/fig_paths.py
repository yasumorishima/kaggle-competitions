"""Figures 4 and 5 for the write-up (11-08).

Figure 4 (figs/hoop_paths.png): run-the-hoop tracking of the three fastest and three slowest DL / edge rookies by the
pre-registered feature (active duration of the best attempt, dl_drills.feats): left, the x-y paths coloured by speed;
right, speed against time from the start of movement.
Figure 5 (figs/getoff_shrink.png): each player's mean game get-off (position-adjusted) before and after empirical-Bayes
shrinkage toward the position mean, against his number of plays: the target used in the hierarchical analysis
(hier.py uses the MixedLM BLUP; the closed form tau2 / (tau2 + sigma2 / n) is the same idea and is what is drawn here).
Usage: python fig_paths.py DATA_DIR OUT_DIR
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_drills import HZ, feats      # noqa: E402

DL = ["DE", "DT", "NT", "OLB"]


def main(data, out):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & (c.drill_name == "RUN_THE_HOOP_DRILL")]
    c = c[c.nfl_id.map(pl.nfl_position).isin(DL)]
    best = {}
    for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):
        g = g.sort_values("time")
        f = feats(g.x.values, g.y.values)
        if f and (pid not in best or f["dur"] < best[pid][0]):
            best[pid] = (f["dur"], g.x.values, g.y.values)
    order = sorted(best, key=lambda p: best[p][0])
    pick = [(p, "#4e79a7") for p in order[:3]] + [(p, "#e15759") for p in order[-3:]]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    for p, col in pick:
        dur, x, y = best[p]
        sp = np.hypot(np.gradient(x), np.gradient(y)) * HZ
        mv = np.nonzero(sp > 0.5)[0]
        a, b = mv[0], mv[-1]
        xs, ys = x[a:b + 1] - x[a], y[a:b + 1] - y[a]
        ax[0].plot(xs, ys, color=col, lw=1.4, alpha=0.85)
        t = np.arange(b - a + 1) / HZ
        lab = f"{pl.display_name.get(p, p)} ({pl.nfl_position.get(p, '')}) {dur:.2f} s"
        ax[1].plot(t, sp[a:b + 1], color=col, lw=1.4, alpha=0.85, label=lab)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x from start (yd)")
    ax[0].set_ylabel("y from start (yd)")
    ax[0].set_title("run the hoop: paths (blue = 3 fastest, red = 3 slowest)")
    ax[1].set_xlabel("time from first movement (s)")
    ax[1].set_ylabel("speed (yd/s)")
    ax[1].set_title(f"speed traces; DL / edge rookies with a hoop attempt: n={len(best)}")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hoop_paths.png"), dpi=130)
    plt.close(fig)
    print("fig 4: n", len(best), "fastest", [round(best[p][0], 2) for p in order[:3]],
          "slowest", [round(best[p][0], 2) for p in order[-3:]])

    pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "player_get_off"]).dropna()
    pp["pos"] = pp.nfl_id.map(pl.nfl_position)
    pp = pp[pp.pos.isin(DL)]
    pp["adj"] = pp.player_get_off - pp.groupby("pos").player_get_off.transform("mean")
    g = pp.groupby("nfl_id").adj.agg(["mean", "var", "size"])
    s2 = np.average(g["var"].fillna(0), weights=g["size"] - 1)
    tau2 = max(g["mean"].var() - np.mean(s2 / g["size"]), 1e-6)
    g["k"] = tau2 / (tau2 + s2 / g["size"])
    g["shr"] = g["k"] * g["mean"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for _, r in g.iterrows():
        ax.plot([r["size"], r["size"]], [r["mean"], r["shr"]], color="#bab0ac", lw=0.7, zorder=1)
    ax.scatter(g["size"], g["mean"], s=10, color="#bab0ac", label="raw mean", zorder=2)
    ax.scatter(g["size"], g["shr"], s=12, color="#4e79a7", label="shrunken (empirical Bayes)", zorder=3)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("plays with a recorded get-off (log scale)")
    ax.set_ylabel("game get-off vs position mean (s; lower = faster)")
    ax.set_title(f"DL / edge (n={len(g)} players): ICC {tau2 / (tau2 + s2):.3f}, median shrink factor {g.k.median():.2f}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "getoff_shrink.png"), dpi=130)
    print("fig 5: players", len(g), "ICC", round(tau2 / (tau2 + s2), 3), "median k", round(g.k.median(), 3),
          "k for < 30 plays", round(g.k[g["size"] < 30].median(), 3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

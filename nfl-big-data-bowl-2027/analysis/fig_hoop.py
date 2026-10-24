"""Figure draft for pillar 2: run-the-hoop active duration vs game get-off (DL / edge), with the R2 comparison.
Usage: python fig_hoop.py DATA_DIR OUT_PNG"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402
from scipy.stats import spearmanr    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_drills import feats          # noqa: E402

data, out = sys.argv[1], sys.argv[2]
pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
c = c[(c.entity_type == "PLAYER") & (c.drill_name == "RUN_THE_HOOP_DRILL")]
rows = []
for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):
    g = g.sort_values("time")
    f = feats(g.x.values, g.y.values)
    if f:
        f["nfl_id"] = pid; rows.append(f)
d = pd.DataFrame(rows).sort_values("dur").groupby("nfl_id").first()
pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "player_get_off"]).dropna()
t = pp.groupby("nfl_id").player_get_off.agg(["mean", "size"])
t = t[t["size"] >= 30]
pos = pl.nfl_position.reindex(t.index)
t = t[pos.isin(["DE", "DT", "NT", "OLB"])]
t["adj"] = t["mean"] - t.groupby(pos.reindex(t.index))["mean"].transform("mean")
ids = t.index.intersection(d.index)
x, y, yr = d.loc[ids, "dur"], t.loc[ids, "adj"], pl.draft_year.reindex(ids)
print("n", len(ids), "rho", round(spearmanr(x, y)[0], 3))
for v in sorted(yr.dropna().unique()):
    m = yr == v
    print(int(v), "n", int(m.sum()), "rho", round(spearmanr(x[m], y[m])[0], 3))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
for v, col in zip(sorted(yr.dropna().unique()), ["#4e79a7", "#f28e2b", "#59a14f"]):
    m = yr == v
    ax[0].scatter(x[m], y[m], s=18, color=col, label=f"draft {int(v)}", alpha=0.8)
b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax[0].plot(xs, np.polyval(b, xs), color="#333", lw=1)
ax[0].set_xlabel("run-the-hoop active duration, best attempt (s)")
ax[0].set_ylabel("game get-off, position-adjusted (s; lower = faster)")
ax[0].set_title(f"DL / edge rookies (n={len(ids)}), Spearman {spearmanr(x, y)[0]:+.2f}")
ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(["stopwatch +\ndraft slot", "hoop\nduration", "both"], [-0.006, 0.082, 0.022],
          color=["#bab0ac", "#4e79a7", "#9c755f"])
ax[1].axhline(0, color="#333", lw=0.8)
ax[1].set_ylabel("cross-validated R²")
ax[1].set_title("predicting game get-off")
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)

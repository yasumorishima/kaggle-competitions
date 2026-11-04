"""Pillar 2d: what is behind the 2025 reversal of DT 40-yd v10 -> pressure, and the figure (11-03 next step).

pillar2c (11-03): play-level GEE, DT + NT, small base + z(v10): beta +0.146 [+0.026, +0.265]; by draft year
2023 +0.14, 2024 +0.25, 2025 -0.21.
Here, in the same form as the 2023 check of 10-26 (few players, position mix, chance):
  1) per draft year: players, NT share, median rush snaps, v10 mean / sd, pick, raw pressure rate
  2) 2025 players one by one (v10, snaps, rate, pick) and the leave-one-player-out 2025 coefficient
  3) a permutation of the draft-year labels: how often a random split of the 41 players into groups of the same
     sizes gives a group coefficient as low as the 2025 one (is one year at -0.21 surprising with ~13 players?)
  4) a heterogeneity test: GEE with z_v10 x year interaction (Wald test of the two interaction terms)
Figure figs/dt_pressure_tercile.png: left, pooled pressure rate per v10 tercile (player cluster bootstrap 95% CI);
right, each player's shrunken (beta-binomial EB) rate against v10, coloured by draft year, with the fitted GEE line.
Usage: python pillar2d.py DATA_DIR CACHE_DIR
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pillar2 import FULL, SMALL                                           # noqa: E402
from pillar2c import DT_POS, eb_rate, gee, rush_plays, zcols              # noqa: E402
from trait_translation import drill_table, drill_traits                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(data, cache):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    base = cr[FULL[:-1]].join(np.log(pl.draft_overall_pick).rename("log_pick"), how="outer")
    D, _ = drill_traits(drill_table(data))
    pp = rush_plays(data, pl)
    cnt = pp.groupby("nfl_id").pr.agg(["sum", "size"])
    ids = cnt.index[pl.nfl_position.reindex(cnt.index).isin(DT_POS)]
    ids = ids[D.reindex(ids).D_v10_40.notna().values]
    v = D.reindex(ids).D_v10_40
    S = zcols(base, SMALL, ids)
    X = S.assign(z_v10=(v - v.mean()) / v.std())
    df = pp[pp.nfl_id.isin(ids)].join(X, on="nfl_id")
    yr = pl.draft_year.reindex(ids)
    df["yr"] = yr.reindex(df.nfl_id).values
    s_terms = " + ".join("b_" + c for c in SMALL)
    f = f"pr ~ C(pos) + {s_terms} + z_v10"
    c = cnt.reindex(ids)
    rate, a, b = eb_rate(c["sum"].values.astype(float), c["size"].values.astype(float))
    eb = pd.Series(rate, index=ids)
    raw = c["sum"] / c["size"]
    pick = pl.draft_overall_pick.reindex(ids)

    print("PILLAR 2d: the 2025 reversal of DT v10 -> pressure, and the tercile figure")
    print(f"DT + NT with v10 and rush snaps: {len(ids)} players, {len(df)} snaps")
    print("\n1) by draft year")
    for t in sorted(yr.dropna().unique()):
        k = ids[(yr == t).values]
        d = df[df.nfl_id.isin(k)]
        nt = (pl.nfl_position.reindex(k) == "NT").mean()
        m = gee(f, d)
        print(f"  {int(t)}: players {len(k):2d}  NT share {nt:.2f}  median snaps {int(c.reindex(k)['size'].median()):4d}  "
              f"snaps {len(d):5d}  v10 mean {v.reindex(k).mean():.2f} sd {v.reindex(k).std():.2f}  "
              f"median pick {pick.reindex(k).median():.0f}  pressure {d.pr.mean():.3f}  "
              f"GEE beta {m.params['z_v10']:+.3f} [{m.conf_int().loc['z_v10'][0]:+.3f}, {m.conf_int().loc['z_v10'][1]:+.3f}]")

    k25 = ids[(yr == 2025).values]
    print("\n2) 2025 players (sorted by v10): v10, rush snaps, raw rate, EB rate, pick, position, and the 2025 beta without them")
    d25 = df[df.nfl_id.isin(k25)]
    b25 = gee(f, d25).params["z_v10"]
    for i in v.reindex(k25).sort_values().index:
        try:
            loo = gee(f, d25[d25.nfl_id != i]).params["z_v10"]
        except Exception:   # noqa: BLE001
            loo = np.nan
        print(f"  {pl.display_name.get(i, i)!s:24s} {pl.nfl_position.get(i)}  v10 {v[i]:.2f}  snaps {int(c.loc[i, 'size']):4d}  "
              f"raw {raw[i]:.3f}  EB {eb[i]:.3f}  pick {pick[i]:5.0f}  2025 beta without {loo:+.3f}  (with all {b25:+.3f})")

    print("\n3) permutation of the draft-year labels (1000 reps): a random group of the 2025 size")
    rng = np.random.default_rng(0)
    groups = {i: g for i, g in df.groupby("nfl_id")}
    n25 = len(k25)
    low = []
    for _ in range(1000):
        pick_ids = rng.choice(np.array(ids), n25, replace=False)
        d = pd.concat([groups[i] for i in pick_ids], ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                low.append(gee(f, d).params["z_v10"])
        except Exception:   # noqa: BLE001
            pass
    low = np.array(low)
    print(f"  random {n25}-player groups: beta mean {low.mean():+.3f}  sd {low.std():.3f}  "
          f"share <= 2025's {b25:+.3f}: {np.mean(low <= b25):.3f}  ({len(low)} fits)")

    print("\n4) heterogeneity: z_v10 x draft year (Wald, cluster-robust)")
    m = gee(f"pr ~ C(pos) + {s_terms} + z_v10 * C(yr)", df)
    names = [n for n in m.params.index if n.startswith("z_v10:")]
    R = np.zeros((len(names), len(m.params)))
    for r, n in enumerate(names):
        R[r, list(m.params.index).index(n)] = 1
    w = m.wald_test(R, scalar=True)
    print("  interaction terms: " + "  ".join(f"{n} {m.params[n]:+.3f}" for n in names)
          + f"  Wald chi2 {float(w.statistic):.2f}  p {float(w.pvalue):.3f}")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tert = pd.qcut(v, 3, labels=["low", "mid", "high"])
    df["tert"] = tert.reindex(df.nfl_id).values
    rows = []
    for t in ["low", "mid", "high"]:
        k = tert.index[tert == t]
        d = df[df.nfl_id.isin(k)]
        per = d.groupby("nfl_id").pr.agg(["sum", "size"])
        bs = []
        for _ in range(2000):
            s = per.iloc[rng.integers(0, len(per), len(per))]
            bs.append(s["sum"].sum() / s["size"].sum())
        rows.append((t, len(k), len(d), d.pr.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                     v.reindex(k).min(), v.reindex(k).max()))
    print("\nFIGURE: pooled pressure rate per v10 tercile (player cluster bootstrap 95%)")
    for t, n, s, r, lo, hi, vmin, vmax in rows:
        print(f"  {t:4s} v10 {vmin:.2f}-{vmax:.2f}  players {n:2d}  snaps {s:5d}  rate {r:.3f} [{lo:.3f}, {hi:.3f}]")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(3)
    ax[0].bar(x, [r[3] for r in rows], color=["#adb5bd", "#868e96", "#d9480f"], width=0.6)
    ax[0].errorbar(x, [r[3] for r in rows], yerr=[[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]],
                   fmt="none", ecolor="#212529", capsize=5)
    ax[0].set_xticks(x, [f"{r[0]} v10\n{r[6]:.1f}-{r[7]:.1f} yd/s\n({r[1]} players)" for r in rows])
    ax[0].set_ylabel("pressure per pass-rush snap")
    ax[0].set_title("DT / NT: pressure rate by 40-yd speed at 1.0 s (tercile)", fontsize=10)
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False)
    cols = {2023: "#1c7ed6", 2024: "#2b8a3e", 2025: "#d9480f"}
    for t, col in cols.items():
        k = ids[(yr == t).values]
        ax[1].scatter(v.reindex(k), eb.reindex(k), s=np.sqrt(c.reindex(k)["size"]) * 3, color=col, alpha=0.75,
                      label=f"draft {t}", edgecolor="white")
    mm = gee(f"pr ~ z_v10", df)
    gx = np.linspace(v.min(), v.max(), 50)
    gz = (gx - v.mean()) / v.std()
    ax[1].plot(gx, 1 / (1 + np.exp(-(mm.params["Intercept"] + mm.params["z_v10"] * gz))), color="#212529", lw=1.5,
               label="GEE fit (v10 only)")
    ax[1].set_xlabel("40-yd drill: speed at 1.0 s after the start (yd/s)")
    ax[1].set_ylabel("shrunken pressure rate (beta-binomial EB)")
    ax[1].set_title("each player; dot area = rush snaps", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax[1].spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "..", "figs", "dt_pressure_tercile.png")
    fig.savefig(out, dpi=150)
    print("saved", os.path.normpath(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

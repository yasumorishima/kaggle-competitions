"""Figure for pillar 1 (trait translation): CB and DT 40-yard speed curves -- same stopwatch 10-yd split, different rise.

Row per position (CB = CB + DB, DT = DT + NT, as in trait_translation.py):
  left : tracked 40-yd speed curves (speed vs time from the player's own start of motion, first 2 s) of two players whose
         stopwatch 10-yd splits are nearly the same but whose drill v10 (speed 1.0 s after start, mean over attempts)
         differs most. Thin lines = single attempts, thick = mean over attempts.
  right: drill v10 (D_v10_40) vs game initial burst (G_v10, 80th pct over plays of speed 1.0 s after his own start after
         the snap), with Spearman rho and the partial rho after the stopwatch 10-yd split (and after split + weight, the
         form reported in trait_translation_out.txt). The pair is highlighted.
Pair rule (objective): among players of the analysed set (D_v10_40, G_v10 and ten_yd_split all present), pairs with
|10-yd split difference| <= 0.02 s; pick the pair with the largest |D_v10_40 diff|.
  First run (11-01) of the bare rule picked, for CB, a player whose low v10 came from a long slow creep before the real
  start (0.8 s from 0.5 to 2 yd/s; the stopwatch clock starts later than our tracking onset). So the figure rule also
  requires both players' creep (mean time from 0.5 to 2 yd/s over attempts) <= the group median, so the pair shows a
  difference in the rise itself, not in where time zero falls. Both picks are printed. Robustness of the translation to
  the onset threshold (0.5 / 1 / 2 / 3 yd/s on the drill side) is printed too.
Speed is computed exactly as in trait_translation.py (3-frame smoothing, 10 Hz gradient, start = first frame > 0.5 yd/s).
Usage: python fig_curve.py DATA_DIR CACHE_DIR OUT_PNG
"""
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trait_translation import (GROUP, HZ, drill_table, drill_traits, game_plays, game_traits, partial, rho,  # noqa: E402
                               smooth)

SPLIT_TOL = 0.02
T_MAX = 2.0
COLS = ["#4e79a7", "#f28e2b"]


def speed_curves(c, pid):
    """Per-attempt speed from the start of motion, first T_MAX s (as trait_translation feats on the 40)."""
    out = []
    for _, g in c[c.nfl_id == pid].groupby(["event_id", "attempt"]):
        g = g.sort_values("time")
        xs, ys = smooth(g.x.values), smooth(g.y.values)
        sp = np.hypot(np.gradient(xs), np.gradient(ys)) * HZ
        mv = np.nonzero(sp > 0.5)[0]
        if not len(mv):
            continue
        s = sp[mv[0]:mv[0] + int(T_MAX * HZ) + 1]
        out.append(s)
    return out


def onset_table(c):
    """Per player (mean over attempts): creep = time from 0.5 to 2 yd/s; v10 with drill onset thresholds 1 / 2 / 3."""
    rows = []
    for (pid, _, _), g in c.groupby(["nfl_id", "event_id", "attempt"]):
        g = g.sort_values("time")
        sp = np.hypot(np.gradient(smooth(g.x.values)), np.gradient(smooth(g.y.values))) * HZ
        r = {"nfl_id": pid}
        m05, m2 = np.nonzero(sp > 0.5)[0], np.nonzero(sp > 2.0)[0]
        if len(m05) and len(m2):
            r["creep"] = (m2[0] - m05[0]) / HZ
        for th in (1.0, 2.0, 3.0):
            mv = np.nonzero(sp > th)[0]
            if len(mv) and len(sp) > mv[0] + 10:
                r[f"v10_on{th:.0f}"] = sp[mv[0] + 10]
        rows.append(r)
    return pd.DataFrame(rows).groupby("nfl_id").mean()


def pick(kk, ok):
    best, n = None, 0
    for a, b in itertools.combinations([i for i in kk.index if ok[i]], 2):
        if abs(kk.ten_yd_split[a] - kk.ten_yd_split[b]) <= SPLIT_TOL + 1e-9:
            n += 1
            dv = abs(kk.D_v10_40[a] - kk.D_v10_40[b])
            if best is None or dv > best[0]:
                best = (dv, a, b)
    return best, n


def main(data, cache, out_png):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    grp = pl.nfl_position.map(GROUP)
    D, _ = drill_traits(drill_table(data))
    G, _ = game_traits(game_plays(data, cache), grp)
    J = D.join(G, how="inner").join(grp.rename("grp")).join(pl[["display_name", "draft_year"]]).join(
        cr[["ten_yd_split", "forty", "combine_weight"]])
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"),
                    usecols=["event_id", "nfl_id", "entity_type", "time", "drill_name", "attempt", "x", "y"])
    c = c[(c.entity_type == "PLAYER") & (c.drill_name == "FORTY_YARD_DASH")]
    J = J.join(onset_table(c))

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4), gridspec_kw={"width_ratios": [3, 2]})
    print(f"pair rule: among players with drill v10, game v10 and a stopwatch 10-yd split, pairs with |split diff| <= "
          f"{SPLIT_TOL:.2f} s and both players' pre-start creep (0.5->2 yd/s) <= group median; choose the pair with the "
          f"largest |drill v10 diff|  (the bare rule without the creep filter is printed too)")
    for row, g in enumerate(["CB", "DT"]):
        k = J[J.grp == g]
        s = k[["D_v10_40", "G_v10"]].dropna()
        r_all = rho(s.D_v10_40.values, s.G_v10.values)
        kk = k[["D_v10_40", "G_v10", "ten_yd_split", "combine_weight", "display_name", "draft_year"]].dropna(
            subset=["D_v10_40", "G_v10", "ten_yd_split"])
        p_split = partial(kk.D_v10_40.values, kk.G_v10.values, kk[["ten_yd_split"]].values)
        kw = kk.dropna(subset=["combine_weight"])
        p_sw = partial(kw.D_v10_40.values, kw.G_v10.values, kw[["ten_yd_split", "combine_weight"]].values)
        r_split = rho(kk.ten_yd_split.values, kk.G_v10.values)
        print(f"\n{g}: n {len(s)}  Spearman drill v10 -> game v10 {r_all:+.2f};  stopwatch split -> game v10 "
              f"{r_split:+.2f} (n {len(kk)});  partial | split {p_split:+.2f};  partial | split, weight {p_sw:+.2f} "
              f"(n {len(kw)})")
        kk = kk.join(J[["creep", "v10_on1", "v10_on2", "v10_on3"]])
        for col in ("v10_on1", "v10_on2", "v10_on3"):
            q = kk[[col, "G_v10"]].dropna()
            print(f"  onset robustness: drill {col} -> game v10 rho {rho(q[col].values, q.G_v10.values):+.2f} (n {len(q)})")
        q = kk[["creep", "G_v10"]].dropna()
        print(f"  creep (0.5->2 yd/s) median {kk.creep.median():.2f} s; creep -> game v10 rho "
              f"{rho(q.creep.values, q.G_v10.values):+.2f}")
        bare, nb = pick(kk, {i: True for i in kk.index})
        print(f"  bare rule ({nb} candidate pairs): " + "  vs  ".join(
            f"{kk.display_name[i]} split {kk.ten_yd_split[i]:.2f} v10 {kk.D_v10_40[i]:.2f} creep {kk.creep[i]:.2f}"
            for i in bare[1:]))
        ok = (kk.creep <= kk.creep.median()).to_dict()
        best, n_cand = pick(kk, ok)
        _, a, b = best
        pair = sorted([a, b], key=lambda i: -kk.D_v10_40[i])  # faster riser first
        print(f"  figure rule: candidate pairs {n_cand}; chosen (largest drill v10 gap {best[0]:.2f} yd/s):")
        for i in pair:
            r = kk.loc[i]
            gr = (kk.G_v10 <= r.G_v10).mean()
            print(f"    {r.display_name} ({int(r.draft_year)})  10-yd split {r.ten_yd_split:.2f} s  forty "
                  f"{J.forty[i]:.2f}  drill v10 {r.D_v10_40:.2f} yd/s  game v10 {r.G_v10:.2f} yd/s "
                  f"(pct in group {100 * gr:.0f})  creep {r.creep:.2f} s")
        a0 = ax[row, 0]
        for i, col in zip(pair, COLS):
            cur = speed_curves(c, i)
            L = min(len(u) for u in cur)
            t = np.arange(L) / HZ
            for u in cur:
                a0.plot(np.arange(len(u)) / HZ, u, color=col, lw=0.8, alpha=0.35)
            m = np.mean([u[:L] for u in cur], axis=0)
            r = kk.loc[i]
            a0.plot(t, m, color=col, lw=2.2,
                    label=f"{r.display_name}: 10-yd {r.ten_yd_split:.2f} s, v@1.0 s {r.D_v10_40:.2f} yd/s, "
                          f"game burst {r.G_v10:.2f}")
            a0.plot([1.0], [r.D_v10_40], "o", color=col, ms=6, mec="white")
        a0.axvline(1.0, color="#999", lw=0.8, ls=":")
        a0.set_xlim(0, T_MAX)
        a0.set_xlabel("time from first movement (> 0.5 yd/s) in the 40-yd dash (s)")
        a0.set_ylabel("tracked speed (yd/s)")
        a0.set_title(f"{g}: same stopwatch 10-yd split, different rise")
        a0.legend(frameon=False, fontsize=7.5, loc="lower right")
        a1 = ax[row, 1]
        for v, col in zip([2023, 2024, 2025], ["#4e79a7", "#f28e2b", "#59a14f"]):
            m = J.loc[s.index, "draft_year"] == v
            a1.scatter(s.D_v10_40[m], s.G_v10[m], s=16, color=col, alpha=0.75, label=f"draft {v}")
        for i in pair:
            a1.scatter([kk.D_v10_40[i]], [kk.G_v10[i]], s=70, facecolor="none", edgecolor="#333", lw=1.4)
        bb = np.polyfit(s.D_v10_40, s.G_v10, 1)
        xs = np.linspace(s.D_v10_40.min(), s.D_v10_40.max(), 50)
        a1.plot(xs, np.polyval(bb, xs), color="#333", lw=1)
        a1.set_xlabel("drill: 40-yd speed at 1.0 s (yd/s)")
        a1.set_ylabel("game: speed 1.0 s after start (yd/s)")
        a1.set_title(f"{g} (n={len(s)}): Spearman {r_all:+.2f}\nbeyond 10-yd split {p_split:+.2f}; "
                     f"split+weight {p_sw:+.2f}; split alone {r_split:+.2f}", fontsize=8.5)
        a1.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print("\nwrote", out_png)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

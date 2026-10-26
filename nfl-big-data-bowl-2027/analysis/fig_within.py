"""Figures for the within-situation result of hier.py (run-the-hoop duration -> game get-off, DL / edge rookies, n=106).

fig 1  within_forest.png : forest plot of Spearman rho with player-bootstrap 95% CIs
         - pooled (all plays, player BLUP as in hier.py)
         - within each play situation (position-adjusted mean get-off of the player's plays in that situation, >= 10 plays)
         - by draft class (BLUP target)
fig 2  within_multiples.png : small multiples, one scatter per situation, with per-panel rho and n.
Positive rho = a faster (shorter) hoop goes with a quicker (shorter) get-off.
Usage: python fig_within.py DATA_DIR CACHE_DIR OUT_DIR
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402
from scipy.stats import spearmanr    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus import base_table                    # noqa: E402
from hier import FLOOR, blup, hoop_table, plays  # noqa: E402

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
B = 4000
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


def rho_ci(x, y, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    bs = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return spearmanr(x, y)[0], lo, hi, len(x)


SITS = [("Pass plays", lambda d: d.pass_play == 1), ("Run plays", lambda d: d.pass_play == 0),
        ("Pass from shotgun", lambda d: (d.pass_play == 1) & (d.shotgun == 1)),
        ("1st & 2nd down", lambda d: d.down <= 2), ("3rd & 4th down", lambda d: d.down >= 3)]


def main(data, cache, out):
    os.makedirs(out, exist_ok=True)
    pl, base = base_table(data)
    d = hoop_table(data, cache)
    pp = plays(data, cache, pl)
    n = pp.groupby("nfl_id").size()
    ids = n.index[n >= FLOOR].intersection(d.index).intersection(base.index)
    df = pp[pp.nfl_id.isin(ids)]
    pos = pl.nfl_position.reindex(ids)
    yr = pl.draft_year.reindex(ids)
    x = d.hoop_dur.reindex(ids)
    y = blup(df, False)[0].reindex(ids)

    rows, scat = [], {}
    rows.append(("All plays (pooled)", "pooled") + rho_ci(x, y))
    for lab, f in SITS:
        t = df[f(df)].groupby("nfl_id").player_get_off.agg(["mean", "size"])
        t = t[t["size"] >= 10]
        i = t.index.intersection(ids)
        a = t.loc[i, "mean"] - t.loc[i, "mean"].groupby(pos.reindex(i)).transform("mean")
        rows.append((lab, "situation") + rho_ci(x[i], a))
        scat[lab] = (x[i], a)
    for v in (2023, 2024, 2025):
        k = yr == v
        rows.append((f"{v} draft class", "year") + rho_ci(x[k], y[k]))
    R = pd.DataFrame(rows, columns=["label", "group", "rho", "lo", "hi", "n"])
    print(R.round(3).to_string(index=False))
    R.to_csv(os.path.join(cache, "within_forest.csv"), index=False)

    # ---------- fig 1: forest ----------
    order = [("pooled", None), ("situation", "Within a situation"), ("year", "By draft class")]
    ypos, labels, ycur, heads = [], [], 0, []
    for g, head in order:
        if head:
            ycur += 0.6
            heads.append((ycur, head)); ycur += 1
        for _, r in R[R.group == g].iterrows():
            ypos.append((ycur, r)); ycur += 1
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 2.6, 1.9], wspace=0.04, left=0.03, right=0.98, top=0.83, bottom=0.09)
    axl, ax, axr = (fig.add_subplot(gs[0, i]) for i in range(3))
    for a_ in (axl, axr):
        a_.set_xlim(0, 1); a_.axis("off")
    for yy, r in ypos:
        col = {"pooled": INK, "situation": BLUE, "year": ORANGE}[r.group]
        ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=2.2, solid_capstyle="round")
        mk = dict(marker="D", s=110) if r.group == "pooled" else dict(s=70)
        ax.scatter([r.rho], [yy], color=col, zorder=3, edgecolor=SURF, linewidth=2, **mk)
        axl.text(0, yy, r.label, ha="left", va="center", color=INK, fontsize=12,
                 fontweight="bold" if r.group == "pooled" else "normal")
        axr.text(0.03, yy, f"{r.rho:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]", ha="left", va="center",
                 color=INK2, fontsize=10, family="DejaVu Sans Mono")
        axr.text(1.0, yy, f"{int(r.n)}", ha="right", va="center", color=INK2, fontsize=10, family="DejaVu Sans Mono")
    for yy, h in heads:
        axl.text(0, yy, h.upper(), ha="left", va="center", color=MUTED, fontsize=9.5, fontweight="bold")
    axr.text(0.03, -0.9, "ρ [95% CI]", ha="left", va="center", color=MUTED, fontsize=9.5, fontweight="bold")
    axr.text(1.0, -0.9, "n", ha="right", va="center", color=MUTED, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(R.rho[0], color=INK, lw=1, ls=(0, (2, 3)), alpha=0.6)
    ax.set_xlim(-0.35, 0.8)
    for a_ in (axl, ax, axr):
        a_.set_ylim(ycur, -1.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.4, 0.8])
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Spearman ρ")
    fig.text(0.03, 0.975, "Faster through the hoop, quicker off the ball", ha="left", va="top",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.03, 0.925, f"Run-the-hoop time at the Combine vs in-game get-off, DL / edge rookies (2023–25 drafts,\n"
             f"n={len(ids)}). Get-off is position-adjusted; positive ρ = shorter hoop time goes with a\n"
             f"quicker get-off. Bars: player-bootstrap 95% CI. Dotted line: pooled ρ.", ha="left", va="top",
             fontsize=10, color=INK2, linespacing=1.35)
    p1 = os.path.join(out, "within_forest.png")
    fig.savefig(p1, dpi=200)
    print("wrote", p1)

    # ---------- fig 2: small multiples ----------
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 9.0), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, (lab, _) in zip(axs, SITS):
        xx, aa = scat[lab]
        r = R[R.label == lab].iloc[0]
        ax.scatter(xx, aa, s=26, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=0.8)
        b = np.polyfit(xx, aa, 1)
        xs = np.linspace(xx.min(), xx.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=INK, lw=2)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_title(lab, loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.text(0.03, 0.95, f"ρ {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]\nn {int(r.n)}", transform=ax.transAxes,
                va="top", fontsize=10, color=INK2)
        for s in ("right", "top"):
            ax.spines[s].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax = axs[-1]
    ax.axis("off")
    ax.text(0.0, 0.85, "How to read", fontsize=12, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.70, "Each dot is one rookie.\nRight = slower hoop.\nUp = slower get-off than\n"
            "position peers in that\nsituation. Line: OLS fit.", fontsize=10.5, color=INK2, transform=ax.transAxes, va="top")
    for a in axs[[2, 4]]:
        a.set_ylabel("get-off vs position (s)")
    for a in axs[[3, 4]]:
        a.set_xlabel("run-the-hoop time (s)")
    axs[3].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle("Hoop time vs get-off, within each play situation", x=0.03, ha="left", fontsize=14,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p2 = os.path.join(out, "within_multiples.png")
    fig.savefig(p2, dpi=200)
    print("wrote", p2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

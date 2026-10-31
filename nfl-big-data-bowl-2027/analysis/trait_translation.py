"""Trait translation: which physical traits measured at the combine carry into the same trait in NFL games, per position.

Reframed pillar 1 (10-31). Instead of identifying a player from his drills, measure the same physical quantity on both
sides with the same code (positions smoothed with a 3-frame mean, speed from the smoothed positions, 10 Hz):
  vmax  top speed
  v10   speed 1.0 s after the player's own start of motion (start = first frame > 0.5 yd/s)      ["burst"]
  amax  peak acceleration over 0.3 s within the first 2 s of motion
  dmax  peak deceleration over 0.3 s from a speed >= 3 yd/s                                   ["brake"]

DRILL traits per player (denoised: several attempts / several drills pooled)
  D_vmax40   40-yd dash tracking top speed (first 40 yd only), mean over attempts
  D_v10_40   40-yd dash speed 1.0 s after start, mean over attempts
  D_acc      composite: amax z-scored within each drill (player mean per drill), averaged over ALL his drills
  D_dec      composite: dmax z-scored within each drill, averaged over all non-40 drills (shuttle, 3-cone, position drills)
GAME traits per player (per-play value from the snap, 6 s window; player value = 80th percentile over plays, i.e. his
intended effort rather than every jog; >= 30 plays, >= 10 for the matched subsets)
  G_vmax  WR/TE: GO/POST/CORNER/WHEEL routes only; others all plays (90th percentile)
  G_v10   all plays
  G_amax  all plays
  G_dec   WR/TE: HITCH/OUT/IN routes (decel into a break); DBs: plays opened in a back-pedal; others all plays

PRE-STATED PRIMARY HYPOTHESES (fixed on 10-31 before this script was first run; one pair per position group).
They are informed by the earlier exploratory scripts (burst.py 10-21: 40-yd v1.5 vs game, route_segs 10-28: GO-route top
speed +0.34), so they are confirmatory for the new denoised definitions and the draft-class split, not "blind":
  WR    D_vmax40 -> G_vmax  (deep routes)       expect > 0
  TE    D_vmax40 -> G_vmax  (deep routes)       expect > 0
  CB    D_v10_40 -> G_v10                        expect > 0
  S     D_v10_40 -> G_v10                        expect > 0
  EDGE  D_v10_40 -> G_v10   (DE + OLB)           expect > 0
  DT    D_v10_40 -> G_v10   (DT + NT)            expect > 0
  OL    D_acc    -> G_amax  (T, G, C)            expect > 0
All 4 x 4 drill x game pairs x 7 groups (= 112 pairs) are also reported, with Benjamini-Hochberg q over all 112, so the
matched (diagonal) pairs can be compared to the unmatched ones (a trait that "translates" should beat the off-diagonal).

Uncertainty: Spearman rho (and "wcls": ranks within draft class, to catch class-level shifts in either side) with a 2,000-sample player bootstrap 90% CI; replication by draft class (2023 / 2024 / 2025 =
disjoint players); reliability: game odd/even-play split, drill attempt-1 vs attempt-2 (40-yd) or odd/even drill halves
(composites), Spearman-Brown; disattenuated rho = rho / sqrt(rel_d * rel_g) (only when both > 0.2). For the primaries also
the partial rho after the stopwatch time (forty for vmax, ten_yd_split for v10, three_cone for composites) and weight.

Usage: python trait_translation.py DATA_DIR CACHE_DIR > trait_translation_out.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

HZ = 10
GROUP = {"WR": "WR", "TE": "TE", "CB": "CB", "DB": "CB", "SS": "S", "FS": "S", "DE": "EDGE", "OLB": "EDGE",
         "DT": "DT", "NT": "DT", "T": "OL", "G": "OL", "C": "OL"}
GROUPS = ["WR", "TE", "CB", "S", "EDGE", "DT", "OL"]
DEEP = {"GO", "POST", "CORNER", "WHEEL"}
BREAK = {"HITCH", "OUT", "IN"}
DT_ = ["D_vmax40", "D_v10_40", "D_acc", "D_dec"]
GT_ = ["G_vmax", "G_v10", "G_amax", "G_dec"]
PRIMARY = {"WR": ("D_vmax40", "G_vmax"), "TE": ("D_vmax40", "G_vmax"), "CB": ("D_v10_40", "G_v10"),
           "S": ("D_v10_40", "G_v10"), "EDGE": ("D_v10_40", "G_v10"), "DT": ("D_v10_40", "G_v10"),
           "OL": ("D_acc", "G_amax")}
STOPWATCH = {"D_vmax40": "forty", "D_v10_40": "ten_yd_split", "D_acc": "three_cone", "D_dec": "three_cone"}
RNG = np.random.default_rng(0)


def smooth(a, k=3):
    a = np.asarray(a, float)
    if len(a) < k:
        return a
    return np.convolve(np.pad(a, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def feats(x, y, start=0, stop=None, o=None):
    """Physical features of one track from frame `start` (snap / attempt start)."""
    xs, ys = smooth(x), smooth(y)
    vx, vy = np.gradient(xs) * HZ, np.gradient(ys) * HZ
    sp = np.hypot(vx, vy)
    stop = len(sp) if stop is None else min(stop, len(sp))
    s = sp[start:stop]
    if len(s) < 15:
        return None
    mv = np.nonzero(s > 0.5)[0]
    if not len(mv) or mv[0] > 20:
        return None
    m0 = mv[0]
    r = {"vmax": s.max(), "v10": s[m0 + 10] if len(s) > m0 + 10 else np.nan}
    a = s[m0:m0 + 20]
    r["amax"] = ((a[3:] - a[:-3]) / 0.3).max() if len(a) > 4 else np.nan
    d = (s[:-3] - s[3:]) / 0.3
    d = d[s[:-3] >= 3.0]
    r["dmax"] = d.max() if len(d) else np.nan
    if o is not None:
        a0, b0 = start + m0 + 1, min(stop, start + m0 + 11)
        hd = np.degrees(np.arctan2(vx[a0:b0], vy[a0:b0])) % 360
        gap = np.abs((np.asarray(o, float)[a0:b0] - hd + 180) % 360 - 180)
        r["back"] = float(np.nanmean(gap)) if b0 > a0 else np.nan
    if r["vmax"] > 12.5 or (r["amax"] == r["amax"] and r["amax"] > 15):  # tracking glitch
        return None
    return r


# ---------------------------------------------------------------- drills
def drill_table(data):
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"),
                    usecols=["event_id", "nfl_id", "entity_type", "time", "drill_type", "drill_name", "attempt", "x",
                             "y"])
    c = c[c.entity_type == "PLAYER"].sort_values(["nfl_id", "event_id", "drill_name", "attempt", "time"])
    rows = []
    for (pid, eid, dn, att), g in c.groupby(["nfl_id", "event_id", "drill_name", "attempt"], sort=False):
        x, y = g.x.values, g.y.values
        stop = None
        if dn == "FORTY_YARD_DASH":  # cut at 40 yd of travel from the start of motion
            sp = np.hypot(np.diff(smooth(x)), np.diff(smooth(y)))
            mv = np.nonzero(sp * HZ > 0.5)[0]
            if not len(mv):
                continue
            cum = np.cumsum(sp[mv[0]:])
            if cum[-1] < 35:  # tracks end at ~40 yd (median 40.1); keep the whole track when it stops just short
                continue
            k = np.nonzero(cum >= 40)[0]
            stop = mv[0] + k[0] + 2 if len(k) else None
        r = feats(x, y, 0, stop)
        if r is None:
            continue
        if dn == "FORTY_YARD_DASH":
            r["dmax"] = np.nan  # deceleration after the line is not a trait
        r.update(nfl_id=pid, drill=dn, att=att)
        rows.append(r)
    return pd.DataFrame(rows)


def drill_traits(dt):
    f = dt[dt.drill == "FORTY_YARD_DASH"].sort_values(["nfl_id", "att"])
    out = f.groupby("nfl_id")[["vmax", "v10"]].mean().rename(columns={"vmax": "D_vmax40", "v10": "D_v10_40"})
    # attempt 1 vs 2 halves for reliability
    f = f.assign(k=f.groupby("nfl_id").cumcount())
    half40 = {h: f[f.k == h].set_index("nfl_id")[["vmax", "v10"]].rename(
        columns={"vmax": "D_vmax40", "v10": "D_v10_40"}) for h in (0, 1)}
    pm = dt.groupby(["nfl_id", "drill"])[["amax", "dmax"]].mean().reset_index()
    for col in ("amax", "dmax"):
        g = pm.groupby("drill")[col]
        pm[col + "_z"] = (pm[col] - g.transform("mean")) / g.transform("std")
    pm = pm.sort_values(["nfl_id", "drill"])
    pm["half"] = pm.groupby("nfl_id").cumcount() % 2
    nf = pm[pm.drill != "FORTY_YARD_DASH"]
    out = out.join(pm.groupby("nfl_id").amax_z.mean().rename("D_acc"), how="outer")
    out = out.join(nf.groupby("nfl_id").dmax_z.mean().rename("D_dec"), how="outer")
    out = out.join(pm.groupby("nfl_id").size().rename("n_drills"), how="outer")
    halves = {}
    for h in (0, 1):
        hh = half40[h].join(pm[pm.half == h].groupby("nfl_id").amax_z.mean().rename("D_acc"), how="outer")
        hh = hh.join(nf[nf.half == h].groupby("nfl_id").dmax_z.mean().rename("D_dec"), how="outer")
        halves[h] = hh
    return out, halves


# ---------------------------------------------------------------- games
def game_plays(data, cache):
    path = os.path.join(cache, "tt_game_plays.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id").nfl_position
    keep = set(pl.index[pl.isin(GROUP.keys())])
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["game_id", "play_id", "nfl_id", "route_ran", "pass_result", "play_nullified_by_penalty"])
    pp = pp[pp.nfl_id.isin(keep)].drop_duplicates(["game_id", "play_id", "nfl_id"])
    rows = []
    for yr in (2023, 2024, 2025):
        parts = []
        for ch in pd.read_csv(os.path.join(data, f"game_tracking_{yr}.csv"), chunksize=4_000_000,
                              usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "o", "event"]):
            parts.append(ch[ch.nfl_id.isin(keep)])
        g = pd.concat(parts, ignore_index=True).sort_values(["game_id", "play_id", "nfl_id", "time"])
        del parts
        key = g[["game_id", "play_id", "nfl_id"]].values
        brk = np.r_[0, np.nonzero((np.diff(key, axis=0) != 0).any(1))[0] + 1, len(g)]
        x, y, o = g.x.values, g.y.values, g.o.values
        snap = np.isin(g.event.values, ["ball_snap", "snap_direct"])
        del g
        for i0, i1 in zip(brk[:-1], brk[1:]):
            sn = np.nonzero(snap[i0:i1])[0]
            if not len(sn):
                continue
            s0 = sn[0]
            r = feats(x[i0:i1], y[i0:i1], s0, s0 + 60, o[i0:i1])
            if r is None:
                continue
            r.update(game_id=key[i0, 0], play_id=key[i0, 1], nfl_id=key[i0, 2], season=yr)
            rows.append(r)
        print("season file", yr, "plays", len(rows), file=sys.stderr, flush=True)
    gp = pd.DataFrame(rows).merge(pp, on=["game_id", "play_id", "nfl_id"], how="left")
    gp = gp[gp.play_nullified_by_penalty.fillna("N") != "Y"].drop(columns="play_nullified_by_penalty")
    gp.to_parquet(path)
    return gp


def game_traits(gp, grp):
    gp = gp.sort_values(["nfl_id", "game_id", "play_id"]).copy()
    gp["grp"] = gp.nfl_id.map(grp)
    gp["half"] = gp.groupby("nfl_id").cumcount() % 2  # odd/even plays in time order
    gp["route"] = gp.route_ran.fillna("")
    recv = gp.grp.isin(["WR", "TE"])
    db = gp.grp.isin(["CB", "S"])
    subsets = {
        "G_vmax": (np.where(recv, gp.route.isin(DEEP), True), "vmax", 0.9),
        "G_v10": (np.ones(len(gp), bool), "v10", 0.8),
        "G_amax": (np.ones(len(gp), bool), "amax", 0.8),
        "G_dec": (np.where(recv, gp.route.isin(BREAK), np.where(db, gp.back > 110, True)), "dmax", 0.8),
    }
    full, halves = {}, {0: {}, 1: {}}
    for name, (mask, col, q) in subsets.items():
        s = gp[mask & gp[col].notna()]
        n = s.groupby("nfl_id").size()
        minn = 10 if name in ("G_vmax", "G_dec") else 30
        ok = n.index[n >= minn]
        s = s[s.nfl_id.isin(ok)]
        full[name] = s.groupby("nfl_id")[col].quantile(q)
        for h in (0, 1):
            halves[h][name] = s[s.half == h].groupby("nfl_id")[col].quantile(q)
    out = pd.DataFrame(full)
    out["n_plays"] = gp.groupby("nfl_id").size()
    return out, {h: pd.DataFrame(v) for h, v in halves.items()}


# ---------------------------------------------------------------- stats
def rho(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def boot_ci(a, b, B=2000):
    n = len(a)
    idx = RNG.integers(0, n, (B, n))
    ra, rb = rankdata(a[idx], axis=1), rankdata(b[idx], axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    r = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.nanpercentile(r, [5, 95])


def p_rho(r, n):
    z = np.arctanh(np.clip(r, -0.999, 0.999)) * np.sqrt(max(n - 3, 1) / 1.06)
    return 2 * norm.sf(abs(z))


def reliability(h0, h1, col, ids, grp=None):
    """Split-half Spearman-Brown on the players `ids`. If fewer than 15 of them have both halves and `grp` (player ->
    group) is given, fall back to all players with both halves, ranked within their group (pooled within-group)."""
    j = pd.concat([h0[col].rename("a"), h1[col].rename("b")], axis=1).dropna()
    jj = j.reindex(ids).dropna()
    if len(jj) >= 15:
        r = rho(jj.a.values, jj.b.values)
        return (2 * r / (1 + r) if r > -1 else np.nan), len(jj)
    if grp is None:
        return np.nan, len(jj)
    j = j.join(grp.rename("g")).dropna()
    j = j.groupby("g").filter(lambda t: len(t) >= 5)
    if len(j) < 15:
        return np.nan, len(j)
    a = j.groupby("g").a.rank(pct=True); b = j.groupby("g").b.rank(pct=True)
    r = np.corrcoef(a - a.groupby(j.g).transform("mean"), b - b.groupby(j.g).transform("mean"))[0, 1]
    return 2 * r / (1 + r), -len(j)  # negative n marks the pooled fallback


def partial(a, b, Z):
    def res(v):
        A = np.column_stack([np.ones(len(v))] + [rankdata(z) for z in Z.T])
        rv = rankdata(v)
        return rv - A @ np.linalg.lstsq(A, rv, rcond=None)[0]
    return np.corrcoef(res(a), res(b))[0, 1]


def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    q = np.empty(m); q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main(data, cache):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    grp = pl.nfl_position.map(GROUP)
    dt = drill_table(data)
    D, Dh = drill_traits(dt)
    gp = game_plays(data, cache)
    G, Gh = game_traits(gp, grp)
    J = D.join(G, how="inner").join(grp.rename("grp")).join(pl.draft_year).join(
        cr[["forty", "ten_yd_split", "three_cone", "combine_weight"]])
    print("TRAIT TRANSLATION  combine drill trait -> same trait in NFL games (rookie classes 2023-25)")
    print(f"drill attempts {len(dt)}  players with drill traits {len(D)}  game player-plays {len(gp)}  "
          f"players with game traits {len(G)}  joined {len(J)}")
    print("players per group (any trait):", J.grp.value_counts().reindex(GROUPS).to_dict())
    print("rho = Spearman, CI = 90% player bootstrap; wcls = rho with ranks taken within draft class; years = rho per draft class 2023/2024/2025 (n);"
          " relD/relG = Spearman-Brown split-half (within the group; if <15 players have 2 halves, pooled within-group over all groups); rho* = disattenuated\n")

    rows = []
    for g in GROUPS:
        k = J[J.grp == g]
        for d in DT_:
            for t in GT_:
                s = k[[d, t, "draft_year"]].dropna()
                n = len(s)
                if n < 15:
                    continue
                r = rho(s[d].values, s[t].values)
                lo, hi = boot_ci(s[d].values, s[t].values)
                yrs = []
                for y in (2023, 2024, 2025):
                    sy = s[s.draft_year == y]
                    yrs.append((rho(sy[d].values, sy[t].values) if len(sy) >= 8 else np.nan, len(sy)))
                wy = s.groupby("draft_year")[[d, t]].rank(pct=True)
                rwy = np.corrcoef(wy[d], wy[t])[0, 1]  # ranks within draft class: guards against class-level shifts
                rd, nd = reliability(Dh[0], Dh[1], d, s.index, grp)
                rg, ng = reliability(Gh[0], Gh[1], t, s.index, grp)
                dis = r / np.sqrt(rd * rg) if (rd > 0.2 and rg > 0.2) else np.nan
                rows.append(dict(grp=g, drill=d, game=t, n=n, rho=r, lo=lo, hi=hi, rwy=rwy, p=p_rho(r, n),
                                 y23=yrs[0][0], y24=yrs[1][0], y25=yrs[2][0], n23=yrs[0][1], n24=yrs[1][1],
                                 n25=yrs[2][1], relD=rd, relG=rg, dis=dis,
                                 matched=DT_.index(d) == GT_.index(t), primary=PRIMARY[g] == (d, t)))
    R = pd.DataFrame(rows)
    R["q"] = bh(R.p.values)
    R["same_sign_yrs"] = [int(np.nansum(np.sign([a, b, c]) == np.sign(r)))
                          for a, b, c, r in zip(R.y23, R.y24, R.y25, R.rho)]

    def line(r):
        return (f"{r.grp:5s} {r.drill:9s}-> {r.game:7s} n {r.n:3d}  rho {r.rho:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]  "
                f"wcls {r.rwy:+.2f}  p {r.p:.3f} q {r.q:.3f}  yrs {r.y23:+.2f}({r.n23}) {r.y24:+.2f}({r.n24}) {r.y25:+.2f}({r.n25})  "
                f"relD {r.relD:.2f} relG {r.relG:.2f} rho* {r.dis:+.2f}")

    print("== 1. PRE-STATED PRIMARY (one per group; 7 tests -> Bonferroni alpha 0.05/7 = 0.007)")
    for _, r in R[R.primary].iterrows():
        print(line(r))
        d, t = PRIMARY[r.grp]
        k = J[(J.grp == r.grp)][[d, t, STOPWATCH[d], "combine_weight"]].dropna()
        if len(k) >= 15:
            sw = STOPWATCH[d]
            Z = k[[sw, "combine_weight"]].values
            pr = partial(k[d].values, k[t].values, Z)
            bs = []
            for _ in range(1000):
                i = RNG.integers(0, len(k), len(k))
                bs.append(partial(k[d].values[i], k[t].values[i], Z[i]))
            plo, phi = np.nanpercentile(bs, [5, 95])
            print(f"      stopwatch {sw} -> {t}: rho {rho(k[sw].values, k[t].values):+.2f} (n {len(k)});  tracking beyond "
                  f"stopwatch: partial rho (drill | {sw}, weight) {pr:+.2f} [{plo:+.2f},{phi:+.2f}]")
    prim = R[R.primary]
    print(f"   primaries significant at Bonferroni 0.007: {int((prim.p < 0.05 / 7).sum())}/7;  "
          f"CI above 0: {int((prim.lo > 0).sum())}/7;  same sign in all 3 classes: {int((prim.same_sign_yrs == 3).sum())}/7")

    print(f"\n== 2. ALL PAIRS TESTED: {len(R)} (4 drill x 4 game traits x 7 groups, minus n<15). "
          f"BH q < 0.10: {int((R.q < 0.10).sum())}; CI above 0: {int((R.lo > 0).sum())}; CI below 0: {int((R.hi < 0).sum())}")
    for _, r in R[(R.q < 0.10) | R.primary].sort_values("rho", ascending=False).iterrows():
        print(("*" if r.primary else " ") + ("m " if r.matched else "  ") + line(r))

    print("\n== 3. MATCHED (same trait both sides) vs UNMATCHED pairs: mean rho per group")
    for g in GROUPS:
        k = R[R.grp == g]
        if len(k):
            print(f"  {g:5s} matched {k[k.matched].rho.mean():+.2f} (n pairs {k.matched.sum()})  unmatched "
                  f"{k[~k.matched].rho.mean():+.2f} ({(~k.matched).sum()})   best matched: "
                  + ", ".join(f"{a[2:]}{b:+.2f}" for a, b in k[k.matched].sort_values('rho', ascending=False)[["game", "rho"]].values[:2]))

    print("\n== 4. MATCHED-PAIR GRID (rho, CI) per group x trait")
    hdr = "        " + "".join(f"{t:>22s}" for t in ["vmax40>vmax", "v10_40>v10", "acc>amax", "dec>dec"])
    print(hdr)
    for g in GROUPS:
        cells = []
        for d, t in zip(DT_, GT_):
            r = R[(R.grp == g) & (R.drill == d) & (R.game == t)]
            cells.append(f"{r.rho.iloc[0]:+.2f}[{r.lo.iloc[0]:+.2f},{r.hi.iloc[0]:+.2f}]n{r.n.iloc[0]}" if len(r) else "-")
        print(f"  {g:5s} " + "".join(f"{c:>22s}" for c in cells))

    print("\n== 5. reliability of traits (Spearman-Brown, all players with both halves)")
    for name, h in (("drill", Dh), ("game", Gh)):
        for col in (DT_ if name == "drill" else GT_):
            rr, nn = reliability(h[0], h[1], col, h[0].index.union(h[1].index))
            print(f"  {name} {col:9s} rel {rr:.2f} (n {nn})")
    R.to_csv(os.path.join(cache, "trait_translation_pairs.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

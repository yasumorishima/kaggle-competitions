"""Why is the 2023 draft class weak for run-the-hoop duration -> game get-off (hier.py: rho +0.07 vs +0.52 / +0.37)?

Same player set and target as hier.py (DL / edge rookies with >= 5 get-off plays and a hoop attempt, n=106; target =
player BLUP of get-off from the play-level MixedLM get_off ~ C(pos) + (1 | player)). Hypotheses checked:
  H1 sample size / plays per player     H2 range restriction (sd of hoop / target within year)
  H3 tracking coverage: which seasons are observed per class (2023 class = 3 seasons, rookie year is 1 of 3)
  H4 outliers / leverage (leave-one-out, drop-k most influential)  H5 position mix
  H6 role / usage (deployment shares) and within-situation rho by year
  H7 drill measurement drift (hoop_dur level, attempts, relation to 10-yd split by year)
Tests: player bootstrap 95% CI of each year's rho; Fisher z (2023 vs 2024, vs 2025, vs 2024+25); permutation of year
labels (rho_2023 - rho_2024+25). Usage: python year2023.py DATA_DIR CACHE_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus import base_table          # noqa: E402
from hier import FLOOR, blup, hoop_table, plays   # noqa: E402

R = lambda a, b: spearmanr(a, b)[0]   # noqa: E731


def boot_ci(x, y, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    bs = [R(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return np.nanpercentile(bs, [2.5, 97.5])


def fisher(r1, n1, r2, n2):
    # Spearman: se of atanh(r) ~ sqrt(1.06 / (n - 3)) (Fieller et al. 1957)
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1.06 / (n1 - 3) + 1.06 / (n2 - 3))
    return z, 2 * norm.sf(abs(z))


def adj(s, pos):
    return s - s.groupby(pos.reindex(s.index)).transform("mean")


def main(data, cache):
    pl, base = base_table(data)
    d = hoop_table(data, cache)
    pp = plays(data, cache, pl)
    n = pp.groupby("nfl_id").size()
    ids = n.index[n >= FLOOR].intersection(d.index).intersection(base.index)
    df = pp[pp.nfl_id.isin(ids)].copy()
    g = pd.read_csv(os.path.join(data, "games.csv")).set_index("game_id").season
    df["season"] = g.reindex(df.game_id).values
    re, *_ = blup(df, False)
    y = re.reindex(ids)
    x = d.hoop_dur.reindex(ids)
    yr = pl.draft_year.reindex(ids)
    pos = pl.nfl_position.reindex(ids)
    years = [2023, 2024, 2025]

    print("== rho by draft year (BLUP target) with player bootstrap 95% CI")
    rho = {}
    for v in years:
        k = yr == v
        rho[v] = (R(x[k], y[k]), int(k.sum()))
        lo, hi = boot_ci(x[k], y[k])
        print(f"  {v}: rho {rho[v][0]:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  n {rho[v][1]}")
    k = yr != 2023
    r_o, n_o = R(x[k], y[k]), int(k.sum())
    print(f"  2024+25: rho {r_o:+.2f}  n {n_o}   all: {R(x, y):+.2f} n {len(ids)}")
    print("== difference tests")
    for lab, (r2, n2) in (("2024", rho[2024]), ("2025", rho[2025]), ("2024+25", (r_o, n_o))):
        z, p = fisher(rho[2023][0], rho[2023][1], r2, n2)
        print(f"  Fisher z 2023 vs {lab}: z {z:+.2f}  p {p:.3f}")
    rng = np.random.default_rng(1)
    obs = rho[2023][0] - r_o
    is23 = (yr == 2023).values
    perm = []
    for _ in range(5000):
        s = rng.permutation(is23)
        perm.append(R(x[s], y[s]) - R(x[~s], y[~s]))
    print(f"  permutation (year labels shuffled within the 106): diff {obs:+.2f}  p(two-sided) "
          f"{np.mean(np.abs(perm) >= abs(obs)):.3f}")
    # heterogeneity across the 3 years (Cochran Q on Fisher z)
    zs = np.array([np.arctanh(rho[v][0]) for v in years]); w = np.array([(rho[v][1] - 3) / 1.06 for v in years])
    zbar = (w * zs).sum() / w.sum(); Q = (w * (zs - zbar) ** 2).sum()
    from scipy.stats import chi2
    print(f"  Cochran Q across 3 years {Q:.2f} (df 2) p {chi2.sf(Q, 2):.3f}; pooled Fisher rho {np.tanh(zbar):+.2f}")

    print("== H1 sample: players, plays per player, games per player")
    gp = df.groupby("nfl_id").game_id.nunique()
    for v in years:
        i = ids[yr == v]
        print(f"  {v}: players {len(i)}  plays/player median {int(n.reindex(i).median())} "
              f"(IQR {int(n.reindex(i).quantile(.25))}-{int(n.reindex(i).quantile(.75))})  games median {int(gp.reindex(i).median())}")

    print("== H2 range restriction: sd within year (hoop_dur s, BLUP s, raw pos-adj mean s)")
    raw = adj(df.groupby("nfl_id").player_get_off.mean(), pos).reindex(ids)
    for v in years:
        k = yr == v
        print(f"  {v}: hoop sd {x[k].std():.3f} (mean {x[k].mean():.2f})  BLUP sd {y[k].std():.4f}  raw sd {raw[k].std():.4f}")

    print("== H3 coverage: seasons observed; rho using each season's plays only (pos-adj raw mean, >= 20 plays)")
    sc = df.groupby(["nfl_id", "season"]).size().unstack(fill_value=0)
    for v in years:
        i = ids[yr == v]
        print(f"  {v} class plays by season: " + "  ".join(f"{s} {int(sc.reindex(i)[s].sum())}" for s in sc.columns)
              + f"   share rookie-season {sc.reindex(i)[v].sum() / sc.reindex(i).values.sum():.2f}")
    for v in years:
        for s in [c for c in sc.columns if c >= v]:
            t = df[df.season == s].groupby("nfl_id").player_get_off.agg(["mean", "size"])
            t = t[t["size"] >= 20]
            i = t.index.intersection(ids[yr == v])
            if len(i) < 8:
                continue
            a = adj(t.loc[i, "mean"], pos)
            lo, hi = boot_ci(x[i], a[i], B=2000)
            print(f"    class {v} season {s} (yr {s - v + 1}): n {len(i):2d}  rho {R(x[i], a[i]):+.2f} [{lo:+.2f}, {hi:+.2f}]")
    # same-year-of-career comparison: rookie season only, every class
    t = df[df.season == df.nfl_id.map(yr)].groupby("nfl_id").player_get_off.agg(["mean", "size"])
    t = t[t["size"] >= 20]; i = t.index.intersection(ids); a = adj(t.loc[i, "mean"], pos)
    print(f"  rookie season only, all classes: n {len(i)} rho {R(x[i], a[i]):+.2f}; "
          + "  ".join(f"{v} {R(x[i][yr[i] == v], a[i][yr[i] == v]):+.2f} ({int((yr[i] == v).sum())})" for v in years))
    # stability of the 2023 class's own get-off across seasons (is the target noisy or drifting?)
    t = df[yr.reindex(df.nfl_id).values == 2023].groupby(["nfl_id", "season"]).player_get_off.agg(["mean", "size"])
    t = t[t["size"] >= 20]["mean"].unstack()
    for a_, b_ in ((2023, 2024), (2024, 2025), (2023, 2025)):
        k = t[[a_, b_]].dropna()
        print(f"  2023 class get-off season {a_} vs {b_}: n {len(k)} rho {R(k[a_], k[b_]):+.2f}")
    for v in (2024,):
        t2 = df[yr.reindex(df.nfl_id).values == v].groupby(["nfl_id", "season"]).player_get_off.agg(["mean", "size"])
        t2 = t2[t2["size"] >= 20]["mean"].unstack().dropna()
        print(f"  {v} class get-off season 2024 vs 2025: n {len(t2)} rho {R(t2[2024], t2[2025]):+.2f}")

    print("== H4 outliers / leverage in 2023")
    k = (yr == 2023).values
    x3, y3 = x[k], y[k]
    loo = pd.Series({i: R(x3.drop(i), y3.drop(i)) for i in x3.index})
    print(f"  leave-one-out rho range [{loo.min():+.2f}, {loo.max():+.2f}]")
    infl = (loo - rho[2023][0]).sort_values()      # most negative influence = removing raises rho most
    drop = list(infl.index[::-1][:3])
    for kk in (1, 2, 3):
        dd = list(infl.sort_values(ascending=False).index[:kk])
        print(f"  drop {kk} most rho-suppressing: rho {R(x3.drop(dd), y3.drop(dd)):+.2f}  "
              f"({', '.join(f'{pl.display_name[j]} {pos[j]} hoop {x3[j]:.2f} y {y3[j]:+.3f}' for j in dd)})")
    print(f"  Pearson 2023 {np.corrcoef(x3, y3)[0, 1]:+.2f}  hoop z>2.5 in 2023: "
          f"{int((np.abs((x3 - x.mean()) / x.std()) > 2.5).sum())}  all years: {int((np.abs((x - x.mean()) / x.std()) > 2.5).sum())}")

    print("== H5 position mix; rho within position group by year")
    grp = pos.replace({"NT": "DT"}).map(lambda p: "interior (DT/NT)" if p == "DT" else "edge (DE/OLB)")
    print("  " + "  ".join(f"{v}: {pos[yr == v].value_counts().to_dict()}" for v in years))
    for p in ("DE", "OLB", "DT"):
        print(f"  {p}: " + "  ".join(f"{v} {R(x[(yr == v) & (pos == p)], y[(yr == v) & (pos == p)]):+.2f} "
                                    f"({int(((yr == v) & (pos == p)).sum())})" for v in years)
              + f"  all years {R(x[pos == p], y[pos == p]):+.2f}")
    for gname in grp.unique():
        print(f"  {gname}: " + "  ".join(
            f"{v} {R(x[(yr == v) & (grp == gname)], y[(yr == v) & (grp == gname)]):+.2f} ({int(((yr == v) & (grp == gname)).sum())})"
            for v in years))

    print("== H6 role / usage: deployment shares by class, and within-situation rho by class")
    dep = df.assign(third=(df.down >= 3).astype(float)).groupby("nfl_id")[["pass_play", "shotgun", "third", "edge"]].mean().reindex(ids)
    print("  mean shares: " + "  ".join(f"{v}: " + ",".join(f"{c} {dep[yr == v][c].mean():.2f}" for c in dep) for v in years))
    for v in years:
        k = yr == v
        print(f"  {v}: rho(hoop, deployment) " + "  ".join(f"{c} {R(x[k], adj(dep[c], pos)[k]):+.2f}" for c in dep))
    for lab, kk in (("pass", df.pass_play == 1), ("run", df.pass_play == 0), ("1st/2nd", df.down <= 2), ("3rd/4th", df.down >= 3)):
        t = df[kk].groupby("nfl_id").player_get_off.agg(["mean", "size"]); t = t[t["size"] >= 10]
        i = t.index.intersection(ids); a = adj(t.loc[i, "mean"], pos)
        print(f"    within {lab:8s}: " + "  ".join(f"{v} {R(x[i][yr[i] == v], a[i][yr[i] == v]):+.2f}" for v in years))
    sn = n.reindex(ids)
    print("  snaps/season by class (plays per observed season): " + "  ".join(
        f"{v} {np.median(sn[yr == v] / (2026 - v)):.0f}" for v in years)
        + f";  rho(hoop, log snaps/season) by class: " + "  ".join(
        f"{v} {R(x[yr == v], np.log1p(sn[yr == v] / (2026 - v))):+.2f}" for v in years))

    print("== H7 drill measurement: hoop_dur by combine year, attempts, relation to 10-yd split / forty")
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"), usecols=["nfl_id", "drill_name", "event_id", "attempt", "entity_type", "time"])
    c = c[(c.entity_type == "PLAYER") & (c.drill_name == "RUN_THE_HOOP_DRILL")]
    c["fmt"] = np.where(c.event_id.str[:2] == "20", "date-id", "hash-id")
    print("  event_id format by class: " + str(c.assign(yr=pl.draft_year.reindex(c.nfl_id).values)
                                              .groupby("yr").fmt.agg(lambda s: sorted(set(s))).to_dict()))
    att = c.groupby("nfl_id")[["event_id", "attempt"]].apply(lambda g: len(g.drop_duplicates())).reindex(ids)
    hz = c.assign(t=pd.to_datetime(c.time, format="mixed")).groupby(["nfl_id", "attempt"]).t.apply(lambda s: s.sort_values().diff().dt.total_seconds().median())
    hz = hz.groupby(level=0).median().reindex(ids)
    for v in years:
        k = yr == v
        print(f"  {v}: hoop mean {x[k].mean():.2f} median {x[k].median():.2f}  attempts/player {att[k].mean():.1f}  "
              f"dt {hz[k].median():.2f}s  rho(hoop, ten_yd) {spearmanr(x[k], base.ten_yd_split.reindex(ids)[k], nan_policy='omit')[0]:+.2f}  "
              f"rho(hoop, weight) {R(x[k], base.combine_weight.reindex(ids)[k]):+.2f}  "
              f"rho(ten_yd, y) {spearmanr(base.ten_yd_split.reindex(ids)[k], y[k], nan_policy='omit')[0]:+.2f}  "
              f"pick median {base.draft_overall_pick.reindex(ids)[k].median():.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

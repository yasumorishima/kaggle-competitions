"""Pillar 2, two follow-ups from 11-01 (pillar2.py found no increment for the translated traits on whole-route outcomes).

Same procedure as pillar2.py: ridge (median impute, standardise, RidgeCV), 5-fold CV by player x 20 repeats, increment =
R2(base + trait) - R2(base) with a player bootstrap 90% interval; full base = 8 stopwatch / size + log(pick), small base =
forty, ten split, weight, log(pick). Plus Spearman, the partial after the small base, and Spearman within draft class
(ranks within 2023 / 2024 / 2025, to catch class-level shifts, cf. trait_translation.py 10-31).

(1) WR: D_vmax40 (40-yd tracking top speed) -> DEEP separation. PRIMARY fixed before the first run:
      sep_deep = separation_at_pass_forward on GO / POST / CORNER routes minus that route's mean (over all players),
                 player mean over >= 20 such routes.
    Secondary (printed, not primary): thresholds 10 / 30 routes; GO only; the short-route complement (sep on all other
    routes, same adjustment) as a contrast; and the game-side top speed G_vmax (trait_translation game_traits, deep
    routes) -> sep_deep, i.e. "does in-game deep speed itself buy deep separation" (the mediator check).
(2) DT (DT + NT, as GROUP in trait_translation.py): D_v10_40 (40-yd tracking speed 1.0 s after start, fig_curve.py) ->
      getoff   PRIMARY. mean player_get_off (s from snap, lower = faster), >= 30 plays, DT / NT means removed
               (focus.py definition restricted to DT + NT). Expected sign of the trait: NEGATIVE (faster rise, earlier off).
      press    pressure rate on pass-rush snaps (blitzing == True rows, i.e. the player rushed): pressure = time_to_pressure
               or time_to_qb_hurry present or sack > 0; >= 50 rush snaps, DT / NT means removed. Expected: positive.
    Secondary: the drill v10 with onset 2 yd/s (fig_curve.onset_table, robustness), the whole DL (DE/DT/NT/OLB) with
    position means removed, and v10 beside hoop_dur (does the curve add to the run-the-hoop result?).
Usage: python pillar2b.py DATA_DIR CACHE_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_curve import onset_table                                                  # noqa: E402
from hier import hoop_table                                                         # noqa: E402
from pillar2 import FULL, SMALL, test                                               # noqa: E402
from trait_translation import GROUP, drill_table, drill_traits, game_plays, game_traits, partial, rho   # noqa: E402

DEEP3 = {"GO", "POST", "CORNER"}
DL_POS = ["DE", "DT", "NT", "OLB"]


def rho_wcls(a, b, yr):
    j = pd.DataFrame({"a": a, "b": b, "g": yr}).dropna()
    ra, rb = j.groupby("g").a.rank(pct=True), j.groupby("g").b.rank(pct=True)
    return np.corrcoef(ra - ra.groupby(j.g).transform("mean"), rb - rb.groupby(j.g).transform("mean"))[0, 1]


def by_year(t, y, yr):
    out = []
    for v in (2023, 2024, 2025):
        k = (yr == v).values
        out.append(f"{rho(t.values[k], y.values[k]):+.2f}({k.sum()})" if k.sum() >= 8 else "na")
    return "/".join(out)


def one(y, trait, base, D, yr, label):
    """Spearman block + the two ridge increments, exactly as pillar2.run for one outcome."""
    y = y.dropna()
    y = y[y.index.isin(D.index[D[trait].notna()])]
    t = D.loc[y.index, trait]
    bb = base.reindex(y.index)
    ok = bb[SMALL].notna().all(axis=1)
    pr = partial(t[ok].values, y[ok].values, bb.loc[ok, SMALL].values)
    yy = yr.reindex(y.index)
    print(f"  [{label}] {trait}: Spearman {rho(t.values, y.values):+.2f} (n {len(y)}); within class "
          f"{rho_wcls(t, y, yy):+.2f}; by class 23/24/25 {by_year(t, y, yy)}; partial | small base {pr:+.2f} "
          f"(n {int(ok.sum())}); outcome sd {y.std():.3f}")
    res = {}
    for nm, cols in (("full", FULL), ("small", SMALL)):
        res[nm] = test(y, bb[cols], D.loc[y.index, [trait]], f"{label} | {nm} base")
    return res


def sep_table(data):
    pp = pd.read_csv(os.path.join(data, "player_play.csv"), usecols=["nfl_id", "route_ran", "separation_at_pass_forward"])
    s = pp.dropna(subset=["route_ran", "separation_at_pass_forward"]).copy()
    s["adj"] = s.separation_at_pass_forward - s.groupby("route_ran").separation_at_pass_forward.transform("mean")
    return s


def agg(s, mask, minn):
    g = s[mask].groupby("nfl_id").adj.agg(["mean", "size"])
    return g["mean"].where(g["size"] >= minn).dropna(), g["size"]


def wr(data, cache, pl, base, D, yr):
    print("\n==== (1) WR: 40-yd top speed -> separation on deep routes (GO / POST / CORNER)")
    ids = pl.index[pl.nfl_position == "WR"]
    s = sep_table(data)
    s = s[s.nfl_id.isin(ids)]
    deep = s.route_ran.isin(DEEP3)
    print(f"WR routes with separation: {len(s)}  deep (GO/POST/CORNER) {int(deep.sum())}  "
          f"deep sep mean {s.separation_at_pass_forward[deep].mean():.2f} yd vs other "
          f"{s.separation_at_pass_forward[~deep].mean():.2f} yd")
    y, n = agg(s, deep, 20)
    print(f"deep routes per WR (with drill vmax): median {int(n.reindex(D.index[D.D_vmax40.notna()]).dropna().median())}; "
          f"WRs with >= 20: {len(y)}")
    print("\n-- PRIMARY: sep_deep (>= 20 deep routes)")
    prim = one(y, "D_vmax40", base, D, yr, "sep_deep")
    print("\n-- secondary")
    for minn in (10, 30):
        one(agg(s, deep, minn)[0], "D_vmax40", base, D, yr, f"sep_deep >= {minn}")
    one(agg(s, s.route_ran == "GO", 15)[0], "D_vmax40", base, D, yr, "sep GO only >= 15")
    one(agg(s, ~deep, 30)[0], "D_vmax40", base, D, yr, "sep other routes >= 30 (contrast)")
    grp = pl.nfl_position.map(GROUP)
    G, _ = game_traits(game_plays(data, cache), grp)
    GD = D.join(G[["G_vmax"]], how="outer")
    print("  mediator: in-game deep-route top speed (G_vmax) -> sep_deep")
    one(y, "G_vmax", base, GD, yr, "sep_deep ~ game G_vmax")
    return prim


def dl_outcomes(data, pl):
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["nfl_id", "player_get_off", "blitzing", "time_to_pressure", "time_to_qb_hurry", "sack"])
    pp["pos"] = pl.nfl_position.reindex(pp.nfl_id).values
    pp = pp[pp.pos.isin(DL_POS)]
    go = pp.dropna(subset=["player_get_off"]).groupby("nfl_id").player_get_off.agg(["mean", "size"])
    rush = pp[pp.blitzing.astype(str).str.upper().isin(["TRUE", "1", "1.0"])].copy()
    rush["pr"] = (rush.time_to_pressure.notna() | rush.time_to_qb_hurry.notna() | (rush.sack.fillna(0) > 0)).astype(float)
    pr = rush.groupby("nfl_id").pr.agg(["mean", "size"])
    o = pd.DataFrame({"getoff": go["mean"].where(go["size"] >= 30), "n_go": go["size"],
                      "press": pr["mean"].where(pr["size"] >= 50), "n_rush": pr["size"]})
    o["pos"] = pl.nfl_position.reindex(o.index)
    return o, rush


def adj(o, col, keep):
    k = o[o.pos.isin(keep)]
    return (k[col] - k.groupby("pos")[col].transform("mean")).dropna()


def dt(data, cache, pl, base, D, yr):
    print("\n==== (2) DT (DT + NT): 40-yd speed at 1.0 s (v10) -> get-off and pressure")
    o, rush = dl_outcomes(data, pl)
    print(f"DL rush snaps {len(rush)}  pressure rate {rush.pr.mean():.3f} (DT/NT {rush.pr[rush.pos.isin(['DT', 'NT'])].mean():.3f})")
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"),
                    usecols=["event_id", "nfl_id", "entity_type", "time", "drill_name", "attempt", "x", "y"])
    c = c[(c.entity_type == "PLAYER") & (c.drill_name == "FORTY_YARD_DASH")]
    DD = D.join(onset_table(c)[["v10_on2"]], how="outer").join(hoop_table(data, cache)[["hoop_dur"]], how="outer")
    res = {}
    for oc, minlab in (("getoff", ">= 30 plays"), ("press", ">= 50 rush snaps")):
        y = adj(o, oc, ["DT", "NT"])
        print(f"\n-- PRIMARY: DT {oc} ({minlab}, DT/NT means removed)  DT/NT with outcome {len(y)}")
        res[oc] = one(y, "D_v10_40", base, DD, yr, f"DT {oc}")
    print("\n-- secondary")
    for oc in ("getoff", "press"):
        y = adj(o, oc, ["DT", "NT"])
        one(y, "v10_on2", base, DD, yr, f"DT {oc} (drill onset 2 yd/s)")
        yl = adj(o, oc, DL_POS)
        one(yl, "D_v10_40", base, DD, yr, f"all DL {oc} (pos means removed)")
        # does v10 add beside the run-the-hoop duration (the pillar-2 headline)?
        k = yl[yl.index.isin(DD.index[DD.D_v10_40.notna() & DD.hoop_dur.notna()])]
        print(f"  [all DL {oc}] v10 on top of base + hoop_dur (n {len(k)}):")
        for nm, cols in (("full", FULL), ("small", SMALL)):
            test(k, base.reindex(k.index)[cols].join(DD.loc[k.index, ["hoop_dur"]]), DD.loc[k.index, ["D_v10_40"]],
                 f"all DL {oc} | {nm} base + hoop")
    return res


def main(data, cache):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    base = cr[FULL[:-1]].join(np.log(pl.draft_overall_pick).rename("log_pick"), how="outer")
    D, _ = drill_traits(drill_table(data))
    yr = pl.draft_year
    print("PILLAR 2b: (1) WR top speed -> deep-route separation, (2) DT 40-yd v10 -> get-off / pressure")
    print("ridge, 5-fold CV by player x 20 repeats; increment = R2(base+trait) - R2(base), player bootstrap 90% CI "
          "(pillar2.test)")
    wr(data, cache, pl, base, D, yr)
    dt(data, cache, pl, base, D, yr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

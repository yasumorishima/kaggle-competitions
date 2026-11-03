"""Pillar 2c: DT 40-yd v10 -> pass-rush pressure, at the play level (11-02 next step; no 50-snap cutoff).

pillar2b (11-02) found the direction clearly positive for DT (DT + NT) pressure rate but the ridge uplift interval
crossed 0 with only 36 players (>= 50 rush snaps). Here every pass-rush snap counts and players are not cut.

PRIMARY (fixed before the first run, no other trait searched):
  play-level logistic GEE, pressure ~ C(pos) + small base (forty, ten split, weight, log pick; standardized,
  median-imputed) + z(D_v10_40), clustered by player (exchangeable), DT + NT with v10 and >= 1 rush snap.
  Pass = the 95% CI of the z(v10) coefficient is above 0 (cluster-robust), confirmed by a player cluster bootstrap.
Secondary (printed, not primary):
  - same model with the full base (8 stopwatch / size + log pick) and with play context (shotgun, down, ytg, edge)
  - logistic GLMM with a player random intercept (statsmodels BinomialBayesMixedGLM, variational Bayes)
  - shrunken target: beta-binomial empirical Bayes rate per player (DT/NT prior), floor 10 rush snaps, ridge CV as
    pillar2.test (20 x 5-fold by player, player bootstrap 90% uplift) for the small and full base
  - the whole DL (DE / DT / NT / OLB) in the GEE with C(pos)
pressure = time_to_pressure or time_to_qb_hurry present or sack > 0, on rows with blitzing == True (the player rushed),
as pillar2b.dl_outcomes.
Usage: python pillar2c.py DATA_DIR CACHE_DIR
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pillar2 import FULL, SMALL, test                                    # noqa: E402
from trait_translation import drill_table, drill_traits                  # noqa: E402

DL_POS = ["DE", "DT", "NT", "OLB"]
DT_POS = ["DT", "NT"]
CTX = "shotgun + C(down) + ytg + edge"


def rush_plays(data, pl):
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["game_id", "play_id", "nfl_id", "down", "yards_to_go", "offense_formation", "lined_up_position",
                              "blitzing", "time_to_pressure", "time_to_qb_hurry", "sack"])
    pp["pos"] = pl.nfl_position.reindex(pp.nfl_id).values
    pp = pp[pp.pos.isin(DL_POS) & pp.blitzing.astype(str).str.upper().isin(["TRUE", "1", "1.0"])].copy()
    pp["pr"] = (pp.time_to_pressure.notna() | pp.time_to_qb_hurry.notna() | (pp.sack.fillna(0) > 0)).astype(float)
    pp["shotgun"] = (pp.offense_formation == "SHOTGUN").astype(float)
    pp["ytg"] = pp.yards_to_go.clip(upper=20).astype(float)
    pp["edge"] = (pp.lined_up_position == "EDGE").astype(float)
    pp["down"] = pp.down.fillna(1).astype(int)
    return pp[["game_id", "play_id", "nfl_id", "pos", "pr", "shotgun", "down", "ytg", "edge"]]


def zcols(B, cols, ids):
    P = B.reindex(ids)[cols].copy()
    P = P.fillna(P.median())
    P = (P - P.mean()) / P.std()
    P.columns = ["b_" + c for c in cols]
    return P


def gee(formula, df):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.gee(formula, "nfl_id", df, family=sm.families.Binomial(),
                       cov_struct=sm.cov_struct.Exchangeable()).fit()


def report(m, lab, df):
    b, ci, p = m.params["z_v10"], m.conf_int().loc["z_v10"], m.pvalues["z_v10"]
    print(f"  {lab:34s} beta {b:+.4f} (OR {np.exp(b):.3f})  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p {p:.2g}  "
          f"(players {df.nfl_id.nunique()}, snaps {len(df)})")
    return b, ci[0], ci[1]


def eb_rate(k, n):
    """Beta-binomial marginal ML prior (a, b) and the posterior mean rate per player."""
    def nll(t):
        a, b = np.exp(t)
        return -np.sum(betaln(k + a, n - k + b) - betaln(a, b))
    m = (k.sum() / n.sum())
    r = minimize(nll, np.log([m * 20, (1 - m) * 20]), method="Nelder-Mead")
    a, b = np.exp(r.x)
    return (k + a) / (n + a + b), a, b


def main(data, cache):
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id")
    cr = pd.read_csv(os.path.join(data, "combine_results.csv")).drop_duplicates("nfl_id").set_index("nfl_id")
    base = cr[FULL[:-1]].join(np.log(pl.draft_overall_pick).rename("log_pick"), how="outer")
    D, _ = drill_traits(drill_table(data))
    pp = rush_plays(data, pl)
    cnt = pp.groupby("nfl_id").pr.agg(["sum", "size"])
    print("PILLAR 2c: DT 40-yd v10 -> pressure on every pass-rush snap (no 50-snap cutoff)")
    print(f"DL rush snaps {len(pp)}  pressure rate {pp.pr.mean():.3f}  DL rushers {len(cnt)}")

    res = {}
    for grp, keep in (("DT", DT_POS), ("all DL", DL_POS)):
        ids = cnt.index[pl.nfl_position.reindex(cnt.index).isin(keep)]
        ids = ids[D.reindex(ids).D_v10_40.notna().values]
        df = pp[pp.nfl_id.isin(ids)].copy()
        old = (cnt.reindex(ids)["size"] >= 50).sum()
        print(f"\n==== {grp} ({'/'.join(keep)}): players with v10 and >= 1 rush snap {len(ids)} (>= 50 snaps: {old})  "
              f"snaps {len(df)}  median snaps/player {int(cnt.reindex(ids)['size'].median())}")
        v = D.reindex(ids).D_v10_40
        S, F = zcols(base, SMALL, ids), zcols(base, FULL, ids)
        X = S.join(F, rsuffix="_f").assign(z_v10=(v - v.mean()) / v.std())
        X = X.loc[:, ~X.columns.duplicated()]
        dfx = df.join(X, on="nfl_id")
        s_terms = " + ".join("b_" + c for c in SMALL)
        f_terms = " + ".join("b_" + c for c in FULL)
        pos = "C(pos) + " if dfx.pos.nunique() > 1 else ""
        print("  play-level logistic GEE, clustered by player (exchangeable); z_v10 = +1 sd of 40-yd speed at 1.0 s")
        res[(grp, "small")] = report(gee(f"pr ~ {pos}{s_terms} + z_v10", dfx), "PRIMARY small base" if grp == "DT" else "small base", dfx)
        report(gee(f"pr ~ {pos}{f_terms} + z_v10", dfx), "full base", dfx)
        report(gee(f"pr ~ {pos}{s_terms} + {CTX} + z_v10", dfx), "small base + play context", dfx)
        report(gee(f"pr ~ {pos}z_v10", dfx), "no base (v10 only)", dfx)
        if grp == "DT":
            # player cluster bootstrap of the primary coefficient
            rng = np.random.default_rng(0)
            groups = {i: g for i, g in dfx.groupby("nfl_id")}
            arr = np.array(list(groups))
            bs = []
            for _ in range(300):
                pick = rng.choice(arr, len(arr))
                bd = pd.concat([groups[i].assign(nfl_id=k) for k, i in enumerate(pick)], ignore_index=True)
                try:
                    bs.append(gee(f"pr ~ {pos}{s_terms} + z_v10", bd).params["z_v10"])
                except Exception:   # noqa: BLE001
                    pass
            bs = np.array(bs)
            print(f"  PRIMARY player cluster bootstrap ({len(bs)} reps): 95% [{np.percentile(bs, 2.5):+.4f}, "
                  f"{np.percentile(bs, 97.5):+.4f}]  share > 0 {np.mean(bs > 0):.3f}")
            # GLMM, player random intercept (variational Bayes)
            try:
                from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = BinomialBayesMixedGLM.from_formula(f"pr ~ {pos}{s_terms} + z_v10", {"player": "0 + C(nfl_id)"}, dfx).fit_vb()
                k = list(g.model.exog_names).index("z_v10")
                mu, sd = g.fe_mean[k], g.fe_sd[k]
                print(f"  GLMM (player random intercept, VB) small base: beta {mu:+.4f}  95% [{mu - 1.96 * sd:+.4f}, "
                      f"{mu + 1.96 * sd:+.4f}]  player sd {np.exp(g.vcp_mean[0]):.3f}")
            except Exception as e:   # noqa: BLE001
                print("  GLMM failed", repr(e)[:200])
            # shrunken target into the pillar-2 ridge CV
            c = cnt.reindex(ids)
            rate, a, b = eb_rate(c["sum"].values.astype(float), c["size"].values.astype(float))
            eb = pd.Series(rate, index=ids)
            eb = eb - eb.groupby(pl.nfl_position.reindex(ids)).transform("mean")
            print(f"  beta-binomial prior a {a:.2f} b {b:.2f} (prior n {a + b:.1f}, mean {a / (a + b):.3f})")
            for floor in (10, 25, 50):
                k_ = c.index[c["size"] >= floor]
                y = eb.reindex(k_)
                print(f"  EB rate, >= {floor} rush snaps (n {len(k_)}): Spearman v10 {spearmanr(v.reindex(k_), y)[0]:+.2f}")
                for nm, cols in (("small", SMALL), ("full", FULL)):
                    test(y, base.reindex(k_)[cols], D.reindex(k_)[["D_v10_40"]], f"    EB >= {floor} | {nm} base")
            yr = pl.draft_year.reindex(ids)
            print("  by draft year (GEE small base): " + "  ".join(
                f"{int(t)} " + "{:+.3f}".format(gee(f"pr ~ {pos}{s_terms} + z_v10", dfx[dfx.nfl_id.isin(ids[(yr == t).values])]).params["z_v10"])
                for t in sorted(yr.dropna().unique())))
    b, lo, hi = res[("DT", "small")]
    print(f"\nVERDICT (primary, DT small base GEE): beta {b:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> "
          + ("PASS (CI above 0)" if lo > 0 else "not passed (CI crosses 0)"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

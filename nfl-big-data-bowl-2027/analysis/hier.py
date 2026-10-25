"""Pillar 2, hierarchical: run-the-hoop duration -> DL game get-off from ALL plays (no 30-play cutoff).

Pre-registered claim (fixed before this run, no other features searched): "run-the-hoop active duration (best attempt,
dl_drills.feats, prefix hoop_dur) -> mean in-game player_get_off" for DL / edge rookies (DE, DT, NT, OLB).

(a) Shrunken target: play-level MixedLM  get_off ~ C(pos) [+ play context] + (1 | player); the player's BLUP (random
    intercept) is the target. Also the closed-form EB check  tau2 / (tau2 + sigma2 / n_plays).  Then the same CV as
    focus.py (ridge single feature vs base vs base+feature, 20 x 5-fold by player, player bootstrap 90% uplift) via
    db_outcome.test, and Spearman by draft year.
(b) Play-level MixedLM with z(hoop_dur) as a player-level fixed effect beside C(pos), the base covariates (standardized,
    median-imputed as in the ridge) and, optionally, play context; player random intercept. Coefficient, 95% CI, p.
Play context (player_play): pass play (pass_result not null), shotgun, down (1-4), yards_to_go (clipped at 20),
    lined up EDGE vs INTERIOR_LINE.
Usage: python hier.py DATA_DIR CACHE_DIR
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_outcome import test          # noqa: E402
from dl_drills import feats          # noqa: E402
from focus import base_table         # noqa: E402

DL_POS = ["DE", "DT", "NT", "OLB"]
FLOOR = 5
CTX = "pass_play + shotgun + C(down) + ytg + edge"
BASE_COLS = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height",
             "combine_weight", "draft_overall_pick"]


def hoop_table(data, cache):
    f = os.path.join(cache, "hoop_feats.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    c = pd.read_csv(os.path.join(data, "combine_tracking.csv"))
    c = c[(c.entity_type == "PLAYER") & (c.drill_name == "RUN_THE_HOOP_DRILL")]
    rows = []
    for (pid, eid, att), g in c.groupby(["nfl_id", "event_id", "attempt"]):     # same as focus.dl
        g = g.sort_values("time")
        r = feats(g.x.values, g.y.values)
        if r:
            r["nfl_id"] = pid; rows.append(r)
    d = pd.DataFrame(rows).sort_values("dur").groupby("nfl_id").first().add_prefix("hoop_")
    d.to_parquet(f)
    return d


def plays(data, cache, pl):
    f = os.path.join(cache, "dl_plays.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    pp = pd.read_csv(os.path.join(data, "player_play.csv"),
                     usecols=["game_id", "play_id", "nfl_id", "down", "yards_to_go", "offense_formation", "pass_result",
                              "lined_up_position", "player_get_off"]).dropna(subset=["player_get_off"])
    pp["pos"] = pl.nfl_position.reindex(pp.nfl_id).values
    pp = pp[pp.pos.isin(DL_POS)].copy()
    pp["pass_play"] = pp.pass_result.notna().astype(float)
    pp["shotgun"] = (pp.offense_formation == "SHOTGUN").astype(float)
    pp["ytg"] = pp.yards_to_go.clip(upper=20).astype(float)
    pp["edge"] = (pp.lined_up_position == "EDGE").astype(float)
    pp["down"] = pp.down.astype(int)
    pp = pp[["game_id", "play_id", "nfl_id", "pos", "player_get_off", "pass_play", "shotgun", "down", "ytg", "edge"]]
    pp.to_parquet(f)
    return pp


def mixed(formula, df):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, df, groups=df["nfl_id"]).fit(reml=True, method=["bfgs", "powell"])


def blup(df, ctx):
    m = mixed("player_get_off ~ C(pos)" + (" + " + CTX if ctx else ""), df)
    re = pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()})
    return re, float(m.cov_re.iloc[0, 0]), float(m.scale), m


def year_rho(x, y, yr, label):
    out = [f"all {spearmanr(x, y)[0]:+.2f} ({len(y)})"]
    for v in sorted(yr.dropna().unique()):
        k = yr == v
        out.append(f"{int(v)} {spearmanr(x[k], y[k])[0]:+.2f} ({int(k.sum())})")
    print(f"  Spearman hoop_dur vs {label}: " + "  ".join(out))


def main(data, cache):
    os.makedirs(cache, exist_ok=True)
    pl, base = base_table(data)
    d = hoop_table(data, cache)
    pp = plays(data, cache, pl)
    n = pp.groupby("nfl_id").size()
    print(f"DL players in players.csv {pl.nfl_position.isin(DL_POS).sum()}  with any get-off play {len(n)}  "
          f"with hoop {len(n.index.intersection(d.index))}  plays {len(pp)}")
    old = n.index[n >= 30].intersection(d.index).intersection(base.index)
    new = n.index[n >= FLOOR].intersection(d.index).intersection(base.index)
    print(f"n before (>= 30 plays, hoop, base) {len(old)}  after (>= {FLOOR} plays) {len(new)}  gained {len(new) - len(old)} "
          f"(their plays: {n.reindex(new.difference(old)).tolist()})")
    nohoop = n.index.difference(d.index)
    print(f"players with get-off but no hoop attempt: {len(nohoop)} "
          f"{pl.nfl_position.reindex(nohoop).value_counts().to_dict()} -> excluded (claim needs the drill)")
    df = pp[pp.nfl_id.isin(new)].copy()
    print(f"plays used {len(df)}  plays/player median {int(n.reindex(new).median())} min {int(n.reindex(new).min())}")

    # ---------- (a) shrunken targets ----------
    yr = pl.draft_year.reindex(new)
    raw = df.groupby("nfl_id").player_get_off.mean()
    raw_adj = raw - raw.groupby(pl.nfl_position.reindex(new)).transform("mean")
    tgt = {"raw mean (pos-adj, focus.py def)": raw_adj.reindex(new)}
    for ctx in (False, True):
        re, tau2, s2, m = blup(df, ctx)
        nn = n.reindex(new)
        lab = "BLUP" + (" + context" if ctx else "")
        print(f"\n(a) MixedLM {lab}: tau (player sd) {np.sqrt(tau2):.4f} s  sigma (play sd) {np.sqrt(s2):.4f} s  "
              f"ICC {tau2 / (tau2 + s2):.3f}  shrink factor median {np.median(tau2 / (tau2 + s2 / nn)):.3f} "
              f"min {np.min(tau2 / (tau2 + s2 / nn)):.3f}")
        if ctx:
            fe = m.fe_params; ci = m.conf_int()
            for k in fe.index:
                if not k.startswith(("C(pos)", "Intercept")):
                    print(f"    context {k:14s} {fe[k]:+.4f} [{ci.loc[k, 0]:+.4f}, {ci.loc[k, 1]:+.4f}]")
        else:   # closed-form EB check against the raw position-adjusted mean
            eb = tau2 / (tau2 + s2 / nn) * raw_adj.reindex(new)
            print(f"    closed-form EB vs BLUP: r {np.corrcoef(eb, re.reindex(new))[0, 1]:.4f}")
        tgt[lab] = re.reindex(new)
    print("\nCV on the enlarged set (ridge, 20 x 5-fold by player, player bootstrap 90% uplift); 'curve only' = hoop_dur alone")
    for lab, y in tgt.items():
        test(y, base.reindex(new), d.reindex(new)[["hoop_dur"]], f"  {lab}")
        year_rho(d.reindex(new).hoop_dur, y, yr, lab)
    y = tgt["BLUP"].reindex(old)
    test(y, base.reindex(old), d.reindex(old)[["hoop_dur"]], "  BLUP on the old 103 (same players, new target)")

    # ---------- (b) play-level mixed model with hoop as a player-level fixed effect ----------
    P = base.reindex(new)[BASE_COLS].copy()
    P = (P.fillna(P.median()) - P.fillna(P.median()).mean()) / P.fillna(P.median()).std()
    P.columns = ["b_" + c for c in P.columns]
    P["z_hoop"] = (d.hoop_dur.reindex(new) - d.hoop_dur.reindex(new).mean()) / d.hoop_dur.reindex(new).std()
    dfb = df.join(P, on="nfl_id")
    bterms = " + ".join(P.columns[:-1])
    sd_y = tgt["raw mean (pos-adj, focus.py def)"].std()
    print(f"\n(b) play-level MixedLM, player random intercept; coefficient = seconds of get-off per +1 sd of hoop_dur "
          f"(hoop sd {d.hoop_dur.reindex(new).std():.3f} s; player-mean get-off sd {sd_y:.4f} s)")
    for lab, f in (("pos + hoop", "C(pos) + z_hoop"),
                   ("pos + base + hoop", f"C(pos) + {bterms} + z_hoop"),
                   ("pos + base + context + hoop", f"C(pos) + {bterms} + {CTX} + z_hoop")):
        m = mixed("player_get_off ~ " + f, dfb)
        ci = m.conf_int().loc["z_hoop"]
        print(f"  {lab:28s} beta {m.fe_params['z_hoop']:+.4f} s  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"p {m.pvalues['z_hoop']:.2g}  (players {dfb.nfl_id.nunique()}, plays {len(dfb)}, "
              f"tau {np.sqrt(float(m.cov_re.iloc[0, 0])):.4f})")
    # player-level cluster bootstrap of the full-model beta (robustness of the Wald CI)
    rng = np.random.default_rng(0)
    ids = np.array(new)
    groups = {i: g for i, g in dfb.groupby("nfl_id")}
    bs = []
    for b in range(200):
        pick = rng.choice(ids, len(ids))
        bd = pd.concat([groups[i].assign(nfl_id=k) for k, i in enumerate(pick)], ignore_index=True)
        try:
            bs.append(mixed(f"player_get_off ~ C(pos) + {bterms} + {CTX} + z_hoop", bd).fe_params["z_hoop"])
        except Exception:      # noqa: BLE001
            pass
    print(f"  full model, player cluster bootstrap ({len(bs)} reps): 95% [{np.percentile(bs, 2.5):+.4f}, "
          f"{np.percentile(bs, 97.5):+.4f}]  share > 0 {np.mean(np.array(bs) > 0):.3f}")

    # ---------- (c) why context weakens it: deployment vs within-situation ----------
    # Context terms are partly role (coaches put players on passing downs / on the edge), so adjusting for them can
    # absorb part of the hoop signal. Show (1) how hoop relates to deployment, (2) hoop vs get-off within one situation.
    pos = pl.nfl_position.reindex(new)
    dep = df.assign(third=(df.down >= 3).astype(float)).groupby("nfl_id")[["pass_play", "shotgun", "third", "edge"]].mean()
    dep = dep.reindex(new) - dep.reindex(new).groupby(pos).transform("mean")
    print("\n(c) deployment (share of plays, position-adjusted) vs hoop_dur, Spearman (negative = faster hoop, more such snaps): "
          + "  ".join(f"{c} {spearmanr(d.hoop_dur.reindex(new), dep[c])[0]:+.2f}" for c in dep))
    for lab, k in (("pass plays", df.pass_play == 1), ("run plays", df.pass_play == 0),
                   ("pass + shotgun", (df.pass_play == 1) & (df.shotgun == 1)),
                   ("1st/2nd down", df.down <= 2), ("3rd/4th down", df.down >= 3)):
        t = df[k].groupby("nfl_id").player_get_off.agg(["mean", "size"])
        t = t[t["size"] >= 10]
        i = t.index.intersection(new)
        a = t.loc[i, "mean"] - t.loc[i, "mean"].groupby(pos.reindex(i)).transform("mean")
        print(f"    within {lab:15s} n {len(i):3d}  Spearman hoop_dur vs pos-adj mean get-off {spearmanr(d.hoop_dur.reindex(i), a)[0]:+.2f}")
    sit = CTX.replace(" + edge", "")
    m = mixed(f"player_get_off ~ C(pos) + {bterms} + {sit} + z_hoop", dfb)
    ci = m.conf_int().loc["z_hoop"]
    print(f"  (b) pos + base + situation (no edge alignment) + hoop: beta {m.fe_params['z_hoop']:+.4f} s  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p {m.pvalues['z_hoop']:.2g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

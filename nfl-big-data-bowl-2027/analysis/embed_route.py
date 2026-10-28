"""ML pillar 1, scene-matched: identify a player's game routes from his combine drills (route_segs.py segments).

Scene matching (each drill only meets the game scenes that look like it):
  WR: drill route -> game route_ran (WR_MAP), WR / SLOT_WR alignments only.
  DB: back-pedal drills -> game snaps where the DB opens in a back-pedal (`back` > 110 deg; CB, SLOT_CB, safeties),
      split by where he is heading 2.5 s later relative to the back-pedal (turn bin): each drill meets the bin it
      ends in most often (computed on the drills with the same rule).
Whole segments (drill: the whole attempt from the first move; game: snap -> pass, onset-aligned) cropped to LC frames.

Per segment, the same hand features on both sides (speed at 0.5..2.5 s, top speed, max accel, max decel, break loss,
shape). A player's print for one scene pair = median over his segments, z-scored across players WITHIN the pair and
side (removes the drill-vs-game offset of the scene). Scores between a game print and a drill print:
  hand   : Pearson over the dimensions both players have (no learning)
  handw  : the same with per-dimension weights = cross-side Spearman learnt on the training players (K-fold by player)
  cnn    : 1D-CNN embedding trained with set-level InfoNCE inside scene pairs (pair + side as input channels),
           embeddings centred within pair x side over the evaluated players, score = mean cosine over shared pairs
Evaluation: held-out players of the same group (WR / DB); top-1 / top-5 of the true player against chance.
Role checks (game -> game, first half of his games vs second half):
  role   : identity from role mix alone (shares of alignment / route / coverage / turn bin)
  motion : hand features z-scored within (scene x alignment) cells, so a slot and an outside WR on the same route
           are compared only with players in the same cell.
Usage: python embed_route.py DATA_DIR CACHE_DIR [folds] [cnn_steps]
"""
import os
import sys

import numpy as np
import pandas as pd

WR_MAP = {"SLANT_ROUTE_LEFT": ["SLANT"], "CURL_ROUTE_RIGHT": ["HITCH"], "COMEBACK_ROUTE_RIGHT": ["HITCH", "OUT"],
          "DAGGER_ROUTE_LEFT": ["IN", "CROSS"], "GO_ROUTE_RIGHT": ["GO"], "RED_ZONE_FADE_RIGHT": ["GO"],
          "POST_CORNER_ROUTE_LEFT": ["CORNER", "POST"], "POST_CORNER_ROUTE_RIGHT": ["CORNER", "POST"],
          "SLOT_SAIL_ROUTE_RIGHT": ["CORNER", "OUT"], "SLOT_STRIKE_ROUTE_LEFT": ["IN", "SLANT"],
          "SPEED_OUT_LEFT": ["OUT"], "SPEED_OUT_ROUTE_LEFT": ["OUT"]}
DB_DRILLS = ["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BOX_DRILL",
             "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2", "W_DRILL"]
LC = {"WR": 30, "DB": 25}
FN = ["v05", "v10", "v15", "v20", "v25", "vmax", "amax", "dmax", "loss", "y10", "y20", "x20", "path"]


def onset_align(X, meta):
    """Game segments start at the snap; shift them to the player's own first move (drills already start there)."""
    X = X.copy()
    nn_ = meta.n.values.copy()
    for i in np.nonzero(meta.game.values == 1)[0]:
        sp = X[i, 2]
        mv = np.nonzero(sp >= 1.0)[0]
        s = max(0, (mv[0] if len(mv) else 0) - 3)
        if s:
            n = nn_[i]
            seg = X[i, :, s:]
            seg = seg - np.array([seg[0, 0], seg[1, 0], 0], np.float32)[:, None]
            X[i] = np.concatenate([seg, np.repeat(seg[:, -1:], s, 1)], 1)
            nn_[i] = max(n - s, 1)
    meta = meta.assign(n=nn_)
    return X, meta


def turn_bin(X, L):
    """Heading at the end of the crop relative to the initial heading (+y): 0 = keeps going (<60 deg), 1 = breaks
    across (60-120), 2 = drives back (>120)."""
    vx = X[:, 0, L - 1] - X[:, 0, L - 6]; vy = X[:, 1, L - 1] - X[:, 1, L - 6]
    a = np.degrees(np.abs(np.arctan2(vx, vy)))
    return np.digitize(a, [60, 120])


def hand_feats(X, n, L):
    sp = X[:, 2, :L]
    acc = np.diff(sp, axis=1) * 10
    valid = np.arange(L)[None] < np.minimum(n, L)[:, None]
    spv = np.where(valid, sp, np.nan)
    F = np.full((len(X), len(FN)), np.nan, np.float32)
    for j, k in enumerate([5, 10, 15, 20, min(25, L - 1)]):
        F[:, j] = np.where(n > k, sp[:, k], np.nan)
    F[:, 5] = np.nanmax(spv, 1)
    F[:, 6] = np.nanmax(np.where(valid[:, 1:16], acc[:, :15], np.nan), 1)
    F[:, 7] = np.nanmax(np.where(valid[:, 1:], -acc, np.nan), 1)
    pre = np.nanmax(spv[:, :15], 1); post = np.nanmin(np.where(np.arange(L)[None] >= 8, spv, np.nan), 1)
    F[:, 8] = 1 - post / np.maximum(pre, 0.5)
    F[:, 9] = np.where(n > 10, X[:, 1, 10], np.nan)
    F[:, 10] = np.where(n > 20, X[:, 1, 20], np.nan)
    F[:, 11] = np.where(n > 20, X[:, 0, 20], np.nan)
    d = np.hypot(np.diff(X[:, 0, :L], axis=1), np.diff(X[:, 1, :L], axis=1))
    F[:, 12] = np.where(valid[:, 1:], d, 0).sum(1) / np.maximum(np.minimum(n, L) - 1, 1) * 10
    return F


def build(data, cache, grp):
    X = np.load(os.path.join(cache, "route_X.npy")); meta = pd.read_parquet(os.path.join(cache, "route_meta.parquet"))
    meta["moving"] = (meta.game == 1) & (X[:, 2, 0] >= 1.0)    # already in motion at the snap: no clean start
    X, meta = onset_align(X, meta)
    L = LC[grp]
    m = meta[(meta.grp == grp) & ~meta.moving].copy()
    m["tb"] = turn_bin(X[m.index.values], L)
    pairs = []                                                   # (pair name, drill mask, game mask)
    if grp == "WR":
        g = (m.game == 1) & m.pos.isin(["WR", "SLOT_WR"])
        for dn, routes in WR_MAP.items():
            pairs.append((dn, (m.game == 0) & (m.scene == dn), g & m.scene.isin(routes)))
    else:
        g = (m.game == 1) & m.pos.isin(["CB", "SLOT_CB", "HIGH_SAFETY", "BOX_SAFETY"]) & (m.back > 110)
        for dn in DB_DRILLS:
            dm = (m.game == 0) & (m.scene == dn)
            b = int(m.loc[dm, "tb"].mode().iat[0])
            pairs.append((f"{dn}->bin{b}", dm, g & (m.tb == b)))
    keep = np.zeros(len(m), bool)
    m["pair"] = -1
    rows = []
    for k, (name, dm, gm) in enumerate(pairs):
        for side, mk in ((0, dm), (1, gm)):
            r = m[mk.values & (m.n.values >= (1 if side == 0 else 15))].copy(); r["pair"] = k; rows.append(r)
    mm = pd.concat(rows)                                         # one row per (segment, pair); a segment can be in 2
    Xs = X[mm.index.values][:, :, :L]
    F = hand_feats(Xs, mm.n.values, L)
    return mm.reset_index(drop=True), Xs, F, [p[0] for p in pairs]


def prints(mm, F, players, min_game=8):
    """{(pair, side): DataFrame players x features} of medians, z-scored across `players` within pair x side."""
    out = {}
    for (k, side), r in mm[mm.nfl_id.isin(players)].groupby(["pair", "game"]):
        f = pd.DataFrame(F[r.index.values], columns=FN).assign(p=r.nfl_id.values).groupby("p")
        cnt = f.size(); med = f.median()
        med = med[cnt >= (1 if side == 0 else min_game)]
        out[(k, side)] = (med - med.mean()) / (med.std() + 1e-6)
    return out


def stack(pr, players, npairs, side):
    M = np.full((len(players), npairs * len(FN)), np.nan)
    pos = {p: i for i, p in enumerate(players)}
    for k in range(npairs):
        d = pr.get((k, side))
        if d is None:
            continue
        for p, v in d.iterrows():
            if p in pos:
                M[pos[p], k * len(FN):(k + 1) * len(FN)] = v.values
    return M


def score(G, D, w=None):
    """Weighted Pearson-like similarity over shared dimensions. G: game prints (n x d), D: drill prints."""
    n = len(G); S = np.full((n, n), -np.inf)
    w = np.ones(G.shape[1]) if w is None else w
    for i in range(n):
        for j in range(n):
            ok = ~np.isnan(G[i]) & ~np.isnan(D[j]) & (w > 0)
            if ok.sum() < 5:
                continue
            a, b, ww = G[i, ok], D[j, ok], w[ok]
            a = a - (ww * a).sum() / ww.sum(); b = b - (ww * b).sum() / ww.sum()
            S[i, j] = (ww * a * b).sum() / np.sqrt((ww * a * a).sum() * (ww * b * b).sum() + 1e-9)
    return S


def topk(S):
    ok = np.isfinite(np.diag(S))
    S = S[ok][:, ok]
    r = (S > np.diag(S)[:, None]).sum(1)
    n = len(S)
    return n, (r < 1).mean(), (r < 5).mean(), 1 / n, min(5, n) / n


def learn_w(G, D):
    w = np.zeros(G.shape[1])
    for c in range(G.shape[1]):
        ok = ~np.isnan(G[:, c]) & ~np.isnan(D[:, c])
        if ok.sum() >= 10:
            w[c] = pd.Series(G[ok, c]).corr(pd.Series(D[ok, c]), method="spearman")
    return np.clip(np.nan_to_num(w), 0, None)


def agg_prints(mm, F, side, cells=False, min_n=None):
    """Player-level trait vector: per scene pair (and alignment cell if `cells`) medians, z-scored across players
    within the pair (cell), then averaged over pairs. cells=True removes slot-vs-outside / CB-vs-safety role."""
    r = mm[mm.game == side]
    f = pd.DataFrame(F[r.index.values], columns=FN)
    keys = ["pair", "pos", "p"] if cells else ["pair", "p"]
    f["pair"] = r.pair.values; f["p"] = r.nfl_id.values; f["pos"] = r.pos.fillna("-").values
    gb = f.groupby(keys)
    med = gb[FN].median(); cnt = gb.size()
    med = med[cnt >= (min_n if min_n else (1 if side == 0 else 4))]
    z = med.groupby(level=keys[:-1]).transform(lambda c: (c - c.mean()) / (c.std() + 1e-6))
    return z.groupby(level="p").mean()


def translate_oof(Xd, Yg, folds=5, seed=0, alpha=None):
    """Out-of-fold ridge translation drill traits -> game traits; Gaussian score with the residual covariance."""
    from sklearn.linear_model import Ridge, RidgeCV
    n = len(Xd); Yhat = np.zeros_like(Yg)
    perm = np.random.default_rng(seed).permutation(n)
    for f in range(folds):
        te = perm[f::folds]; tr = np.setdiff1d(np.arange(n), te)
        mdl = RidgeCV(alphas=np.logspace(-1, 3, 13)).fit(Xd[tr], Yg[tr]) if alpha is None else Ridge(alpha).fit(Xd[tr], Yg[tr])
        Yhat[te] = mdl.predict(Xd[te])
    res = Yg - Yhat
    C = np.cov(res.T); C = 0.7 * C + 0.3 * np.diag(np.diag(C))
    Ci = np.linalg.inv(C)
    d = Yg[:, None, :] - Yhat[None, :, :]                       # game i vs predicted-from-drill j
    S = -np.einsum("ijk,kl,ijl->ij", d, Ci, d)
    r2 = 1 - (res ** 2).sum(0) / ((Yg - Yg.mean(0)) ** 2).sum(0)
    return S, r2


def cnn_scores(mm, Xs, tr, te, npairs, steps, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_
    torch.set_num_threads(4); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    L = Xs.shape[2]
    base = Xs.copy(); base[:, :2] /= 10.0; base[:, 2] /= 8.0

    def inp(idx, k, side):
        x = torch.from_numpy(base[idx])
        oh = torch.zeros(len(idx), npairs + 1, L); oh[:, k] = 1; oh[:, npairs] = side
        return torch.cat([x, oh], 1)

    class Enc(nn.Module):
        def __init__(s, d=32):
            super().__init__()
            s.net = nn.Sequential(nn.Conv1d(3 + npairs + 1, 48, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(48, 64, 5, padding=2, stride=2), nn.GELU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(64, d))

        def forward(s, x):
            return F_.normalize(s.net(x), dim=-1)

    idx = {}
    for (p, k, side), r in mm.groupby(["nfl_id", "pair", "game"]):
        idx[(p, k, side)] = r.index.values
    trs = set(tr)
    cand = {}
    for (p, k, side) in idx:
        if p in trs and side == 0 and (p, k, 1) in idx and len(idx[(p, k, 1)]) >= 8:
            cand.setdefault(k, []).append(p)
    enc = Enc(); opt = torch.optim.AdamW(enc.parameters(), 2e-3, weight_decay=1e-4)
    KA, KB, B = 2, 16, 32
    ks = [k for k in cand if len(cand[k]) >= B // 2]
    for s in range(steps):
        k = ks[rng.integers(len(ks))]
        ps = rng.choice(cand[k], size=min(B, len(cand[k])), replace=False)
        a = np.concatenate([rng.choice(idx[(p, k, 0)], KA) for p in ps])
        b = np.concatenate([rng.choice(idx[(p, k, 1)], KB) for p in ps])
        xa, xb = inp(a, k, 0), inp(b, k, 1)
        xa[:, :3] += 0.02 * torch.randn_like(xa[:, :3]); xb[:, :3] += 0.02 * torch.randn_like(xb[:, :3])
        za = F_.normalize(enc(xa).view(len(ps), KA, -1).mean(1), dim=-1)
        zb = F_.normalize(enc(xb).view(len(ps), KB, -1).mean(1), dim=-1)
        lg = za @ zb.T / 0.1; y = torch.arange(len(ps))
        loss = (F_.cross_entropy(lg, y) + F_.cross_entropy(lg.T, y)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    n = len(te); S = np.zeros((n, n)); C = np.zeros((n, n))
    with torch.no_grad():
        for k in range(npairs):
            Z = {}
            for side in (0, 1):
                zs = {}
                for p in te:
                    ii = idx.get((p, k, side))
                    if ii is None or (side == 1 and len(ii) < 8):
                        continue
                    zs[p] = enc(inp(ii, k, side)).mean(0).numpy()
                if len(zs) < 5:
                    Z = None; break
                mu = np.mean(list(zs.values()), 0)
                Z[side] = {p: (v - mu) / (np.linalg.norm(v - mu) + 1e-9) for p, v in zs.items()}
            if Z is None:
                continue
            for i, pg in enumerate(te):
                if pg not in Z[1]:
                    continue
                for j, pd_ in enumerate(te):
                    if pd_ in Z[0]:
                        S[i, j] += Z[1][pg] @ Z[0][pd_]; C[i, j] += 1
    return np.where(C > 0, S / np.maximum(C, 1), -np.inf), loss.item()


def game_game(mm, F, grp, players, data):
    """Identity check inside the game data: first half of a player's games vs the second half."""
    g = mm[mm.game == 1].copy()
    g["gid"] = g.src.str.split("_").str[0].astype(int)
    half = {}
    for p, r in g.groupby("nfl_id"):
        gs = np.sort(r.gid.unique())
        half[p] = set(gs[len(gs) // 2:])
    g["h"] = [int(gid in half[p]) for p, gid in zip(g.nfl_id, g.gid)]
    # role-only prints: shares of alignment / scene / coverage / turn bin
    role = []
    for h in (0, 1):
        r = g[g.h == h]
        parts = [pd.crosstab(r.nfl_id, r[c], normalize="index") for c in ("pos", "scene", "cov", "tb", "pair")
                 if r[c].notna().any()]
        role.append(pd.concat(parts, axis=1).fillna(0))
    cols = role[0].columns.union(role[1].columns)
    R0 = role[0].reindex(index=players, columns=cols); R1 = role[1].reindex(index=players, columns=cols)
    # motion prints z-scored within (pair x alignment) cells, then averaged into pair-level vectors
    mot = []
    for h in (0, 1):
        r = g[g.h == h]
        f = pd.DataFrame(F[r.index.values], columns=FN)
        f[["p", "pair", "pos"]] = r[["nfl_id", "pair", "pos"]].values
        med = f.groupby(["pair", "pos", "p"])[FN].median()
        cnt = f.groupby(["pair", "pos", "p"]).size()
        med = med[cnt >= 4]
        z = med.groupby(level=[0, 1]).transform(lambda c: (c - c.mean()) / (c.std() + 1e-6))
        z = z.groupby(level=[0, 2]).mean()                       # pair x player (alignments pooled after z-scoring)
        mot.append(z.unstack(0))
    cols = mot[0].columns.intersection(mot[1].columns)
    M0 = mot[0].reindex(index=players, columns=cols).values; M1 = mot[1].reindex(index=players, columns=cols).values
    out = {}
    out["role"] = topk(score(R1.values, R0.values))
    out["motion"] = topk(score(M1, M0))
    return out


def main(data, cache, folds=2, steps=1500, seeds=(0, 1, 2)):
    res = []
    for grp in ("WR", "DB"):
        mm, Xs, F, names = build(data, cache, grp)
        npairs = len(names)
        print(f"\n=== {grp}: pairs", names, flush=True)
        for k, nm in enumerate(names):
            r = mm[mm.pair == k]
            print(f"  {nm}: drill segs {(r.game == 0).sum()} ({r[r.game == 0].nfl_id.nunique()} pl), "
                  f"game segs {(r.game == 1).sum()} ({r[r.game == 1].nfl_id.nunique()} pl)", flush=True)
        dpl = set(mm[mm.game == 0].nfl_id); gc = mm[mm.game == 1].groupby("nfl_id").size()
        players = np.array(sorted(dpl & set(gc.index[gc >= 30])))
        print(f"  players with drills and >= 30 matched game segments: {len(players)}", flush=True)
        # (1) hand, no learning, all players at once (chance ~5%)
        pr = prints(mm, F, players)
        G, D = stack(pr, players, npairs, 1), stack(pr, players, npairs, 0)
        n, t1, t5, c1, c5 = topk(score(G, D))
        print(f"  hand all-players: n {n} top1 {t1:.3f} top5 {t5:.3f} chance {c1:.3f}/{c5:.3f}", flush=True)
        res.append((grp, "hand_all", -1, n, t1, t5, c1, c5))
        # per-feature cross-side Spearman (all players; descriptive)
        wr = learn_w_raw(G, D).reshape(npairs, len(FN))
        print("  cross-side Spearman by feature (mean over pairs):", dict(zip(FN, np.round(np.nanmean(wr, 0), 2))),
              flush=True)
        print("  by pair (mean over features):", dict(zip(names, np.round(np.nanmean(wr, 1), 2))), flush=True)
        # (2) K-fold by player: hand, handw, cnn
        for seed in seeds:
            perm = np.random.default_rng(seed).permutation(players)
            for f in range(folds):
                te = np.array(sorted(perm[f::folds])); tr = np.setdiff1d(players, te)
                prt = prints(mm, F, tr); pre = prints(mm, F, te)
                w = learn_w(stack(prt, tr, npairs, 1), stack(prt, tr, npairs, 0))
                Gt, Dt = stack(pre, te, npairs, 1), stack(pre, te, npairs, 0)
                for name, S in (("hand", score(Gt, Dt)), ("handw", score(Gt, Dt, w))):
                    res.append((grp, name, seed * 10 + f) + topk(S))
                if steps:
                    S, lo = cnn_scores(mm, Xs, tr, te, npairs, steps, seed * 10 + f)
                    res.append((grp, "cnn", seed * 10 + f) + topk(S))
                    Sh = score(Gt, Dt, w)
                    zc = lambda A: (A - np.nanmean(np.where(np.isfinite(A), A, np.nan))) / np.nanstd(np.where(np.isfinite(A), A, np.nan))
                    res.append((grp, "cnn+handw", seed * 10 + f) + topk(np.where(np.isfinite(S) & np.isfinite(Sh), zc(S) + zc(Sh), -np.inf)))
                print(f"  seed {seed} fold {f}: " + " | ".join(f"{r[1]} {r[5]:.2f}/{r[7]:.2f}" for r in res[-4:]), flush=True)
        # (2b) ridge translation of player-level drill traits into game traits (OOF, ranked among all players)
        Xa = agg_prints(mm, F, 0)
        Xp = []                                                  # per-pair drill dims as extra inputs
        for k in range(npairs):
            r = mm[(mm.game == 0) & (mm.pair == k)]
            f = pd.DataFrame(F[r.index.values], columns=[f"{c}_{k}" for c in FN]).assign(p=r.nfl_id.values)
            med = f.groupby("p").median(); Xp.append((med - med.mean()) / (med.std() + 1e-6))
        Xp = pd.concat(Xp, axis=1)
        for tgt in ("plain", "cells"):
            Yg = agg_prints(mm, F, 1, cells=(tgt == "cells"), min_n=4)
            for inp_name, Xin in (("agg", Xa), ("agg+pair", pd.concat([Xa, Xp], axis=1))):
                ix = Xin.index.intersection(Yg.index).intersection(players)
                A = Xin.loc[ix].fillna(0).values; Y = Yg.loc[ix].fillna(0).values
                for seed in seeds:
                    S, r2 = translate_oof(A, Y, seed=seed)
                    res.append((grp, f"ridge_{inp_name}_{tgt}", seed) + topk(S))
                print(f"  ridge {inp_name} -> game[{tgt}]: n {len(ix)} top5 "
                      f"{np.mean([x[5] for x in res[-len(seeds):]]):.3f} chance {5/len(ix):.3f}  OOF R2 by feature "
                      + str(dict(zip(FN, np.round(r2, 2)))), flush=True)
            # no-learning aggregated correlation (signed traits, all players)
            ix = Xa.index.intersection(Yg.index).intersection(players)
            res.append((grp, f"aggcorr_{tgt}", -1) + topk(score(Yg.loc[ix].values, Xa.loc[ix].values)))
        # (3) game -> game identity, and whether it is role
        gg = game_game(mm, F, grp, players, data)
        for k, v in gg.items():
            res.append((grp, "gg_" + k, -1) + v)
            print(f"  game->game {k}: n {v[0]} top1 {v[1]:.3f} top5 {v[2]:.3f} chance {v[3]:.3f}/{v[4]:.3f}", flush=True)
    r = pd.DataFrame(res, columns=["grp", "method", "fold", "n", "top1", "top5", "ch1", "ch5"])
    s = r.groupby(["grp", "method"])[["n", "top1", "top5", "ch1", "ch5"]].mean()
    s["x_chance5"] = s.top5 / s.ch5
    print("\n", s.round(3).to_string(), flush=True)
    r.to_csv(os.path.join(cache, "embed_route_res.csv"), index=False)
    return r


def learn_w_raw(G, D):
    w = np.full(G.shape[1], np.nan)
    for c in range(G.shape[1]):
        ok = ~np.isnan(G[:, c]) & ~np.isnan(D[:, c])
        if ok.sum() >= 10:
            w[c] = pd.Series(G[ok, c]).corr(pd.Series(D[ok, c]), method="spearman")
    return w


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 2, int(a[4]) if len(a) > 4 else 1500)

"""ML pillar 1: contrastive embedding of cut windows (segs.py), drill <-> game, the same player as the positive pair.

Encoder: small 1D-CNN over the 21-frame cut window (x, y, speed in the cut-centred frame) -> 32-d unit vector.
Training: each step takes B players; for each one drill window and one game window; symmetric InfoNCE across players.
Evaluation (players never seen in training, K-fold by player): a player's drill print = mean of his drill-window
embeddings, his game print = mean over game windows; for each held-out player, rank the held-out players of the same
position group by cosine(game print, drill print). Reports top-1 / top-5 accuracy against chance, and the same for a
no-learning baseline (hand-made cut features: loss, v_in, v_min, decel, t_rec by angle bin).
Usage: python embed.py DATA_DIR CUTS_DIR [folds] [steps]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)


class Enc(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 64, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Linear(64, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def prints(enc, X, ids, game, players, src_flag):
    out = {}
    with torch.no_grad():
        for p in players:
            m = (ids == p) & (game == src_flag)
            z = enc(torch.from_numpy(X[m]))
            out[p] = F.normalize(z.mean(0), dim=0).numpy()
    return out


def rank_eval(dp, gp, players):
    P = [p for p in players if p in dp and p in gp]
    D = np.stack([dp[p] for p in P]); G = np.stack([gp[p] for p in P])
    S = G @ D.T
    r = (S > np.diag(S)[:, None]).sum(1)              # rank of the true player (0 = top)
    return len(P), (r < 1).mean(), (r < 5).mean()


def feat_prints(cuts, players, src):
    c = cuts[cuts.source == src].copy()
    c["ab"] = pd.cut(c.angle, [30, 60, 90, 120, 180])
    f = c.groupby(["nfl_id", "ab"], observed=True)[["loss", "v_in", "v_min", "decel"]].median().unstack()
    f = (f - f.mean()) / f.std()
    f = f.fillna(0)
    return {p: f.loc[p].values / (np.linalg.norm(f.loc[p].values) + 1e-9) for p in players if p in f.index}


def main(data, cd, folds=2, steps=3000, B=48, seed=0, mode="dg", KA=8, KB=16):
    Z = np.load(os.path.join(cd, "segs.npz"), allow_pickle=True)
    X, ids, game = Z["X"], Z["nfl_id"], Z["game"]
    X = X.copy(); X[:, :2] /= 5.0; X[:, 2] /= 8.0              # rough scale to ~1
    pl = pd.read_csv(os.path.join(data, "players.csv")).set_index("nfl_id").nfl_position
    grp = pd.Series(ids).map(lambda p: "WR" if pl.get(p) == "WR" else "DB").values
    cnt = pd.DataFrame({"p": ids, "g": game}).groupby(["p", "g"]).size().unstack(fill_value=0)
    ok = cnt.index[(cnt.get(0, 0) >= 5) & (cnt.get(1, 0) >= 30)].values
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ok)
    fold = {p: i % folds for i, p in enumerate(perm)}
    cuts = pd.concat([pd.read_parquet(os.path.join(cd, "drill_cuts.parquet")).assign(source="drill"),
                      pd.read_parquet(os.path.join(cd, "game_cuts.parquet")).assign(source="game")])
    idx = {p: (np.nonzero((ids == p) & (game == 0))[0], np.nonzero((ids == p) & (game == 1))[0]) for p in ok}
    res = []
    for k in range(folds):
        tr = [p for p in ok if fold[p] != k]; te = [p for p in ok if fold[p] == k]
        torch.manual_seed(seed + k)
        enc = Enc(); opt = torch.optim.AdamW(enc.parameters(), 2e-3, weight_decay=1e-4)
        tau = 0.1
        for s in range(steps):
            ps = rng.choice(tr, size=min(B, len(tr)), replace=False)
            # set-level pairs: mean of KA drill windows vs mean of KB game windows of the same player
            if mode == "gg":                                   # identity check: two disjoint game sets
                a = np.stack([rng.choice(idx[p][1], KA) for p in ps]); b = np.stack([rng.choice(idx[p][1], KB) for p in ps])
            else:
                a = np.stack([rng.choice(idx[p][0], KA) for p in ps]); b = np.stack([rng.choice(idx[p][1], KB) for p in ps])
            xa = torch.from_numpy(X[a.ravel()]); xb = torch.from_numpy(X[b.ravel()])
            xa = xa + 0.02 * torch.randn_like(xa); xb = xb + 0.02 * torch.randn_like(xb)
            za = F.normalize(enc(xa).view(len(ps), KA, -1).mean(1), dim=-1)
            zb = F.normalize(enc(xb).view(len(ps), KB, -1).mean(1), dim=-1)
            lg = za @ zb.T / tau
            y = torch.arange(len(ps))
            loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 1000 == 0:
                print(f"fold {k} step {s} loss {loss.item():.3f}", flush=True)
        enc.eval()
        if mode == "gg":                                       # game first half vs second half (by window order)
            dp, gp = {}, {}
            with torch.no_grad():
                for p in te:
                    ii = idx[p][1]; h = len(ii) // 2
                    dp[p] = F.normalize(enc(torch.from_numpy(X[ii[:h]])).mean(0), dim=0).numpy()
                    gp[p] = F.normalize(enc(torch.from_numpy(X[ii[h:]])).mean(0), dim=0).numpy()
        else:
            dp = prints(enc, X, ids, game, te, 0); gp = prints(enc, X, ids, game, te, 1)
        fd, fg = feat_prints(cuts, te, "drill"), feat_prints(cuts, te, "game")
        for gname in ("WR", "DB"):
            members = [p for p in te if ("WR" if pl.get(p) == "WR" else "DB") == gname]
            n, t1, t5 = rank_eval(dp, gp, members)
            nb, b1, b5 = rank_eval(fd, fg, members)
            res.append((k, gname, n, t1, t5, 1 / n, min(5, n) / n, b1, b5))
            print(f"fold {k} {gname}: n {n}  model top1 {t1:.3f} top5 {t5:.3f} | chance {1/n:.3f} {min(5,n)/n:.3f} | "
                  f"hand features top1 {b1:.3f} top5 {b5:.3f}", flush=True)
    r = pd.DataFrame(res, columns=["fold", "grp", "n", "top1", "top5", "ch1", "ch5", "hand1", "hand5"])
    print(r.groupby("grp")[["n", "top1", "top5", "ch1", "ch5", "hand1", "hand5"]].mean().round(3))
    return r


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 2, int(a[4]) if len(a) > 4 else 3000, mode=a[5] if len(a) > 5 else "dg")

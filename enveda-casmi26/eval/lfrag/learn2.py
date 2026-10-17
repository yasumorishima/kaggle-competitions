import sys, pickle, numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-kaggle-competitions/e14dc4c1-b1cc-5b07-bb7b-e4b5776308f1/scratchpad/lf"); import feat2 as feat
R = pickle.load(open("/tmp/claude-0/-home-user-kaggle-competitions/e14dc4c1-b1cc-5b07-bb7b-e4b5776308f1/scratchpad/lf/panels2.pkl", "rb"))
e180 = [r for r in R if r[0] == "e180"]; rng = np.random.RandomState(0); rng.shuffle(e180)
sets = {"train": e180[500:] + [r for r in R if r[0] == "oth"], "e180_ho": e180[:500], "np": [r for r in R if r[0] == "np"]}
def rr(sc, t): return 1 / (1 + (sc > sc[t]).sum() + 0.5 * ((sc == sc[t]).sum() - 1))
def mrr(rs, f): return np.mean([rr(f(F), t) for _, F, t, ok in rs])
NF = feat.NF
def fit(rs, l2=1e-3, it=400, lr=0.5):
    w = np.zeros(NF); w[:feat.NH] = 1; w[feat.NH:2 * feat.NH] = 0.6   # start at the base score
    w0 = w.copy()
    for _ in range(it):
        g = np.zeros(NF)
        for _, F, t, ok in rs:
            s = F @ w; s = s - s.max(); p = np.exp(s * 10); p /= p.sum()   # temperature 0.1
            g += 10 * (F[t] - p @ F)
        w += lr * (g / len(rs) - l2 * (w - w0))
    return w
print({k: len(v) for k, v in sets.items()})
base = lambda F: np.array([feat.base_score(f) for f in F])
for k in ("e180_ho", "np", "train"): print("base", k, "%.3f" % mrr(sets[k], base))
for l2 in (1e-2, 1e-3):
    w = fit(sets["train"], l2)
    for k in ("e180_ho", "np", "train"): print("learned l2", l2, k, "%.3f" % mrr(sets[k], lambda F: F @ w))
    np.save(f"/tmp/claude-0/-home-user-kaggle-competitions/e14dc4c1-b1cc-5b07-bb7b-e4b5776308f1/scratchpad/lf/w2_{l2}.npy", w)
print("w (lin lvl01 | lin lvl2 | sqrt lvl01 | sqrt lvl2 | bias) per dh", feat.HS); print(np.round(w, 2))

"""Per-candidate features for a learned fragment-credit model (ours): intensity share of peaks whose cheapest
explanation (fewest cleavages, then smallest |dh|) falls in bucket (level 0/1 vs 2) x dh in -2..3; linear and sqrt."""
import sys, random, numpy as np, pyarrow.parquet as pq, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-kaggle-competitions/e14dc4c1-b1cc-5b07-bb7b-e4b5776308f1/scratchpad/pub")
import frag_rescore_451 as R5
C = R5.resolve_cfg(None); HS = C['h_shifts']; NH = len(HS)
NF = 2 * 2 * NH + 1
def spec_feats(lv, s):
    mz = np.asarray(s['mz'], float); it = np.asarray(s['it'], float)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0); mz, it = mz[ok], it[ok]
    if len(mz) == 0: return None
    car = R5._carriers(s.get('adduct'), s.get('mode'))
    shifts = np.array([c + dh * R5.H for c in car for dh in HS]); ns = len(shifts)
    q = (mz[None, :] - shifts[:, None])
    best_l = np.full(len(mz), 9); best_d = np.zeros(len(mz), int); best_a = np.full(len(mz), 99)
    for L, arr in enumerate(lv[:3]):
        if len(arr) == 0: continue
        a = np.searchsorted(arr, q - C['tol'], 'left'); b = np.searchsorted(arr, q + C['tol'], 'right')
        hit = (b > a)                                  # (ns, npk)
        lvl = 0 if L <= 1 else 1
        for si in range(ns):
            di = si % NH; ad = abs(HS[di])
            m = hit[si] & ((lvl < best_l) | ((lvl == best_l) & (ad < best_a)))
            best_l = np.where(m, lvl, best_l); best_d = np.where(m, di, best_d); best_a = np.where(m, ad, best_a)
    f = np.zeros(NF)
    for w_i, w in enumerate((it / it.sum(), np.sqrt(it) / np.sqrt(it).sum())):
        for k in range(len(mz)):
            if best_l[k] < 9:
                f[w_i * 2 * NH + best_l[k] * NH + best_d[k]] += w[k]
    f[-1] = 1.0
    return f
def cand_feats(smi, spectra, cache):
    lv = R5._levels_cached(smi, C, cache)
    if lv is None: return None
    fs = [x for x in (spec_feats(lv, s) for s in spectra) if x is not None]
    return np.mean(fs, 0) if fs else np.zeros(NF)
def base_score(f):  # reproduces frag_rescore's default score from the linear-intensity buckets
    return f[:NH].sum() + 0.6 * f[NH:2 * NH].sum()

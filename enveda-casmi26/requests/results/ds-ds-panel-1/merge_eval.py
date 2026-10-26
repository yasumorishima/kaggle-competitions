"""End-to-end merge replay (main list + PubChem list + gate) for three simulated classes on the NP panel.
usage: python merge_eval.py <main_out> <tables> <pool_meta> <poolfz.parquet> <out_json>"""
import json, os, pickle, sys
import numpy as np, pandas as pd
MAIN, TAB, POOL, PFZ, OUTJ = sys.argv[1:6]
LIB_TAU, REL_TH = 0.9, 600.0
AGG, GENTLE, FIRST = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20], [1, 3, 5, 7, 9]
S = {s: pickle.load(open(os.path.join(MAIN, s + '.pkl'), 'rb')) for s in ('lib', 'nospec', 'dbdrop')}
keys = sorted(k for k in S['nospec'] if all('error' not in S[s][k] for s in S))
pm = pd.read_parquet(POOL, columns=['key']); pool_all = np.array(sorted(k.encode() for k in pm.key), 'S14')
pfz = pd.read_parquet(PFZ).set_index('key'); rng = np.random.default_rng(0)

def main_list(rec, mu=0.15, topn=60):
    ks, sc, pid, pop = rec['keys'], rec['score'], rec['pid'], rec['pop']
    seen, idx = set(), []
    for i, k in enumerate(ks):
        if k not in seen:
            seen.add(k); idx.append(i)
        if len(idx) >= topn: break
    idx = np.array(idx); s = sc[idx].astype(np.float64); z = (s - s.mean()) / (s.std() + 1e-9); isg = pid[idx] < 0
    f = z + mu * np.where(isg, 0.0, pop[idx]); o = np.empty(len(idx), np.int64); pi = np.where(~isg)[0]
    o[isg] = np.where(isg)[0]; o[~isg] = pi[np.argsort(-f[pi], kind='stable')]
    return [ks[i] for i in idx[o]][:25]

PCL = {}
for k in keys:
    d = np.load(os.path.join(TAB, k + '.npz'))
    if int(d['n']) == 0:
        PCL[k] = None; continue
    fz = d['fz']; ok = np.isfinite(fz); ident = np.where(d['sk'] != b'', d['sk'], d['ik']); inpool = np.isin(ident, pool_all)
    pop = d['lsid'].astype(np.float32) + d['lpmid'].astype(np.float32); sc1 = np.where(ok, d['sc1'], -np.inf); nok = int(ok.sum())
    short = np.where(ok)[0] if 5000 >= nok else np.argpartition(-sc1, 4999)[:5000]
    short = np.union1d(short[ok[short]], np.argpartition(-np.where(ok, pop, -np.inf), min(200, nok) - 1)[:min(200, nok)])
    f = fz[short].astype(np.float64); z = (f - f.mean()) / (f.std() + 1e-9); sc = z + 0.25 * pop[short]
    out = {}
    for tag, excl in (('A', inpool), ('B', inpool & ~d['is_truth'])):          # A: truth stays in the pool; B: truth only in PubChem
        order = short[np.argsort(-sc, kind='stable')]; scs = np.sort(sc)[::-1]
        lst, seen, top_sc, top_pop = [], set(), None, 0.0
        for j, i in enumerate(order):
            kk = ident[i]
            if kk in seen: continue
            seen.add(kk)
            if excl[i]: continue
            if top_sc is None: top_sc = float(scs[j]); top_pop = float(pop[i])
            lst.append(k if d['is_truth'][i] else kk.decode())
            if len(lst) >= 25: break
        m = ~excl[short]; fz_top = float(f[m].max()) if m.any() else None
        out[tag] = (lst, fz_top, top_sc, top_pop)
    PCL[k] = out

def merge(base, pc, slots, n=25):
    bk = set(base); pcs = [x for x in pc if x not in bk]; out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs): out.append(pcs[pj]); pj += 1
        elif bi < len(base): out.append(base[bi]); bi += 1
        elif pj < len(pcs): out.append(pcs[pj]); pj += 1
    return out

def rr(lst, key):
    return next((1.0 / (j + 1) for j, x in enumerate(lst[:25]) if x == key), 0.0)

def final(scen, k, rule):
    rec = S[scen][k]; base = main_list(rec)
    if PCL[k] is None: return base
    tag = 'B' if scen == 'dbdrop' else 'A'; lst, fz_top, top_sc, top_pop = PCL[k][tag]
    if not lst or rec['lib_max'] >= LIB_TAU: return base
    bp = pfz.best_excl[k] if tag == 'B' else pfz.best_incl[k]
    rel = (fz_top - bp) if (fz_top is not None and np.isfinite(bp)) else 1e9
    slots = rule(rel, top_sc, top_pop)
    if isinstance(slots, tuple):                      # ('promote', fallback_slots): the confident top proposal leads the usual list
        usual = merge(base, lst, slots[1])
        return ([lst[0]] + [x for x in usual if x != lst[0]])[:25]
    return merge(base, lst, slots) if slots else base

v19 = lambda rel: AGG if rel > REL_TH else GENTLE
rules = {
 'no PubChem insertion': lambda rel, s, p: None,
 'v19 gate': lambda rel, s, p: v19(rel),
 'promote if S>6': lambda rel, s, p: ('promote', v19(rel)) if s > 6 else v19(rel),
 'promote if S>6 and pop>=5': lambda rel, s, p: ('promote', v19(rel)) if (s > 6 and p >= 5) else v19(rel),
 'promote if S>6; early slots if S>5': lambda rel, s, p: ('promote', v19(rel)) if s > 6 else (AGG if (s > 5 or rel > REL_TH) else GENTLE),
 'promote if S>6 and pop>=5; early slots if S>5': lambda rel, s, p: ('promote', v19(rel)) if (s > 6 and p >= 5) else (AGG if (s > 5 or rel > REL_TH) else GENTLE),
 'promote if S>5.5 and pop>=8; early slots if S>5': lambda rel, s, p: ('promote', v19(rel)) if (s > 5.5 and p >= 8) else (AGG if (s > 5 or rel > REL_TH) else GENTLE),
}
res = {}
for name, rule in rules.items():
    res[name] = {sc: [rr(final(sc, k, rule), k) for k in keys] for sc in S}
ref = res['v19 gate']
out = []
for name in rules:
    row = dict(name=name)
    for sc in S:
        v = np.array(res[name][sc]); d = v - np.array(ref[sc]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        row[sc] = dict(mrr25=float(v.mean()), top1=float((v == 1).mean()), delta=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    out.append(row)
    print(f"{name:<58} " + ' | '.join(f"{sc} {row[sc]['mrr25']:.4f} ({row[sc]['delta']:+.4f} [{row[sc]['ci'][0]:+.3f},{row[sc]['ci'][1]:+.3f}])" for sc in S))
fired = {sc: int(sum(1 for k in keys if PCL[k] is not None and S[sc][k]['lib_max'] < LIB_TAU and PCL[k]['B' if sc == 'dbdrop' else 'A'][0]
                       and PCL[k]['B' if sc == 'dbdrop' else 'A'][2] > 6 and PCL[k]['B' if sc == 'dbdrop' else 'A'][3] >= 5)) for sc in S}
correct = int(sum(1 for k in keys if PCL[k] is not None and S['dbdrop'][k]['lib_max'] < LIB_TAU and PCL[k]['B'][0] and PCL[k]['B'][2] > 6
                  and PCL[k]['B'][3] >= 5 and PCL[k]['B'][0][0] == k))
print('promotions fired', fired, 'correct in dbdrop', correct)
json.dump(dict(n=len(keys), rules=out, promotion=dict(rule='S>6 and pop>=5', fired=fired, correct_when_removed_from_pool=correct)), open(OUTJ, 'w'), indent=1)

"""Offline replay of main-channel and merge variants on the NP-example panel (development proxy, not clean CV).

usage: python main_eval.py <main_out_dir> <np_tables_dir> <pool_meta.parquet> <poolfz.parquet> <out_json>
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd

MAIN, TAB, POOL, PFZ, OUTJ = sys.argv[1:6]
LIB_TAU, REL_TH = 0.9, 600.0
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]
S = {s: pickle.load(open(os.path.join(MAIN, s + '.pkl'), 'rb')) for s in ('lib', 'nospec', 'dbdrop') if os.path.exists(os.path.join(MAIN, s + '.pkl'))}
keys = sorted(set.intersection(*[set(k for k, r in d.items() if 'error' not in r) for d in S.values()]))
pm = pd.read_parquet(POOL, columns=['key', 'src'])
pool_keys = set(k.encode() for k in pm.key); H = set(k.encode() for k in keys)
pool_ex = np.array(sorted(pool_keys - H), 'S14')
pfz = pd.read_parquet(PFZ).set_index('key')
rng = np.random.default_rng(0)


def main_list(rec, key, mu=0.0, topn=60, gen_fixed=True):
    """Distinct-key main list, optionally re-ordered by z(score) + mu * pop inside the top `topn`."""
    ks, sc, pid, pop = rec['keys'], rec['score'], rec['pid'], rec['pop']
    seen, idx = set(), []
    for i, k in enumerate(ks):
        if k not in seen:
            seen.add(k); idx.append(i)
        if len(idx) >= topn:
            break
    idx = np.array(idx)
    if mu > 0 and len(idx) > 1:
        s = sc[idx].astype(np.float64); z = (s - s.mean()) / (s.std() + 1e-9)
        isg = pid[idx] < 0
        f = z + mu * np.where(isg, 0.0, pop[idx])
        o = np.empty(len(idx), np.int64)
        if gen_fixed:
            pi = np.where(~isg)[0]; o[isg] = np.where(isg)[0]; o[~isg] = pi[np.argsort(-f[pi], kind='stable')]
        else:
            o = np.argsort(-f, kind='stable')
        idx = idx[o]
    return [ks[i] for i in idx]


def rr(lst, key):
    for j, k in enumerate(lst[:25]):
        if k == key:
            return 1.0 / (j + 1)
    return 0.0


def pc_list(key, n1=5000, lam=0.25, union=200):
    p = os.path.join(TAB, key + '.npz')
    d = np.load(p)
    if int(d['n']) == 0:
        return [], None
    fz = d['fz']; ok = np.isfinite(fz); sc1 = np.where(ok, d['sc1'], -np.inf)
    ident = np.where(d['sk'] != b'', d['sk'], d['ik']); inpool = np.isin(ident, pool_ex) & ~d['is_truth']
    pop = d['lsid'].astype(np.float32) + d['lpmid'].astype(np.float32)
    nok = int(ok.sum())
    short = np.where(ok)[0] if n1 >= nok else np.argpartition(-sc1, n1 - 1)[:n1]
    short = short[ok[short]]
    if union and lam > 0:
        short = np.union1d(short, np.argpartition(-np.where(ok, pop, -np.inf), min(union, nok) - 1)[:min(union, nok)])
    f = fz[short].astype(np.float64); z = (f - f.mean()) / (f.std() + 1e-9)
    order = short[np.argsort(-(z + lam * pop[short]), kind='stable')]
    raw_order = short[np.argsort(-f, kind='stable')]
    fz_top = next((float(fz[i]) for i in raw_order if not inpool[i]), None)
    out, seen = [], set()
    for i in order:
        k = ident[i]
        if k in seen:
            continue
        seen.add(k)
        if inpool[i]:
            continue
        out.append(key if d['is_truth'][i] else k.decode())
        if len(out) >= 25:
            break
    return out, fz_top


def merge(base, pc, slots, n=25):
    bk = set(base); pcs = [k for k in pc if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out


def summarize(name, vals, base=None):
    v = np.array(vals); r = dict(name=name, mrr25=float(v.mean()), hit25=float((v > 0).mean()), top1=float((v == 1).mean()))
    if base is not None:
        d = v - np.array(base); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        r.update(delta=float(d.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    return r


out = dict(n=len(keys), scenarios={})
for sc_name, D in S.items():
    rows = []
    cov = np.mean([D[k].get('truth_rank') is not None for k in keys])
    libmax = np.array([D[k]['lib_max'] for k in keys])
    ref = [rr(main_list(D[k], k, mu=0.15), k) for k in keys]                     # v19 main-list setting
    rows.append(summarize('ranker only (mu=0)', [rr(main_list(D[k], k, 0.0), k) for k in keys], ref))
    for mu in (0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00, 2.00):
        rows.append(summarize(f'pop mu={mu:.2f} top60', [rr(main_list(D[k], k, mu), k) for k in keys], ref))
    for mu in (0.15, 0.40, 1.00):
        rows.append(summarize(f'pop mu={mu:.2f} top200', [rr(main_list(D[k], k, mu, topn=200), k) for k in keys], ref))
    rows.append(summarize('pop mu=0.40 top60, generated may move', [rr(main_list(D[k], k, 0.40, gen_fixed=False), k) for k in keys], ref))
    out['scenarios'][sc_name] = dict(truth_among_candidates=float(cov), lib_max_ge_0_9=float((libmax >= LIB_TAU).mean()),
                                     lib_max_median=float(np.median(libmax)), n_cand_median=float(np.median([D[k]['n_cand'] for k in keys])),
                                     variants=rows)

if 'dbdrop' in S:                      # full merge: the truth can only come from the PubChem channel (or generation)
    D = S['dbdrop']; PC = {k: pc_list(k) for k in keys}
    PCnp = {k: pc_list(k, lam=0.0, union=0) for k in keys}

    def final(k, slots_fn, mu=0.15, pc=PC, tau=LIB_TAU):
        base = main_list(D[k], k, mu)[:25]
        lst, fz_top = pc[k]
        if not lst or D[k]['lib_max'] >= tau:
            return base
        bp = pfz.best_excl.get(k, np.nan)
        rel = (fz_top - bp) if (fz_top is not None and np.isfinite(bp)) else 1e9
        return merge(base, lst, slots_fn(rel))

    cur = lambda rel: SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
    ref = [rr(final(k, cur), k) for k in keys]
    rows = [summarize('v19 gate (slots 2,4,.. or 4,8,..)', ref, ref),
            summarize('v18 gate (no popularity in PubChem list)', [rr(final(k, cur, pc=PCnp), k) for k in keys], ref),
            summarize('always aggressive 2,4,6,8,10', [rr(final(k, lambda r: SLOTS_AGG), k) for k in keys], ref),
            summarize('PubChem first: slots 1,3,5,7,9 when rel>600 else 2,4,6,8,10',
                      [rr(final(k, lambda r: [1, 3, 5, 7, 9] if r > REL_TH else SLOTS_AGG), k) for k in keys], ref),
            summarize('PubChem first always: slots 1,3,5,7,9', [rr(final(k, lambda r: [1, 3, 5, 7, 9]), k) for k in keys], ref),
            summarize('ten PubChem slots 2,4,..,20', [rr(final(k, lambda r: list(range(2, 21, 2))), k) for k in keys], ref),
            summarize('no lib_max gate, v19 slots', [rr(final(k, cur, tau=9.9), k) for k in keys], ref),
            summarize('PubChem list alone', [rr(PC[k][0], k) for k in keys], ref)]
    libmax = np.array([D[k]['lib_max'] for k in keys])
    rel = np.array([(PC[k][1] - pfz.best_excl.get(k, np.nan)) if PC[k][1] is not None else np.nan for k in keys])
    out['merge_dbdrop'] = dict(variants=rows, gate_blocked_by_lib=float((libmax >= LIB_TAU).mean()), aggressive_share=float(np.nanmean(rel > REL_TH)),
                               rel_quantiles=[float(x) for x in np.nanquantile(rel, [.1, .25, .5, .75, .9])])
json.dump(out, open(OUTJ, 'w'), indent=1)
for sc_name, r in out['scenarios'].items():
    print(f"== {sc_name}: truth among candidates {r['truth_among_candidates']:.3f} | lib_max>=0.9 {r['lib_max_ge_0_9']:.3f} | median candidates {r['n_cand_median']:.0f}")
    for v in r['variants']:
        print(f"   {v['name']:<40} MRR@25 {v['mrr25']:.4f} top1 {v['top1']:.3f} hit25 {v['hit25']:.3f}  d {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
if 'merge_dbdrop' in out:
    m = out['merge_dbdrop']; print('== merge on dbdrop: gate blocked by lib_max', m['gate_blocked_by_lib'], 'aggressive share', m['aggressive_share'], 'rel q', m['rel_quantiles'])
    for v in m['variants']:
        print(f"   {v['name']:<58} MRR@25 {v['mrr25']:.4f} top1 {v['top1']:.3f} hit25 {v['hit25']:.3f}  d {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")

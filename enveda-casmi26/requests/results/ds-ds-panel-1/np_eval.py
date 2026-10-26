"""Replay PubChem-channel variants on the NP-example candidate tables (see np_tables.py).

usage: python np_eval.py <tables_dir> <pool_meta.parquet> <out_json>
Every variant follows the v19 channel: pass-1 shortlist by partial score, optional popularity union, full f.z,
score = z(f.z over the shortlist) + lam * prior, walk down skipping pool structures and repeated identities, keep 25.
The panel's own truths are removed from the pool-exclusion set (known structure, no reference spectrum).
"""
import json, os, sys
import numpy as np
import pandas as pd

TAB, POOL, OUTJ = sys.argv[1:4]
panel = pd.read_parquet(os.path.join(TAB, 'panel.parquet'))
panel = panel[panel.status == 'ok'].reset_index(drop=True)
pm = pd.read_parquet(POOL, columns=['key', 'src'])
pool_keys = set(k.encode() for k in pm.key)
coco = set(pm.key[pm.src >= 2])
panel['in_coconut'] = panel.key.isin(coco)
H = set(k.encode() for k in panel.key)
pool_ex = np.array(sorted(pool_keys - H), 'S14')
T = {}
for k in panel.key:
    d = np.load(os.path.join(TAB, k + '.npz'))
    if int(d['n']) == 0:
        T[k] = None; continue
    t = {f: d[f] for f in d.files}
    ident = np.where(t['sk'] != b'', t['sk'], t['ik'])              # scorer key where computed, else plain connectivity
    t['ident'] = ident
    t['inpool'] = np.isin(ident, pool_ex) & ~t['is_truth']
    t['pop'] = t['lsid'].astype(np.float32) + t['lpmid'].astype(np.float32)
    t['ok'] = np.isfinite(t['fz'])
    T[k] = t


def group_top(score, groups, n):
    """Indices of all rows belonging to the n groups with the highest max score."""
    order = np.argsort(-score, kind='stable')
    seen, keep = set(), []
    for i in order:
        g = groups[i]
        if g not in seen:
            if len(seen) >= n:
                break
            seen.add(g)
    return np.where(np.isin(groups, np.array(list(seen), groups.dtype)))[0]


def run(t, n1=5000, lam=0.25, union=200, dedup=False, prior=None, best_per_group=False):
    ok = t['ok']; idx_all = np.where(ok)[0]
    if not len(idx_all):
        return dict(short=False, rank=None)
    sc1 = np.where(ok, t['sc1'], -np.inf)
    pri = t['pop'] if prior is None else prior(t)
    if n1 is None or n1 >= len(idx_all):
        short = idx_all
    elif dedup:
        short = group_top(sc1, t['ik'], n1)
        short = short[ok[short]]
    else:
        short = np.argpartition(-sc1, n1 - 1)[:n1]
        short = short[ok[short]]
    if union and lam > 0:
        pp = np.where(ok, pri, -np.inf)
        if dedup:
            u = group_top(pp, t['ik'], union); u = u[ok[u]]
        else:
            ku = min(union, len(idx_all)); u = np.argpartition(-pp, ku - 1)[:ku]
        short = np.union1d(short, u)
    fz = t['fz'][short].astype(np.float64)
    z = (fz - fz.mean()) / (fz.std() + 1e-9)
    sc = z + lam * pri[short]
    order = short[np.argsort(-sc, kind='stable')]
    in_short = bool(t['is_truth'][short].any())
    seen, rank = set(), 0
    for i in order:
        k = t['ident'][i]
        if k in seen:
            continue
        seen.add(k)
        if t['inpool'][i]:
            continue
        rank += 1
        if t['is_truth'][i]:
            return dict(short=in_short, rank=rank, n_short=len(short))
        if rank >= 200:
            break
    return dict(short=in_short, rank=None, n_short=len(short))


def evaluate(name, **kw):
    rr, rec25, top1, short, nshort, r200 = [], [], [], [], [], []
    for k in panel.key:
        t = T[k]
        if t is None or not t['is_truth'].any():
            rr.append(0.0); rec25.append(0); top1.append(0); short.append(0); r200.append(0); continue
        r = run(t, **kw)
        rank = r['rank']
        rr.append(1.0 / rank if rank and rank <= 25 else 0.0); rec25.append(int(bool(rank and rank <= 25)))
        top1.append(int(rank == 1)); short.append(int(r['short'])); nshort.append(r.get('n_short', 0)); r200.append(int(bool(rank)))
    return dict(name=name, mrr25=float(np.mean(rr)), recall25=float(np.mean(rec25)), top1=float(np.mean(top1)),
                recall200=float(np.mean(r200)), in_shortlist=float(np.mean(short)), mean_shortlist=float(np.mean(nshort)) if nshort else 0.0,
                rr=rr)


cov = [T[k] is not None and bool(T[k]['is_truth'].any()) for k in panel.key]
single = (panel.n_sid == 1).values
noncoco = (~panel.in_coconut).values
variants = [
    ('v18: N1=1000 rows', dict(n1=1000, lam=0, union=0)),
    ('v4i: N1=5000 rows', dict(n1=5000, lam=0, union=0)),
    ('v19: N1=5000 + pop 0.25 + union 200', dict(n1=5000, lam=0.25, union=200)),
    ('N1=1000 + pop 0.25 + union 200', dict(n1=1000, lam=0.25, union=200)),
    ('dedup N1=1000 conn', dict(n1=1000, lam=0, union=0, dedup=True)),
    ('dedup N1=5000 conn', dict(n1=5000, lam=0, union=0, dedup=True)),
    ('dedup N1=5000 + pop 0.25 + union 200', dict(n1=5000, lam=0.25, union=200, dedup=True)),
    ('full window, no prior', dict(n1=None, lam=0, union=0)),
    ('full window + pop 0.10', dict(n1=None, lam=0.10, union=0)),
    ('full window + pop 0.25', dict(n1=None, lam=0.25, union=0)),
    ('full window + pop 0.50', dict(n1=None, lam=0.50, union=0)),
    ('full window + pop 1.00', dict(n1=None, lam=1.00, union=0)),
    ('full window + pop 2.00', dict(n1=None, lam=2.00, union=0)),
    ('full + (pop + patents) 0.25', dict(n1=None, lam=0.25, union=0, prior=lambda t: t['pop'] + t['lpat'].astype(np.float32))),
]
res = [evaluate(n, **kw) for n, kw in variants]
base_rr = np.array(next(r for r in res if r['name'].startswith('v19'))['rr'])
rng = np.random.default_rng(0)
n = len(panel)
for r in res:
    d = np.array(r['rr']) - base_rr
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
    r['delta_vs_v19'] = float(d.mean()); r['ci95'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    r['mrr25_single_sid'] = float(np.array(r['rr'])[single].mean())
    r['mrr25_non_coconut'] = float(np.array(r['rr'])[noncoco].mean()); r['delta_non_coconut'] = float(d[noncoco].mean())
    del r['rr']
# where does the truth sit? rank by partial score (rows and connectivities) and by full f.z (distinct identities)
diag = []
for k in panel.key:
    t = T[k]
    if t is None or not t['is_truth'].any():
        continue
    ok = t['ok']; sc1 = np.where(ok, t['sc1'], -np.inf); fz = np.where(ok, t['fz'], -np.inf)
    best = np.where(t['is_truth'], sc1, -np.inf).max()
    row_rank = int((sc1 > best).sum()) + 1
    conn_rank = len(set(t['ik'][sc1 > best].tolist())) + 1
    bf = np.where(t['is_truth'], fz, -np.inf).max()
    fz_rank = len(set(t['ident'][(fz > bf) & ~t['inpool']].tolist())) + 1
    diag.append(dict(key=k, n=int(t['n']), n_conn=len(set(t['ik'].tolist())), row_rank=row_rank, conn_rank=conn_rank, fz_rank=fz_rank,
                     pop_truth=float(t['pop'][t['is_truth']].max()), pop_pct=float((t['pop'] < t['pop'][t['is_truth']].max()).mean()),
                     npf_truth=int(t['npf'][t['is_truth']].max()), npf_share=float((t['npf'] > 0).mean())))
D = pd.DataFrame(diag)
summary = dict(panel=len(panel), non_coconut=int(noncoco.sum()), truth_in_window=int(np.sum(cov)), single_sid=int(single.sum()),
               window_rows_median=float(D.n.median()), window_conn_median=float(D.n_conn.median()),
               pass1_row_rank=dict(median=float(D.row_rank.median()), le1000=float((D.row_rank <= 1000).mean()), le5000=float((D.row_rank <= 5000).mean())),
               pass1_conn_rank=dict(median=float(D.conn_rank.median()), le1000=float((D.conn_rank <= 1000).mean()), le5000=float((D.conn_rank <= 5000).mean())),
               full_fz_rank=dict(median=float(D.fz_rank.median()), top1=float((D.fz_rank == 1).mean()), le5=float((D.fz_rank <= 5).mean()),
                                 le25=float((D.fz_rank <= 25).mean()), le100=float((D.fz_rank <= 100).mean())),
               truth_pop_percentile_median=float(D.pop_pct.median()), truth_np_flag=float((D.npf_truth > 0).mean()),
               window_np_flag_share_median=float(D.npf_share.median()))
json.dump(dict(summary=summary, variants=res), open(OUTJ, 'w'), indent=1)
D.to_parquet(OUTJ.replace('.json', '_diag.parquet'), index=False)
print(json.dumps(summary, indent=1))
for r in res:
    print(f"{r['name']:<42} MRR@25 {r['mrr25']:.4f}  R@25 {r['recall25']:.3f}  top1 {r['top1']:.3f}  short {r['in_shortlist']:.3f}  "
          f"d_v19 {r['delta_vs_v19']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  single-sid {r['mrr25_single_sid']:.4f}  nonCOCO {r['mrr25_non_coconut']:.4f} ({r['delta_non_coconut']:+.4f})")

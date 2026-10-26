"""Trade-off of letting the top PubChem proposal take rank 1, on the NP panel (FPNet-only gate quantities).
usage: python gate_eval.py <main_out> <tables> <pool_meta> <poolfz.parquet>"""
import os, pickle, sys
import numpy as np, pandas as pd
MAIN, TAB, POOL, PFZ = sys.argv[1:5]
S = {s: pickle.load(open(os.path.join(MAIN, s + '.pkl'), 'rb')) for s in ('lib', 'nospec', 'dbdrop')}
keys = sorted(k for k in S['nospec'] if all('error' not in S[s][k] for s in S))
pm = pd.read_parquet(POOL, columns=['key']); pool_keys = set(k.encode() for k in pm.key)
pfz = pd.read_parquet(PFZ).set_index('key')
rows = []
for k in keys:
    d = np.load(os.path.join(TAB, k + '.npz'))
    if int(d['n']) == 0:
        continue
    fz = d['fz']; ok = np.isfinite(fz); ident = np.where(d['sk'] != b'', d['sk'], d['ik'])
    inpool_all = np.isin(ident, np.array(sorted(pool_keys), 'S14'))          # truth counts as pool here (scenario A)
    pop = d['lsid'].astype(np.float32) + d['lpmid'].astype(np.float32)
    sc1 = np.where(ok, d['sc1'], -np.inf); nok = int(ok.sum())
    short = np.where(ok)[0] if 5000 >= nok else np.argpartition(-sc1, 4999)[:5000]
    short = np.union1d(short[ok[short]], np.argpartition(-np.where(ok, pop, -np.inf), min(200, nok) - 1)[:min(200, nok)])
    f = fz[short].astype(np.float64); z = (f - f.mean()) / (f.std() + 1e-9); sc = z + 0.25 * pop[short]
    def top(mask_excl):
        m = ~mask_excl[short]
        if not m.any():
            return None
        i = np.argmax(np.where(m, sc, -np.inf)); j = np.argmax(np.where(m, f, -np.inf))
        s2 = np.sort(sc[m])[::-1]
        return dict(truth=bool(d['is_truth'][short[i]]), fz_top=float(f[j]), sc=float(sc[i]), margin=float(s2[0] - s2[1]) if len(s2) > 1 else 9.0,
                    pop=float(pop[short[i]]), z=float(z[i]))
    A = top(inpool_all)                                   # truth in pool: PubChem list cannot contain it
    B = top(inpool_all & ~d['is_truth'])                   # truth only in PubChem
    rows.append(dict(key=k, relA=A['fz_top'] - pfz.best_incl[k] if A else np.nan, relB=B['fz_top'] - pfz.best_excl[k] if B else np.nan,
                     A_margin=A['margin'] if A else np.nan, B_margin=B['margin'] if B else np.nan, A_pop=A['pop'] if A else np.nan,
                     B_pop=B['pop'] if B else np.nan, A_sc=A['sc'] if A else np.nan, B_sc=B['sc'] if B else np.nan, B_truth_top=B['truth'] if B else False,
                     lib_lib=S['lib'][k]['lib_max'], lib_nospec=S['nospec'][k]['lib_max'],
                     r_lib=S['lib'][k].get('truth_rank') or 99, r_nospec=S['nospec'][k].get('truth_rank') or 99))
D = pd.DataFrame(rows); D.to_parquet(os.path.join(os.path.dirname(PFZ), 'gate_table.parquet'), index=False)
print('n', len(D), 'B truth is PubChem top', D.B_truth_top.mean())
print('rel quantiles  A(truth in pool):', np.nanquantile(D.relA, [.1, .5, .9, .99]).round(0), ' B(truth only in PubChem):', np.nanquantile(D.relB, [.1, .5, .9]).round(0))
print('top PubChem pop: A', np.nanquantile(D.A_pop, [.1, .5, .9]).round(2), ' B', np.nanquantile(D.B_pop, [.1, .5, .9]).round(2))
print('top PubChem margin: A', np.nanquantile(D.A_margin, [.1, .5, .9]).round(2), ' B', np.nanquantile(D.B_margin, [.1, .5, .9]).round(2))
print('top PubChem score: A', np.nanquantile(D.A_sc, [.1, .5, .9]).round(2), ' B', np.nanquantile(D.B_sc, [.1, .5, .9]).round(2))
def rr(r): return np.where(r <= 25, 1.0 / r, 0.0)
for name, fa, fb in [('rel>600', D.relA > 600, D.relB > 600), ('rel>300', D.relA > 300, D.relB > 300), ('rel>0', D.relA > 0, D.relB > 0),
                     ('margin>1', D.A_margin > 1, D.B_margin > 1), ('margin>2', D.A_margin > 2, D.B_margin > 2), ('margin>3', D.A_margin > 3, D.B_margin > 3),
                     ('score>6', D.A_sc > 6, D.B_sc > 6), ('score>8', D.A_sc > 8, D.B_sc > 8), ('pop>5', D.A_pop > 5, D.B_pop > 5),
                     ('margin>2 & pop>4', (D.A_margin > 2) & (D.A_pop > 4), (D.B_margin > 2) & (D.B_pop > 4))]:
    for scen, lib, r in (('lib', D.lib_lib, D.r_lib), ('nospec', D.lib_nospec, D.r_nospec)):
        fire = fa & (lib < 0.9)
        loss = (rr(r) - rr(r + 1))[fire].sum() / len(D)             # truth pushed down one place (vs no insertion above it)
        if scen == 'lib': l1 = loss; f1 = fire.mean()
        else: l2 = loss; f2 = fire.mean()
    fireB = fb & (D.lib_nospec < 0.9)
    gainB = (np.where(D.B_truth_top, 1.0, 0.0) - np.where(D.B_truth_top, 0.25, 0.0))[fireB].sum() / len(D)   # vs gentle slot 4
    print(f'{name:<18} fires: pool-truth lib {f1:.3f} nospec {f2:.3f} | PubChem-only {fireB.mean():.3f} (precision {D.B_truth_top[fireB].mean() if fireB.any() else float("nan"):.3f})'
          f' | MRR change: lib {-l1:+.4f} nospec {-l2:+.4f} PubChem-only {gainB:+.4f}')

import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace import parse, Tracer, rank_gf2, MAXX
ops = parse(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
rng = np.random.default_rng(1)
xs = np.unique(np.concatenate([np.arange(1, 4097), rng.integers(1, MAXX + 1, N), [MAXX]])).astype(np.int64)
nq = 1 + max(int(s[1:]) for o, a, c in ops for s in a if s[0] == 'q')
nb = 1 + max([int(s[1:]) for o, a, c in ops for s in a if s[0] == 'b'] + [c for o, a, c in ops if c is not None])
T = Tracer(xs, nq, nb, rng)
ones = T.full
out = []
for i, (o, a, c) in enumerate(ops):
    T.step(o, a, c)
    if i % int(sys.argv[3] if len(sys.argv) > 3 else 50) == 0 or o in ('CCX',):
        live = [j for j in range(nq) if T.q[j].any()]
        rk = rank_gf2([T.q[j] for j in live] + [ones]) - 1
        out.append((i, len(live), rk))
live_ok = all(not T.q[j].any() for j in range(22, nq))
xr = all((T.q[j] == Tracer(xs, nq, nb, rng).q[j]).all() for j in range(22))
print('end: scratch clean', live_ok, 'x restored', xr, 'nq', nq, 'nb', nb, 'inputs', len(xs))
a = np.array(out)
print('max live', a[:, 1].max(), 'max rank(nonconst)', a[:, 2].max())
for L in range(a[:, 1].max() - 3, a[:, 1].max() + 1):
    sel = a[a[:, 1] == L]
    if len(sel): print('live', L, 'points', len(sel), 'rank min/max', sel[:, 2].min(), sel[:, 2].max(), 'ops', sel[0, 0], '..', sel[-1, 0])
np.save(sys.argv[1] + '.rank.npy', a)

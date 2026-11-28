"""For each Toffoli-like op: is its effect a GF(2)-linear function of the other qubits (=> free CNOTs)?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace import parse, Tracer, MAXX

def in_span(vecs, names, v):
    """reduce v by basis built from vecs; return list of names combining to v or None"""
    basis = []  # (pivot_word, pivot_bit, vec, set_of_names)
    for nm, x in zip(names, vecs):
        x = x.copy(); s = {nm}
        for w, bt, bv, bs in basis:
            if x[w] & bt: x ^= bv; s ^= bs
        nz = np.flatnonzero(x)
        if len(nz):
            w = nz[0]; bt = x[w] & (~x[w] + np.uint64(1)); basis.append((w, bt, x, s))
    y = v.copy(); s = set()
    for w, bt, bv, bs in basis:
        if y[w] & bt: y ^= bv; s ^= bs
    return None if y.any() else s

ops = parse(sys.argv[1]); N = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
xs = np.unique(np.concatenate([np.arange(1, 4097), rng.integers(1, MAXX + 1, N), [MAXX - i for i in range(64)]])).astype(np.int64)
nq = 1 + max(int(s[1:]) for o, a, c in ops for s in a if s[0] == 'q')
nb = 1 + max([int(s[1:]) for o, a, c in ops for s in a if s[0] == 'b'] + [c for o, a, c in ops if c is not None])
T = Tracer(xs, nq, nb, rng)
found = []
for i, (o, a, c) in enumerate(ops):
    if o in ('CCX', 'CCZ', 'CSWAP'):
        m = T.mask(c)
        if m.any():
            qs = [int(s[1:]) for s in a]
            if o == 'CCX':
                v = T.q[qs[0]] & T.q[qs[1]] & m; excl = {qs[2]}
            elif o == 'CCZ':
                v = T.q[qs[0]] & T.q[qs[1]] & T.q[qs[2]] & m; excl = set()
            else:
                v = T.q[qs[0]] & (T.q[qs[1]] ^ T.q[qs[2]]) & m; excl = {qs[1], qs[2]}
            if o != 'CCZ':
                names = ['one'] + [j for j in range(nq) if j not in excl and T.q[j].any()]
                vecs = [T.full & m] + [T.q[j] & m for j in names[1:]]
                s = in_span(vecs, names, v)
                if s is not None:
                    found.append((i, o, ' '.join(a), c, sorted(s, key=str)))
            else:
                # CCZ: phase (-1)^(abc); free if abc==0, or equals a 2-control product (CZ) of a pair
                if not v.any(): found.append((i, o, ' '.join(a), c, ['zero']))
                else:
                    for p in range(3):
                        u = T.q[qs[(p+1) % 3]] & T.q[qs[(p+2) % 3]] & m
                        if (u == v).all(): found.append((i, o, ' '.join(a), c, ['pair', p])); break
    T.step(o, a, c)
print('toffoli-like ops', sum(o in ('CCX','CCZ','CSWAP') for o, a, c in ops), 'linear-replaceable', len(found))
for f in found[:80]: print(f)

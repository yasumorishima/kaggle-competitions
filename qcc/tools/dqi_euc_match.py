"""Score each S-register snapshot (dqi_fprint.py PATHOUT) against every Euclid remainder r_k of the lane (our analysis code).
usage: python dqi_euc_match.py <host dir> <fpath.pkl>"""
import sys, os, pickle, collections
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi')]
import kickmix as km, verify_submission_dqi as V
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
LOG = T.log.astype(np.int64); EXP = T.antilog.astype(np.int64); ORD = int(T.order)
def mul(a, b): return 0 if a == 0 or b == 0 else int(EXP[(LOG[a] + LOG[b]) % ORD])
def inv(a): return int(EXP[(-LOG[a]) % ORD])
def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0: p.pop()
    return p
def divmod_(a, b):
    a = list(a); q = [0] * max(1, len(a) - len(b) + 1); ib = inv(b[-1])
    for i in range(len(a) - len(b), -1, -1):
        c = mul(a[i + len(b) - 1], ib)
        if c:
            q[i] = c
            for j, x in enumerate(b): a[i + j] ^= mul(c, x)
    return trim(q), trim(a[:len(b) - 1] or [0])
def pmul(a, b):
    o = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b): o[i + j] ^= mul(x, y)
    return trim(o)
def padd(a, b):
    n = max(len(a), len(b)); return trim([(a[i] if i < len(a) else 0) ^ (b[i] if i < len(b) else 0) for i in range(n)])
def euclid(S):
    r0, r1 = [0] * 70 + [1], trim(S); t0, t1 = [0], [1]; seq = [('r', 0, r1), ('t', 0, t1)]; k = 0
    while len(r1) - 1 >= 35:
        q, r = divmod_(r0, r1); r0, r1 = r1, r; t0, t1 = t1, padd(t0, pmul(q, t1)); k += 1
        seq += [('r', k, r1), ('t', k, t1)]
    return seq
def score(words, P):
    ws = [int(w) for w in words if w]; cnt = collections.Counter()
    for p in P:
        if p:
            for w in set(ws): cnt[(LOG[w] - LOG[p]) % ORD] += 1
    return max(cnt.values()) if cnt else 0
d = pickle.load(open(sys.argv[2], 'rb')); lanes, path = d['lanes'], d['path']
seqs = [euclid([int(x) for x in sy]) for _, sy, _ in lanes]
print('euclid steps per lane', [len(s) // 2 - 1 for s in seqs])
prev = None
for t, S in path[::3]:
    row = []
    for l in range(len(lanes)):
        best = max(((score(S[l], P), f'{kind}{k}') for kind, k, P in seqs[l]), key=lambda x: x[0])
        row.append(f'{best[1]}:{best[0]}')
    line = ' '.join(f'{r:>7}' for r in row)
    if line != prev: print(f'{t:>9} {line}'); prev = line

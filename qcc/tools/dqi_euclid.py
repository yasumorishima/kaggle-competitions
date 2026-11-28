"""Remainder-degree sequence of the extended Euclid on x^n and S(x) for DQI shots (our analysis code).
usage: python dqi_euclid.py <host dir> <fails.jsonl> [n random passing shots]"""
import json, os, sys
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi'), os.path.dirname(os.path.abspath(__file__))]
import kickmix as km
import verify_submission_dqi as V
import dqi_fails as F
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
LOG, EXP, ORD = T.log, T.antilog, T.order
def mul(a, b): return 0 if a == 0 or b == 0 else int(EXP[(int(LOG[a]) + int(LOG[b])) % ORD])
def inv(a): return int(EXP[(-int(LOG[a])) % ORD])
def deg(p):
    for i in range(len(p) - 1, -1, -1):
        if p[i]: return i
    return -1
def euclid(S, n):
    a = [0] * n + [1]; b = list(S) + [0]
    degs = []; quots = []
    l = n // 2
    while deg(b) >= l:
        da, db = deg(a), deg(b); qd = da - db; quots.append(qd)
        r = a[:]
        while deg(r) >= db:
            d = deg(r); f = mul(r[d], inv(b[db]))
            for i in range(db + 1):
                if b[i]: r[i + d - db] ^= mul(f, b[i])
        a, b = b, r; degs.append(deg(b))
    return degs, quots
def run(syn):
    degs, q = euclid([int(v) for v in syn], spec.n)
    return len(q), max(q) if q else 0, sum(x > 1 for x in q)
wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
for line in open(sys.argv[2]):
    d = json.loads(line)
    syn, mbuc, ph, errs = F.sample(spec, T, np.random.default_rng(d['seed']), 256, wp)
    print('FAIL w', d['weight'], 'steps, max quotient degree, #quotients deg>1:', run(syn[d['shot']]))
syn, mbuc, ph, errs = F.sample(spec, T, np.random.default_rng(7), int(sys.argv[3]) if len(sys.argv) > 3 else 256, wp)
from collections import Counter
cnt = Counter(run(s)[1:] for s in syn)
print('random shots: (max quotient degree, #deg>1) ->', sorted(cnt.items()))

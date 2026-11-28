"""Berlekamp-Massey discrepancy pattern of DQI shots: zero discrepancies and runs of them (our analysis code).
usage: python dqi_bm_disc.py <host dir> <fails.jsonl> [n random]"""
import json, os, sys
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi'), os.path.dirname(os.path.abspath(__file__))]
import kickmix as km
import verify_submission_dqi as V
import dqi_fails as F
from dqi_chien import mul, inv
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
def disc(S):
    C = [1]; B = [1]; L = 0; m = 1; b = 1; ds = []
    for n in range(len(S)):
        d = S[n]
        for i in range(1, L + 1):
            if i < len(C): d ^= mul(C[i], S[n - i])
        ds.append((d, L))
        if d == 0: m += 1; continue
        coef = mul(d, inv(b)); Tt = C[:]
        C = C + [0] * (len(B) + m - len(C))
        for i, x in enumerate(B): C[i + m] ^= mul(coef, x)
        if 2 * L <= n: L = n + 1 - L; B = Tt; b = d; m = 1
        else: m += 1
    return ds
def summary(S):
    ds = disc([int(x) for x in S]); Lf = ds[-1][1]
    zeros = [n for n, (d, L) in enumerate(ds) if d == 0 and n < 2 * 35]
    runs = []; r = 0
    for n, (d, L) in enumerate(ds):
        if d == 0: r += 1
        else:
            if r: runs.append((n - r, r))
            r = 0
    if r: runs.append((len(ds) - r, r))
    return zeros, runs
wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
for line in open(sys.argv[2]):
    d = json.loads(line)
    syn, _, _, _ = F.sample(spec, T, np.random.default_rng(d['seed']), 256, wp)
    print('FAIL w', d['weight'], 'zero-discrepancy steps / runs (start, length):', summary(syn[d['shot']]))
syn, _, _, errs = F.sample(spec, T, np.random.default_rng(9), int(sys.argv[3]) if len(sys.argv) > 3 else 30, wp)
from collections import Counter
c = Counter()
for s in range(len(syn)):
    if errs[s]:
        z, runs = summary(syn[s]); w = len(errs[s][0])
        c[(max([l for st, l in runs if st < 2 * w] + [0]))] += 1
print('random shots: longest zero run before step 2w ->', sorted(c.items()))

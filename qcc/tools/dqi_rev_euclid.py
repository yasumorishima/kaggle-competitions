"""Quotient degrees of the extended Euclid on x^n and S(x), for S in natural and in reversed coefficient order,
for failing shots and random passing shots (our analysis code). A continued-fraction (Schur) decoder that walks the
syndromes from the other end meets the reversed sequence's quotients; a degree >= 2 there is a gap it may not handle.
usage: python dqi_rev_euclid.py <host dir> <fails.jsonl> [n passing]"""
import json, os, sys
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi'), os.path.dirname(os.path.abspath(__file__))]
import kickmix as km
import verify_submission_dqi as V
import dqi_fails as F
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
LOG = T.log.astype(np.int64); EXP = T.antilog.astype(np.int64); ORD = int(T.order)
def mul(a, b): return 0 if a == 0 or b == 0 else int(EXP[(LOG[a] + LOG[b]) % ORD])
def inv(a): return int(EXP[(-LOG[a]) % ORD])
def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0: p.pop()
    return p
def qdegs(S, n=70, stop=35):
    a, b = [0] * n + [1], trim(S); out = []
    while len(b) - 1 >= stop and any(b):
        a = list(a); ib = inv(b[-1]); qd = len(a) - len(b)
        for i in range(qd, -1, -1):
            c = mul(a[i + len(b) - 1], ib)
            if c:
                for j, x in enumerate(b): a[i + j] ^= mul(c, x)
        a, b = b, trim(a[:len(b) - 1] or [0]); out.append(qd)
    return out
def summ(S):
    nat, rev = qdegs(S), qdegs(S[::-1])
    return max(nat), max(rev), [i for i, d in enumerate(rev) if d > 1], S[0] == 0, S[-1] == 0
wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
for line in open(sys.argv[2]):
    d = json.loads(line)
    syn, _, _, _ = F.sample(spec, T, np.random.default_rng(d['seed']), 256, wp)
    print('FAIL w', d['weight'], 'max q-deg natural/reversed, reversed steps with deg>1, S1==0, Sn==0:', summ([int(x) for x in syn[d['shot']]]))
n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
syn, _, _, errs = F.sample(spec, T, np.random.default_rng(5), n, wp)
cnt = {'nat>1': 0, 'rev>1': 0, 'both': 0, 'n': 0}
for s in range(n):
    if not errs[s] or len(errs[s][0]) < 20: continue
    a, b, _, _, _ = summ([int(x) for x in syn[s]]); cnt['n'] += 1
    cnt['nat>1'] += a > 1; cnt['rev>1'] += b > 1; cnt['both'] += a > 1 and b > 1
print('passing shots with weight >= 20:', cnt)

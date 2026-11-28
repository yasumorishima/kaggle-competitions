"""Berlekamp-Massey locator of DQI error words and its values at every position (our analysis code).
For failing and random passing words: the smallest popcount of Lambda(alpha^-p) over non-error positions,
and which bits are zero there.  usage: python dqi_chien.py <host dir> <fails.jsonl> [n random]"""
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
def bm(S):
    C = [1]; B = [1]; L = 0; m = 1; b = 1
    for n in range(len(S)):
        d = S[n]
        for i in range(1, L + 1):
            if i < len(C): d ^= mul(C[i], S[n - i])
        if d == 0: m += 1; continue
        coef = mul(d, inv(b)); Tt = C[:]
        C = C + [0] * (len(B) + m - len(C))
        for i, x in enumerate(B): C[i + m] ^= mul(coef, x)
        if 2 * L <= n: L = n + 1 - L; B = Tt; b = d; m = 1
        else: m += 1
    return C[:L + 1]
def values(C):
    p = np.arange(spec.m, dtype=np.int64); acc = np.zeros(spec.m, np.int64)
    for i, ci in enumerate(C):
        if ci: acc ^= EXP[(LOG[ci] - i * p) % ORD]
    return acc
def popc(v): return np.array([bin(int(x)).count('1') for x in v])
def report(tag, syn, errpos):
    C = bm([int(x) for x in syn]); v = values(C); roots = set(np.flatnonzero(v == 0).tolist())
    non = np.setdiff1d(np.arange(spec.m), list(errpos)); pc = popc(v[non])
    j = non[np.argmin(pc)]
    print(f'{tag} deg {len(C)-1} roots_ok {roots == set(errpos)} min popcount {pc.min()} at {j} value {int(v[j]):012b} count(pc<=2) {(pc<=2).sum()}')
wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
for line in open(sys.argv[2]):
    d = json.loads(line)
    syn, mbuc, ph, errs = F.sample(spec, T, np.random.default_rng(d['seed']), 256, wp)
    report('FAIL', syn[d['shot']], d['pos'])
syn, mbuc, ph, errs = F.sample(spec, T, np.random.default_rng(9), int(sys.argv[3]) if len(sys.argv) > 3 else 20, wp)
for s in range(len(syn)):
    if errs[s]: report('pass', syn[s], errs[s][0])

"""Read the S-register path saved by dqi_fprint.py (PATHOUT) and test what the 70 words hold at each snapshot
(our analysis code). For every lane the classical locator Lambda and evaluator Omega are computed from its input;
at each snapshot we count, for each of them, the largest number of its coefficients that sit in the S words
times one common nonzero scalar (a register that holds the polynomial up to scale scores about its length).
Passing lanes show which snapshot holds which polynomial; a failing lane that stops matching where passing lanes
still match marks the first stage that differs.
usage: python dqi_fpath.py <host dir> <fpath.pkl>"""
import os, sys, pickle, collections
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi'), os.path.dirname(os.path.abspath(__file__))]
import kickmix as km
import verify_submission_dqi as V
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
def omega(S, C):
    Om = [0] * len(S)
    for i, ci in enumerate(C):
        for j in range(len(S) - i): Om[i + j] ^= mul(ci, S[j])
    while len(Om) > 1 and Om[-1] == 0: Om.pop()
    return Om
def score(words, P):
    """max over scalars c of #{i : c*P[i] is among the words}, with the share of words that are zero"""
    ws = set(int(w) for w in words if w)
    cnt = collections.Counter()
    for p in P:
        if not p: continue
        lp = LOG[p]
        for w in ws: cnt[(LOG[w] - lp) % ORD] += 1
    return max(cnt.values()) if cnt else 0
d = pickle.load(open(sys.argv[2], 'rb')); lanes, path = d['lanes'], d['path']
polys = []
for tag, sy, ev in lanes:
    S = [int(x) for x in sy]; C = bm(S); polys.append((C, omega(S, C)))
print('lanes:', [(t, len(p[0]) - 1) for (t, _, _), p in zip(lanes, polys)])
print('op        ' + ' '.join(f'{t[:6]:>13}' for t, _, _ in lanes))
prev = None
for t, Sall in path:
    row = []
    for l in range(len(lanes)):
        w = Sall[l]; C, Om = polys[l]
        row.append(f'{score(w, C):>2}/{len(C):<2} {score(w, Om):>2}/{len(Om):<2} z{int((w == 0).sum()):<2}'[:13])
    line = ' '.join(f'{r:>13}' for r in row)
    if line != prev: print(f'{t:>9} {line}'); prev = line

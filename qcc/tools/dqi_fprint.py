"""Fingerprint DQI failures by the syndrome register's path through the circuit (our analysis code).

Runs a kmx text classically over a few input lanes (failing shots + passing shots), as kmx_lanes.py does, and every
SHK ops reads the S registers. The difference D_t = S_t xor S_0 is again a syndrome (of the errors the circuit has
added so far), so a classical Berlekamp-Massey + Chien + Forney on D_t tells which (position, value) pairs the
circuit has applied by time t. The first time a failing lane holds a pair that is not one of its true errors (or
D_t stops decoding) is the first divergence: the fingerprint is (op index, the wrong pair, how many true pairs
were right by then).
usage: SHK=500000 python dqi_fprint.py <host dir> <circuit.kmx> <fails.jsonl> [n passing lanes] > out.txt
"""
import json, os, sys, random
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
def bm(S):                                  # the same Berlekamp-Massey as dqi_chien.py (that file runs on import)
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
wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2

def peval(P, x):
    acc = 0; xp = 1
    for c in P:
        if c: acc ^= mul(c, xp)
        xp = mul(xp, x)
    return acc

def decode(S):
    """{pos: val} for a syndrome S_j = sum v X^j (j = 1..n), or None when it does not decode."""
    S = [int(x) for x in S]
    if not any(S): return {}
    C = bm(S); L = len(C) - 1
    if 2 * L > len(S): return None
    p = np.arange(spec.m, dtype=np.int64); acc = np.zeros(spec.m, np.int64)
    for i, ci in enumerate(C):
        if ci: acc ^= EXP[(LOG[ci] - i * p) % ORD]
    roots = np.flatnonzero(acc == 0).tolist()
    if len(roots) != L: return None
    Om = [0] * len(S)                       # Omega = S(x) * Lambda(x) mod x^n, S(x) = sum S_{j+1} x^j
    for i, ci in enumerate(C):
        for j in range(len(S) - i): Om[i + j] ^= mul(ci, S[j])
    dC = [C[i] if i % 2 == 1 else 0 for i in range(1, len(C))]   # formal derivative
    out = {}
    for r in roots:
        xi = int(EXP[(-r) % ORD])           # X^-1
        num, den = peval(Om, xi), peval(dC, xi)
        if den == 0: return None
        out[r] = mul(num, inv(den))
    return out

lanes = []                                  # (tag, syndrome, {pos: val})
if sys.argv[3].endswith('.json'):           # a list of explicit inputs (syndrome, positions, values), e.g. a public scan's output
    for d in json.load(open(sys.argv[3])):
        lanes.append((f"EXT{len(lanes)} w{d['weight']} {d['kind']}", np.array(d['syndrome'], np.int64), dict(zip(d['positions'], d['values']))))
for line in (open(sys.argv[3]) if not sys.argv[3].endswith('.json') else []):
    d = json.loads(line)
    syn, _, _, _ = F.sample(spec, T, np.random.default_rng(d['seed']), 256, wp)
    lanes.append((f"FAIL{len(lanes)} w{d['weight']}", syn[d['shot']], dict(zip(d['pos'], d['vals']))))
npass = int(sys.argv[4]) if len(sys.argv) > 4 else 3
syn, _, _, errs = F.sample(spec, T, np.random.default_rng(9), 64, wp)
for s in range(64):
    if errs[s] and len(errs[s][0]) >= 20 and npass > 0:
        lanes.append((f"pass w{len(errs[s][0])}", syn[s], dict(zip(map(int, errs[s][0]), map(int, errs[s][1]))))); npass -= 1
for tag, sy, ev in lanes:                   # self-test of the classical decoder on the inputs
    got = decode(sy)
    assert got == ev, (tag, 'decoder self-test failed', None if got is None else len(got), len(ev))
print('decoder self-test ok on', len(lanes), 'lanes', flush=True)

L = len(lanes); ALL = (1 << L) - 1
regs, cur, q, b = {}, {}, {}, {}
rnd = random.Random(1); stack = [ALL]; n = 0; inited = False
SHK = int(os.environ.get('SHK', '500000')); Sq = None; path = []
def Q(t): return q.get(t, 0)
def read_S():
    out = np.zeros((L, spec.n), np.int64)
    for i, qs in enumerate(Sq):
        for k, qq in enumerate(qs):
            v = Q(qq)
            for l in range(L):
                if (v >> l) & 1: out[l, i] |= 1 << k
    return out
with open(sys.argv[2]) as f:
    for ln in f:
        t = ln.split(); op = t[0]; n += 1
        if op == 'APPEND_TO_REGISTER':
            regs.setdefault(t[2], []).append(t[1]); continue
        if op == 'REGISTER':
            cur[t[1]] = t[2].strip('"'); continue
        if not inited:
            inited = True
            byname = {name: rid for rid, name in cur.items()}
            Sq = [regs[byname[f'S{i}']] for i in range(spec.n)]
            for i, qs in enumerate(Sq):
                for k, qq in enumerate(qs):
                    q[qq] = sum(((int(lanes[l][1][i]) >> k) & 1) << l for l in range(L))
            S0 = read_S(); path.append((n, S0))
        if n % SHK == 0: path.append((n, read_S()))
        m = stack[-1]
        if len(t) > 2 and t[-2] == 'if':
            m &= b.get(t[-1], 0); t = t[:-2]
        if op == 'CX':
            if m: q[t[2]] = Q(t[2]) ^ (Q(t[1]) & m)
        elif op == 'BIT_INVERT': b[t[1]] = b.get(t[1], 0) ^ m
        elif op == 'BIT_STORE0': b[t[1]] = b.get(t[1], 0) & ~m
        elif op == 'BIT_STORE1': b[t[1]] = b.get(t[1], 0) | m
        elif op in ('CZ', 'Z', 'CCZ', 'NEG'): pass
        elif op == 'PUSH_CONDITION': stack.append(m)
        elif op == 'POP_CONDITION': stack.pop()
        elif op == 'SWAP':
            u, v = t[1], t[2]; x = (Q(u) ^ Q(v)) & m; q[u] = Q(u) ^ x; q[v] = Q(v) ^ x
        elif op == 'CCX': q[t[3]] = Q(t[3]) ^ (Q(t[1]) & Q(t[2]) & m)
        elif op == 'X': q[t[1]] = Q(t[1]) ^ m
        elif op == 'HMR':
            r = rnd.getrandbits(L); b[t[2]] = (b.get(t[2], 0) & ~m) | (r & m); q[t[1]] = Q(t[1]) & ~m
        elif op == 'R': q[t[1]] = Q(t[1]) & ~m
        else: raise ValueError(op)
path.append((n, read_S()))
print('ops', n, 'snapshots', len(path), flush=True)
if os.environ.get('PATHOUT'):
    import pickle; pickle.dump({'lanes': lanes, 'path': path}, open(os.environ['PATHOUT'], 'wb'))
for l, (tag, sy, ev) in enumerate(lanes):
    print(f'{tag} END syndrome words wrong {int((path[-1][1][l] != path[0][1][l]).sum())} of {spec.n}')
for l, (tag, sy, ev) in enumerate(lanes):
    S0 = path[0][1][l]; first = None; prev = None
    for t, S in path:
        D = S[l] ^ S0; got = decode(D)
        if got is None: state = 'undecodable'
        else:
            wrong = {p: v for p, v in got.items() if ev.get(p) != v}
            state = f'{len(got) - len(wrong)} right {len(wrong)} wrong' + (f' {sorted(wrong.items())[:4]}' if wrong else '')
            if wrong and first is None: first = (t, sorted(wrong.items())[:4], len(got) - len(wrong))
        if state != prev: print(f'{tag} op {t:>9}: {state}'); prev = state
        if got is None and first is None: first = (t, 'undecodable', None)
    print(f'{tag} FINGERPRINT first divergence {first}  (true errors {len(ev)})', flush=True)

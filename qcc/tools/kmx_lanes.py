"""Lean classical simulation of a kmx circuit text over a few input lanes (python ints as lane masks).
Records every R that hits a qubit holding 1 (information loss), and the end state per lane.
usage: python kmx_lanes.py <host dir> <circuit.kmx> <fails.jsonl> [extra passing lanes]   (DQI register layout)"""
import json, os, sys, random
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi'), os.path.dirname(os.path.abspath(__file__))]
import kickmix as km
import verify_submission_dqi as V
import dqi_fails as F
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
syns = []; tags = []
for line in open(sys.argv[3]):
    d = json.loads(line)
    syn, _, _, _ = F.sample(spec, T, np.random.default_rng(d['seed']), 256, wp)
    syns.append(syn[d['shot']]); tags.append(f"FAIL w{d['weight']}")
syn, _, _, errs = F.sample(spec, T, np.random.default_rng(9), 64, wp)
extra = int(sys.argv[4]) if len(sys.argv) > 4 else 3
for s in range(64):
    if errs[s] and len(syns) < len(tags) + extra and len(tags) < 64:
        syns.append(syn[s]); tags.append(f"pass w{len(errs[s][0])}")
L = len(syns); ALL = (1 << L) - 1
regs = {}; cur = {}
q = {}; b = {}
rnd = random.Random(1)
stack = [ALL]; rhits = {}; n = 0; inited = False
HK = int(os.environ.get('HK', '0')); NQ = 900; hist = []
SNAP = set(int(x) for x in os.environ.get('SNAP', '').split(',') if x); snaps = {}
def Q(t): return q.get(t, 0)
with open(sys.argv[2]) as f:
    for ln in f:
        t = ln.split(); op = t[0]; n += 1
        if op == 'APPEND_TO_REGISTER':
            regs.setdefault(t[2], []).append(t[1]); continue
        if op == 'REGISTER':
            cur[t[1]] = t[2].strip('"')
            if cur[t[1]].startswith('S'):
                pass
            continue
        if not inited:
            inited = True
            for rid, name in cur.items():
                if name.startswith('S'):
                    i = int(name[1:])
                    for k, qq in enumerate(regs[rid]):
                        q[qq] = sum(((int(syns[l][i]) >> k) & 1) << l for l in range(L))
        if n % 5000000 == 0: print('ops', n, flush=True)
        if n in SNAP: snaps[n] = dict(q)
        if HK and n % HK == 0:
            arr = np.array([q.get(f'q{i}', 0) for i in range(NQ)], dtype=np.uint64)
            bits = np.unpackbits(arr.view(np.uint8).reshape(NQ, 8), axis=1, bitorder='little')[:, :L]
            hist.append((n, [hash(bits[:, l].tobytes()) for l in range(L)]))
        m = stack[-1]
        if len(t) > 2 and t[-2] == 'if':
            m &= b.get(t[-1], 0); t = t[:-2]
        if op == 'CX':
            if m: q[t[2]] = Q(t[2]) ^ (Q(t[1]) & m)
        elif op == 'BIT_INVERT':
            b[t[1]] = b.get(t[1], 0) ^ m
        elif op == 'BIT_STORE0':
            b[t[1]] = b.get(t[1], 0) & ~m
        elif op == 'BIT_STORE1':
            b[t[1]] = b.get(t[1], 0) | m
        elif op in ('CZ', 'Z', 'CCZ', 'NEG'):
            pass
        elif op == 'PUSH_CONDITION':
            stack.append(m)
        elif op == 'POP_CONDITION':
            stack.pop()
        elif op == 'SWAP':
            u, v = t[1], t[2]; x = (Q(u) ^ Q(v)) & m; q[u] = Q(u) ^ x; q[v] = Q(v) ^ x
        elif op == 'CCX':
            q[t[3]] = Q(t[3]) ^ (Q(t[1]) & Q(t[2]) & m)
        elif op == 'X':
            q[t[1]] = Q(t[1]) ^ m
        elif op == 'HMR':
            r = rnd.getrandbits(L); b[t[2]] = (b.get(t[2], 0) & ~m) | (r & m); q[t[1]] = Q(t[1]) & ~m
        elif op == 'R':
            hit = Q(t[1]) & m
            if hit: rhits.setdefault(hit, []).append((n, t[1]))
            q[t[1]] = Q(t[1]) & ~m
        else:
            raise ValueError(op)
print('lanes', tags)
ks = sorted(snaps)
for a, c in zip(ks, ks[1:]):
    qs = set(snaps[a]) | set(snaps[c])
    for l in range(min(L, 6)):
        dif = sum(1 for qq in qs if ((snaps[a].get(qq, 0) ^ snaps[c].get(qq, 0)) >> l) & 1)
        print(f'{tags[l]}: qubits differing between op {a} and op {c}: {dif}')
import pickle; pickle.dump({'tags': tags, 'snaps': snaps, 'hist': hist}, open(os.environ.get('SNAPOUT', '/dev/null'), 'wb'))
sq = {qq for rid, name in cur.items() if name.startswith('S') for qq in regs[rid]}
for l in range(L):
    badS = sum(1 for rid, name in cur.items() if name.startswith('S') for k, qq in enumerate(regs[rid]) if ((Q(qq) >> l) & 1) != ((int(syns[l][int(name[1:])]) >> k) & 1))
    badA = sum(1 for qq, v in q.items() if qq not in sq and (v >> l) & 1)
    print(tags[l], 'syndrome bits wrong', badS, 'ancilla left 1', badA)
for hit, lst in rhits.items():
    print('R on a 1 in lanes', [tags[i] for i in range(L) if hit >> i & 1], 'times', len(lst), 'first', lst[:5])

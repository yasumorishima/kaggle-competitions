"""Differential test: the original field block (pa001) vs its D16 replacement (entry, 84-term product, between,
70-term product, exit), simulated classically on random lanes. Reads the generator's data files only."""
import json, lzma, random, sys
from pathlib import Path
D = Path(sys.argv[1]); field = int(sys.argv[2]); ZERO = sys.argv[3] if len(sys.argv) > 3 else 'none'
L = 64; ALL = (1 << L) - 1
def kmx(path): return lzma.open(path, 'rt').read().split('\n')
def masks(path):
    for line in open(path):
        if line.strip() and not line.startswith('#'): yield tuple(int(x, 16) for x in line.split())
def support(mask, wires): return {w for i, w in enumerate(wires) if mask >> i & 1}
def product_ops(wpath):
    w = json.loads(Path(wpath).read_text()); ins, outs = w['input_wires'], w['output_wires']
    ops = []
    for (l, r), (o,) in zip(masks(D / 'd16' / w['input_forms']), masks(D / 'd16' / w['output_forms']), strict=True):
        lhs, rhs, out = support(l, ins), support(r, ins), support(o, outs)
        a = min(lhs)
        if a in rhs: rhs = rhs ^ (lhs - {a})
        b = min(rhs - {a}); t = min(out)
        basis = [(s, a) for s in sorted(lhs - {a})] + [(s, b) for s in sorted(rhs - {b})] + [(t, d) for d in sorted(out - {t})]
        setup = [f'CX q{x} q{y}' for x, y in basis]
        ops += setup + [f'CCX q{a} q{b} q{t}'] + setup[::-1]
    return ops
orig = kmx(D / 'decoder' / f'pa001_field_{field}.kmx.xz')
new = (kmx(D / 'd16' / f'field_{field}_entry.kmx.xz') + product_ops(D / 'd16' / f'field_{field}_wiring_84.json')
       + kmx(D / 'd16' / f'field_{field}_between_products.kmx.xz') + product_ops(D / 'd16' / f'field_{field}_wiring_70.json')
       + kmx(D / 'd16' / f'field_{field}_exit.kmx.xz'))
def run(lines, q):
    q = dict(q); stack = [ALL]; b = {}; rnd = random.Random(7)
    for ln in lines:
        t = ln.split()
        if not t: continue
        op = t[0]; m = stack[-1]
        if len(t) > 2 and t[-2] == 'if': m &= b.get(t[-1], 0); t = t[:-2]
        g = lambda k: q.get(k, 0)
        if op == 'CX': q[t[2]] = g(t[2]) ^ (g(t[1]) & m)
        elif op == 'CCX': q[t[3]] = g(t[3]) ^ (g(t[1]) & g(t[2]) & m)
        elif op == 'X': q[t[1]] = g(t[1]) ^ m
        elif op == 'SWAP': u, v = t[1], t[2]; x = (g(u) ^ g(v)) & m; q[u] = g(u) ^ x; q[v] = g(v) ^ x
        elif op == 'HMR': b[t[2]] = rnd.getrandbits(L); q[t[1]] = g(t[1]) & ~m
        elif op == 'R': q[t[1]] = g(t[1]) & ~m
        elif op in ('CZ', 'Z', 'CCZ', 'NEG'): pass
        elif op == 'PUSH_CONDITION': stack.append(m)
        elif op == 'POP_CONDITION': stack.pop()
        elif op.startswith('BIT_'): pass
        else: raise ValueError(op)
    return q
def touched(lines):
    s = set()
    for ln in lines:
        for x in ln.split()[1:]:
            if x.startswith('q'): s.add(x)
    return s
qs = sorted(touched(orig) | touched(new), key=lambda x: int(x[1:]))
ins70 = {f'q{w}' for w in json.loads((D / 'd16' / f'field_{field}_wiring_70.json').read_text())['input_wires']}
print('field', field, 'ops orig', len(orig), 'new', len(new), 'qubits touched', len(qs), 'only in orig', len(touched(orig) - touched(new)), 'only in new', len(touched(new) - touched(orig)))
# qubits that the original resets (R) or measures (HMR) are ancilla-like; zero them in mode "anc"
anc = {ln.split()[1] for ln in orig if ln.startswith(('R ', 'HMR '))}
rng = random.Random(field); bad = 0; diffq = {}
for rep in range(int(sys.argv[4]) if len(sys.argv) > 4 else 40):
    q0 = {x: (0 if (ZERO == 'anc' and x in anc) else rng.getrandbits(L)) for x in qs}
    a, c = run(orig, q0), run(new, q0)
    for x in qs:
        d = a.get(x, 0) ^ c.get(x, 0)
        if d: diffq[x] = diffq.get(x, 0) | 0; bad += bin(d).count('1'); diffq[x] = diffq.get(x, 0) + bin(d).count('1')
print('mode', ZERO, 'lanes', 64 * (rep + 1), 'differing qubit-lane bits', bad, 'qubits', len(diffq), sorted(diffq.items(), key=lambda kv: -kv[1])[:12])

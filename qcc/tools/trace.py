"""Classical value trace of a kickmix circuit text (our own code).
Values are bit-packed over a set of inputs x (one bit per input); HMR outcomes random.
Usage: trace.py circuit.txt  -> per-op GF(2) rank stats of the live qubit values."""
import sys, re, numpy as np
MAXX = 3133452

def parse(path):
    ops = []
    for ln in open(path):
        t = ln.split()
        if not t or t[0] in ('APPEND_TO_REGISTER', 'REGISTER'): continue
        cond = None
        if 'if' in t:
            i = t.index('if'); cond = int(t[i+1][1:]); t = t[:i]
        ops.append((t[0], [a for a in t[1:]], cond))
    return ops

class Tracer:
    def __init__(self, xs, nq, nb, rng):
        self.n = len(xs); W = (self.n + 63) // 64
        self.W = W; self.rng = rng
        self.q = np.zeros((nq, W), np.uint64); self.b = np.zeros((nb, W), np.uint64)
        for i in range(22):
            bits = ((xs >> i) & 1).astype(np.uint8)
            self.q[i] = np.packbits(np.pad(bits, (0, W*64 - self.n)), bitorder='little').view(np.uint64)
        self.full = np.full(W, ~np.uint64(0)); self.stack = [self.full]
        if self.n % 64: self.full[-1] = np.uint64((1 << (self.n % 64)) - 1)
    def mask(self, cond):
        m = self.stack[-1]
        return m if cond is None else (m & self.b[cond])
    def step(self, op, a, cond):
        q = self.q
        if op == 'CX':
            c, t = int(a[0][1:]), int(a[1][1:]); q[t] ^= q[c] & self.mask(cond)
        elif op == 'X':
            t = int(a[0][1:]); q[t] ^= self.mask(cond)
        elif op == 'CCX':
            c1, c2, t = (int(s[1:]) for s in a); q[t] ^= q[c1] & q[c2] & self.mask(cond)
        elif op == 'SWAP':
            u, v = int(a[0][1:]), int(a[1][1:]); m = self.mask(cond); d = (q[u] ^ q[v]) & m; q[u] ^= d; q[v] ^= d
        elif op == 'CSWAP':
            c, u, v = (int(s[1:]) for s in a); m = self.mask(cond) & q[c]; d = (q[u] ^ q[v]) & m; q[u] ^= d; q[v] ^= d
        elif op == 'HMR':
            t, bb = int(a[0][1:]), int(a[1][1:]); m = self.mask(cond)
            r = self.rng.integers(0, 2**63, self.W, dtype=np.uint64) ^ (self.rng.integers(0, 2, self.W, dtype=np.uint64) << np.uint64(63))
            self.b[bb] = (self.b[bb] & ~m) | (r & m); q[t] &= ~m
        elif op == 'R':
            t = int(a[0][1:]); q[t] &= ~self.mask(cond)
        elif op == 'BIT_INVERT':
            bb = int(a[0][1:]); self.b[bb] ^= self.mask(cond)
        elif op == 'PUSH_CONDITION':
            self.stack.append(self.mask(cond))
        elif op == 'POP_CONDITION':
            self.stack.pop()
        elif op in ('CZ', 'Z', 'CCZ', 'Z_POW', 'NEG'):
            pass
        else:
            raise ValueError(op)

def rank_gf2(vecs):
    """rank of a list of packed uint64 vectors."""
    rows = [v.copy() for v in vecs if v.any()]
    r = 0
    while rows:
        p = rows.pop()
        nz = np.flatnonzero(p)
        if not len(nz): continue
        w = nz[0]; bit = p[w] & (~p[w] + np.uint64(1))
        r += 1
        rows = [x ^ p if (x[w] & bit) else x for x in rows]
        rows = [x for x in rows if x.any()]
    return r

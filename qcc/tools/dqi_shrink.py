"""Shrink a failing DQI error word: drop errors one at a time while the circuit still fails (our analysis code).
usage: python dqi_shrink.py <host dir> <dqi.kmb> <fails.jsonl> <line index>"""
import json, os, sys
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi')]
import kickmix as km
import verify_submission_dqi as V
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
c = km.Circuit.from_file(sys.argv[2], 'kmb'); s_q, m_b, anc = V.validate_registers(c, spec)
rows = np.arange(1, spec.n + 1, dtype=np.int64)[:, None]
def run(words):
    syn = np.zeros((256, spec.n), np.int64)
    for s, (p, v) in enumerate(words):
        if len(p):
            p = np.array(p, np.int64); v = np.array(v, np.int64)
            syn[s] = np.bitwise_xor.reduce(T.antilog[(T.log[v][None, :] + rows * p[None, :]) % T.order], axis=1)
    sim = km.Simulator(batch_size=256, debug_print=False, seed=1); sim.use_same_registers_as(c)
    inp = V.to_bits(syn, spec.degree); sim.write_across_shots(km.array(s_q), inp)
    sim.write_across_shots(km.array(m_b), V.to_bits(np.zeros((256, spec.m), np.int64), spec.degree))
    sim.do(c)
    return (sim.read_across_shots(km.array(s_q)) ^ inp).any(axis=0)[:len(words)]
d = [json.loads(l) for l in open(sys.argv[3])][int(sys.argv[4])]
P, Vv = list(d['pos']), list(d['vals'])
while True:
    words = [(P, Vv)] + [(P[:i] + P[i+1:], Vv[:i] + Vv[i+1:]) for i in range(len(P))]
    f = run(words)
    print('size', len(P), 'fails', bool(f[0]), 'still failing after dropping:', [P[i] for i in range(len(P)) if f[i + 1]], flush=True)
    keep = [i for i in range(len(P)) if f[i + 1]]
    if not f[0] or not keep: break
    i = keep[0]; P, Vv = P[:i] + P[i+1:], Vv[:i] + Vv[i+1:]
print('minimal failing word:', sorted(zip(P, Vv)))

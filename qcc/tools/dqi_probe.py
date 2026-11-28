"""Run a DQI circuit on crafted error words: which forced position makes it fail?
usage: python dqi_probe.py <host dir> <dqi.kmb> <weight> <forced positions comma list or 'none'> [batches]"""
import os, sys
import numpy as np
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi')]
import kickmix as km
import verify_submission_dqi as V
spec = V.ProblemSpecDQI.benchmark_instance(); T = V.FieldTables(km.GF2Field(spec.degree))
c = km.Circuit.from_file(sys.argv[2], 'kmb'); s_q, m_b, anc = V.validate_registers(c, spec)
k = int(sys.argv[3]); forced = [] if sys.argv[4] == 'none' else [int(x) for x in sys.argv[4].split(',')]
nb = int(sys.argv[5]) if len(sys.argv) > 5 else 1
rng = np.random.default_rng(12345); rows = np.arange(1, spec.n + 1, dtype=np.int64)[:, None]
tot = bad = 0; badpos = {}
for b in range(nb):
    syn = np.zeros((256, spec.n), np.int64); mbuc = np.zeros((256, spec.m), np.int64)
    for s in range(256):
        others = rng.choice(np.setdiff1d(np.arange(spec.m), forced), size=k - len(forced), replace=False)
        pos = np.concatenate([np.array(forced, np.int64), others]); vals = rng.integers(1, 4095, size=k, endpoint=True)
        exps = (T.log[vals][None, :] + rows * pos[None, :]) % T.order
        syn[s] = np.bitwise_xor.reduce(T.antilog[exps], axis=1)
    sim = km.Simulator(batch_size=256, debug_print=False, seed=b + 1); sim.use_same_registers_as(c)
    inp = V.to_bits(syn, spec.degree); sim.write_across_shots(km.array(s_q), inp); sim.write_across_shots(km.array(m_b), V.to_bits(mbuc, spec.degree))
    sim.do(c)
    regd = (sim.read_across_shots(km.array(s_q)) ^ inp).any(axis=0)
    tot += 256; bad += int(regd.sum())
print(f'weight {k} forced {forced}: {bad}/{tot} value errors')

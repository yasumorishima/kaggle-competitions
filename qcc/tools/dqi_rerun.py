"""Re-run the batches of failed DQI shots with other simulator seeds: is a failure set by the input or by HMR outcomes?
usage: python dqi_rerun.py <host dir> <dqi.kmb> <fails.jsonl> <n seeds>"""
import json, os, sys
import numpy as np
sys.argv_ = list(sys.argv)
H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi'), os.path.dirname(os.path.abspath(__file__))]
import kickmix as km
import verify_submission_dqi as V
import dqi_fails as F
c = km.Circuit.from_file(sys.argv[2], 'kmb'); spec = V.ProblemSpecDQI.benchmark_instance()
tables = V.FieldTables(km.GF2Field(spec.degree)); wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
s_q, m_b, anc = V.validate_registers(c, spec)
for line in open(sys.argv[3]):
    d = json.loads(line)
    syn, mbuc, ph, errs = F.sample(spec, tables, np.random.default_rng(d['seed']), 256, wp)
    fails = []
    for k in range(int(sys.argv[4])):
        sim = km.Simulator(batch_size=256, debug_print=False, seed=d['seed'] if k == 0 else 10**6 + k)
        sim.use_same_registers_as(c); inp = V.to_bits(syn, spec.degree)
        sim.write_across_shots(km.array(s_q), inp); sim.write_across_shots(km.array(m_b), V.to_bits(mbuc, spec.degree))
        sim.do(c)
        regd = sim.read_across_shots(km.array(s_q)) ^ inp
        fails.append(bool(regd[:, d['shot']].any()))
    print('shot w', d['weight'], 'fails with sim seeds:', fails, flush=True)

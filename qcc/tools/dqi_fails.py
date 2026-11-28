"""Find the DQI shots a circuit gets wrong, with their error words (our code; uses the hosts' verifier module).

usage: python dqi_fails.py <host dir> <dqi.kmb> <seed> <batches of 256> <out.jsonl> [workers]
Samples shots exactly as the verifier's sample_shots does (same rng call order) but keeps the error
positions and values, runs the circuit, and writes one json line per failed shot.
"""
import json, os, sys, multiprocessing as mp
import numpy as np

H = sys.argv[1]
sys.path[:0] = [os.path.join(H, 'kickmix'), os.path.join(H, 'verifiers', 'dqi')]
import kickmix as km
import verify_submission_dqi as V

def sample(spec, tables, rng, n, wp):
    max_value = (1 << tables.degree) - 1
    weights = rng.choice(len(wp), size=n, p=wp)
    mbuc = rng.integers(0, max_value, size=(n, spec.m), endpoint=True, dtype=np.int64)
    syn = np.zeros((n, spec.n), dtype=np.int64); ph = np.zeros(n, dtype=bool)
    rows = np.arange(1, spec.n + 1, dtype=np.int64)[:, None]; errs = [None] * n
    for shot in np.flatnonzero(weights):
        k = int(weights[shot])
        pos = rng.choice(spec.m, size=k, replace=False)
        vals = rng.integers(1, max_value, size=k, endpoint=True, dtype=np.int64)
        exps = (tables.log[vals][None, :] + rows * pos[None, :]) % tables.order
        syn[shot] = np.bitwise_xor.reduce(tables.antilog[exps], axis=1)
        ph[shot] = sum(int(x).bit_count() for x in vals & mbuc[shot, pos]) & 1 == 1
        errs[shot] = (pos.tolist(), vals.tolist())
    return syn, mbuc, ph, errs

G = {}
def init(path):
    G['c'] = km.Circuit.from_file(path, 'kmb')
    G['spec'] = V.ProblemSpecDQI.benchmark_instance() if hasattr(V.ProblemSpecDQI, 'benchmark_instance') else None

def run(task):
    i, seed = task
    c = G['c']; spec = G['spec']
    tables = V.FieldTables(km.GF2Field(spec.degree))
    wp = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    rng = np.random.default_rng(seed)
    syn, mbuc, ph, errs = sample(spec, tables, rng, 256, wp)
    s_q, m_b, anc = V.validate_registers(c, spec)
    sim = km.Simulator(batch_size=256, debug_print=False, seed=seed)
    sim.use_same_registers_as(c)
    inp = V.to_bits(syn, spec.degree)
    sim.write_across_shots(km.array(s_q), inp); sim.write_across_shots(km.array(m_b), V.to_bits(mbuc, spec.degree))
    sim.do(c)
    flip = sim.read_phase_flipped_across_shots()
    regd = sim.read_across_shots(km.array(s_q)) ^ inp
    junk = sim.read_across_shots(anc) if len(anc) else np.zeros((0, 256), bool)
    ve = regd.any(axis=0) | junk.any(axis=0); pe = (flip ^ ph) & ~ve
    out = []
    for s in np.flatnonzero(ve | pe):
        e = errs[s] or ([], [])
        out.append({'batch': i, 'seed': seed, 'shot': int(s), 'kind': 'value' if ve[s] else 'phase',
                    'weight': len(e[0]), 'pos': e[0], 'vals': e[1],
                    'bad_syndrome_bits': int(regd[:, s].sum()), 'bad_ancilla': int(junk[:, s].sum())})
    return out

if __name__ == '__main__':
    path, seed0, nb, outp = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    w = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    seeds = np.random.SeedSequence(seed0).generate_state(nb, dtype=np.uint64).tolist()
    n = 0
    with mp.get_context('fork').Pool(w, initializer=init, initargs=(path,)) as pool, open(outp, 'a') as f:
        for res in pool.imap_unordered(run, list(enumerate(seeds))):
            n += 1
            for r in res: f.write(json.dumps(r) + '\n'); f.flush()
            if n % 50 == 0: print(n * 256, 'shots', flush=True)
    print('done', nb * 256)

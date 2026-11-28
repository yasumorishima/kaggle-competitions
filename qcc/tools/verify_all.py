"""Run the hosts' three verifiers on a built submission at every salt; print a summary and the geometric-mean score.

usage: python verify_all.py <host dir (kickmix/, verifiers/)> <submission dir> "<salt1>|<salt2>|..." <dqi shots> [invsqrt shots per x]
invsqrt runs 1 shot per x by default (all 3,133,452 inputs; the board runs 64 with its own salt).
Prints "ALL PASS" only when every task passes at every salt.
"""
import json, math, os, subprocess, sys, time

# the hosts' sample-submission costs (the score is (prod REF/cost)^(1/3)); values as re-verified in
# teddybrown414's "Where the Points Are" notebook, checked there against the hosts' sample circuits
REF = {'invsqrt': 2076774.233715282, 'dqi': 24439463583672.93, 'chem': 3355599588.3155365}

def run(host, args, timeout=14400):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(host, 'kickmix'), os.path.join(host, 'verifiers')]))
    t = time.time()
    r = subprocess.run([sys.executable] + args, cwd=os.path.join(host, 'verifiers'), env=env,
                       capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; dec = json.JSONDecoder(); res = None
    for i, ch in enumerate(out):
        if ch == '{':
            try:
                o, _ = dec.raw_decode(out[i:])
                if isinstance(o, dict) and 'status' in o: res = o
            except json.JSONDecodeError: pass
    if res is None: res = {'status': 'FAIL', 'error': out[-2000:]}
    res['_sec'] = round(time.time() - t)
    return res

def main():
    host, sub, salts, shots = sys.argv[1], sys.argv[2], sys.argv[3].split('|'), int(sys.argv[4])
    spx = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    V = os.path.join(host, 'verifiers'); ok = True
    for salt in salts:
        s = ['--salt', salt] if salt else []
        res = {
            'invsqrt': run(host, [os.path.join(V, 'invsqrt/verify_submission_invsqrt.py'), os.path.join(sub, 'invsqrt.kmb'),
                                  '--shots-per-x', str(spx), '--batch-size', '4' if spx == 1 else '256'] + s),
            'dqi': run(host, [os.path.join(V, 'dqi/verify_submission_dqi.py'), os.path.join(sub, 'dqi.kmb'),
                              '--num-shots', str(shots)] + s),
            'chem': run(host, [os.path.join(V, 'dfthcsos/verify.py'), os.path.join(sub, 'dfthcsos')] + s),
        }
        line = []
        for k in ('invsqrt', 'dqi', 'chem'):
            r = res[k]; ok &= r.get('status') == 'PASS'
            extra = {'invsqrt': ['num_qubits', 'toffoli_count', 'worst_error_over_tolerance'],
                     'dqi': ['num_qubits', 'toffoli_count', 'value_errors', 'phase_errors', 'repetition_factor'],
                     'chem': ['lambda_eff', 'toffoli_count', 'num_qubits']}[k]
            print(f"salt {salt!r:12} {k:8} {r.get('status')} score {r.get('score')} " +
                  ' '.join(f"{e}={r.get(e)}" for e in extra) + f" ({r['_sec']} s)" + ('' if r.get('status') == 'PASS' else f" error={r.get('error')}"))
            line.append(r.get('score'))
        if all(isinstance(x, (int, float)) for x in line):
            gm = (REF['invsqrt'] / line[0] * REF['dqi'] / line[1] * REF['chem'] / line[2]) ** (1 / 3)
            print(f"salt {salt!r:12} geometric mean {gm:.5f}")
        sys.stdout.flush()
    print('ALL PASS' if ok else 'NOT ALL PASS')

if __name__ == '__main__':
    main()

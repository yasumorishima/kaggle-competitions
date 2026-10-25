"""Build kernels/e3/main.ipynb = two-pipeline fusion (ours, 10-25).

    python build_fu.py NAME "ARGS_A" "ARGS_B" [k=10] [wa=1] [wb=1]

A and B are two full builds of build.py (for example A = "base=f465 ownco=1 lfrag=1", B = "base=v22 ownco=1").
The kernel is a small orchestrator:
  1. Each build's code cells become one Python script (the bases have no magics). A header in each script filters
     glob.glob / glob.iglob under /kaggle/input to that build's own datasets: both bases look files up by name and take
     the shortest hit, so with the union of inputs attached one base could pick the other's file.
  2. A runs first in a subprocess with cwd /kaggle/working. Its submission.csv is kept, everything else it wrote under
     /kaggle/working is deleted. B then runs the same way with what is left of the 9 h (minus a margin); if B fails or
     runs out of time, the final file is A's.
  3. Fusion: per molecule, score(c) = wa / (k + rank_A(c)) + wb / (k + rank_B(c)) over the two top-25 lists
     (candidates matched by InChIKey first block when RDKit is importable, else by SMILES), the top 25 by score are
     written; ties keep A's order.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
name, args_a, args_b = sys.argv[1], sys.argv[2], sys.argv[3]
_kv = dict(a.split("=", 1) for a in sys.argv[4:])
K, WA, WB = float(_kv.get("k", 10)), float(_kv.get("wa", 1)), float(_kv.get("wb", 1))


def build(args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "build.py"), name + "_x"] + args.split(), cwd=HERE,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    nb = json.load(open(os.path.join(HERE, "main.ipynb"), encoding="utf-8"))
    meta = json.load(open(os.path.join(HERE, "kernel-metadata.json")))
    code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    md = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "markdown"]
    for s in code:
        for line in s.split("\n"):
            assert not line.lstrip().startswith(("!", "%")), ("magic in a base cell", line[:80])
    return code, md[0] if md else "", meta


code_a, md_a, meta_a = build(args_a)
code_b, md_b, meta_b = build(args_b)
assert meta_a.get("enable_gpu") == meta_b.get("enable_gpu") == "true", "both builds must be GPU builds"

HEADER = r'''# --- ours (fusion orchestrator): this script's own inputs only (all names kept inside a function) ---
def _ours_fu_install_filter(allowed, other_slugs):
    import glob as gm
    og, oig = gm.glob, gm.iglob
    def keep(p):
        p = str(p)
        if not p.startswith('/kaggle/input/'):
            return True
        parts = p[len('/kaggle/input/'):].split('/')
        if parts[0] == 'datasets' and len(parts) >= 3:
            return parts[1] + '/' + parts[2] in allowed
        return parts[0] not in other_slugs
    def g(*a, **k):
        return [p for p in og(*a, **k) if keep(p)]
    def ig(*a, **k):
        return (p for p in oig(*a, **k) if keep(p))
    gm.glob, gm.iglob = g, ig
    print('ours: input filter', sorted(allowed), flush=True)
_ours_fu_install_filter(set(%(allowed)r), set(%(other_slugs)r))
del _ours_fu_install_filter
try:
    from IPython.display import display
except Exception:
    display = print
'''


def script(code, meta, other_meta):
    allowed = sorted(meta["dataset_sources"])
    other = sorted(set(d.split("/")[1] for d in other_meta["dataset_sources"]) - set(d.split("/")[1] for d in allowed))
    body = HEADER % dict(allowed=allowed, other_slugs=other)
    for i, s in enumerate(code):
        body += f"\n# ===== cell {i} =====\n" + s + "\n"
    compile(body, "pipeline", "exec")   # syntax check at build time
    return body


SA, SB = script(code_a, meta_a, meta_b), script(code_b, meta_b, meta_a)

ORCH = r'''# ours: two-pipeline fusion orchestrator (built by enveda-casmi26/kernels/e3/build_fu.py)
import json, os, shutil, signal, subprocess, sys, time, csv
T0 = time.time()
W = '/kaggle/working'
SCRIPTS = %(scripts)s
K, WA, WB = %(k)r, %(wa)r, %(wb)r
LIMIT = 9 * 3600 - 25 * 60          # leave 25 min for the fusion and the kernel's own exit
os.makedirs('/tmp/ours_fu', exist_ok=True)
for nm, src in SCRIPTS.items():
    open(f'/tmp/ours_fu/{nm}.py', 'w').write(src)
KEEP = set(os.listdir(W))


def run(nm, timeout):
    print(f'=== ours: run {nm} (timeout {timeout:.0f}s, elapsed {time.time() - T0:.0f}s)', flush=True)
    t = time.time()
    p = subprocess.Popen([sys.executable, '-u', f'/tmp/ours_fu/{nm}.py'], cwd=W, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    tail = []
    try:
        import threading
        def pump():
            for line in p.stdout:
                print(f'[{nm}] ' + line, end='', flush=True)
        th = threading.Thread(target=pump, daemon=True); th.start()
        rc = p.wait(timeout=max(60, timeout))
        th.join(30)
    except subprocess.TimeoutExpired:
        print(f'=== ours: {nm} timed out, killing', flush=True)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception as e:
            print('kill failed', e)
        rc = -9
    sub = os.path.join(W, 'submission.csv')
    ok = rc == 0 and os.path.exists(sub)
    dst = f'/tmp/ours_fu/sub_{nm}.csv'
    if ok:
        shutil.copy(sub, dst)
    print(f'=== ours: {nm} rc={rc} ok={ok} in {time.time() - t:.0f}s', flush=True)
    for f in os.listdir(W):            # clean the working dir for the next pipeline
        if f in KEEP:
            continue
        q = os.path.join(W, f)
        try:
            shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
        except Exception as e:
            print('cleanup', f, e)
    return dst if ok else None


def read(path):
    rows = list(csv.reader(open(path)))
    return rows[0], {r[0]: [s for s in r[1].split(';') if s] for r in rows[1:]}


sub_a = run('A', LIMIT - (time.time() - T0))
sub_b = run('B', LIMIT - (time.time() - T0)) if sub_a else None
if sub_a is None:
    # A failed: B gets the whole remaining time; its file goes out alone
    sub_b = run('B', LIMIT - (time.time() - T0))
    assert sub_b, 'both pipelines failed'
    shutil.copy(sub_b, os.path.join(W, 'submission.csv'))
    print('=== ours: A failed, submission = B', flush=True)
elif sub_b is None:
    shutil.copy(sub_a, os.path.join(W, 'submission.csv'))
    print('=== ours: B failed, submission = A', flush=True)
else:
    head, LA = read(sub_a)
    _, LB = read(sub_b)
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        _cache = {}
        def key(s):
            if s not in _cache:
                m = Chem.MolFromSmiles(s)
                _cache[s] = Chem.MolToInchiKey(m)[:14] if m is not None else s
            return _cache[s]
        print('ours: fusion keys = InChIKey first block', flush=True)
    except Exception as e:
        key = lambda s: s
        print('ours: fusion keys = SMILES (no RDKit:', e, ')', flush=True)
    out, n_top1, n_b_only, n_mol = {}, 0, 0, 0
    for mid, la in LA.items():
        lb = LB.get(mid, [])
        score, first, smi = {}, {}, {}
        for w, lst in ((WA, la), (WB, lb)):
            seen = set()
            for r, s in enumerate(lst[:25]):
                kk = key(s)
                if kk in seen:
                    continue
                seen.add(kk)
                score[kk] = score.get(kk, 0.0) + w / (K + r + 1)
                smi.setdefault(kk, s)
        order_a = {key(s): i for i, s in enumerate(la)}
        fused = sorted(score, key=lambda kk: (-score[kk], order_a.get(kk, 1e9)))[:25]
        out[mid] = [smi[kk] for kk in fused]
        n_mol += 1
        if la and out[mid] and key(la[0]) != fused[0]:
            n_top1 += 1
        n_b_only += sum(1 for kk in fused if kk not in order_a)
    with open(os.path.join(W, 'submission.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(head)
        for mid, lst in out.items():
            wr.writerow([mid, ';'.join(lst)])
    print(f'=== ours: fused {n_mol} molecules; top-1 changed vs A in {n_top1}; B-only rows in the top 25: {n_b_only}; '
          f'molecules only in B: {len(set(LB) - set(LA))}', flush=True)
print(f'=== ours: done in {time.time() - T0:.0f}s', flush=True)
'''

orch = ORCH % dict(scripts=json.dumps({"A": SA, "B": SB}), k=K, wa=WA, wb=WB)
compile(orch, "orch", "exec")
md = ("# casmi26 e3" + name + " (ours: two-pipeline fusion)\n\n"
      f"Ours: the two builds below run one after the other in subprocesses (each sees only its own datasets) and their "
      f"top-25 lists are fused per molecule by weighted reciprocal rank (k={K:g}, wa={WA:g}, wb={WB:g}); if B fails or runs "
      f"out of time the file is A's. Built by enveda-casmi26/kernels/e3/build_fu.py.\n\n"
      f"## A: build.py {args_a}\n\n" + md_a + "\n\n## B: build.py " + args_b + "\n\n" + md_b)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": orch}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "main.ipynb"), "w", encoding="utf-8"), indent=1)
meta = dict(meta_a)
meta["id"], meta["title"] = f"yasunorim/casmi26-e3{name}", f"casmi26 e3{name}"
meta["dataset_sources"] = sorted(set(meta_a["dataset_sources"]) | set(meta_b["dataset_sources"]))
meta["kernel_sources"] = sorted(set(meta_a["kernel_sources"]) | set(meta_b["kernel_sources"]))
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote main.ipynb", meta["id"], "A:", args_a, "| B:", args_b, "| k", K, "wa", WA, "wb", WB,
      "| scripts", len(SA), len(SB), "| datasets", len(meta["dataset_sources"]))

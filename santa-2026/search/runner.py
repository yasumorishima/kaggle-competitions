"""Our search runner for Santa 2026 (runs inside a Kaggle kernel, not in the cloud session).

Runs the public C++ strategies in third_party/ (Wang Jakie's graph search, Atakan Aldemir's optimizers; Apache-2.0,
see the NOTICE.md files) on our pool, rotating strategies and seeds until a time budget runs out. What is ours:
the scheduling (fresh-first: a puzzle that just improved is re-searched at once with new seeds), the per-puzzle
archive of near-length distinct programs fed back as crossover tapes, and the sharding over many free machines.
Every program an engine returns is replayed by our referee (tools/common.py) before it counts.

usage: python3 search/runner.py --pool POOL.csv --puzzles puzzles.csv --bin BINDIR --out OUT.csv
         [--ids ids.txt] [--shard i/n] [--minutes 60] [--jobs 4] [--archive arch.jsonl] [--archive-out new.csv]
OUT.csv holds only the puzzles that improved (id,program).
"""
import argparse, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from common import load_puzzles, load_sub, replay, write_sub

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True); ap.add_argument("--puzzles", required=True)
ap.add_argument("--bin", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ids", default=""); ap.add_argument("--shard", default="0/1")
ap.add_argument("--minutes", type=float, default=60); ap.add_argument("--jobs", type=int, default=os.cpu_count())
ap.add_argument("--archive", default="", help="jsonl of near-length programs to use as crossover tapes")
ap.add_argument("--archive-out", default="", help="CSV (id,program; many rows per id) of new near-length programs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fresh-ids", default="", help="ids that just got shorter (e.g. in the public frontier): searched first")
ap.add_argument("--fresh-repeat", type=int, default=12)
ap.add_argument("--plan", default="anneal_seeded:4000,incremental_seeded:4000,anneal_seeded:20000,incremental_seeded:20000,quartet:4000")
ap.add_argument("--fresh-plan", default="frontier:0,frontier:1,frontier:2,closure:0,closure:1,closure:2,anneal_seeded:20000,incremental_seeded:20000",
                help="strategies for fresh puzzles; frontier:i / closure:i are Atakan's optimizers with config i")
ap.add_argument("--log", default="")
a = ap.parse_args()

P = load_puzzles(a.puzzles); POOL = load_sub(a.pool)
ids = [l.strip() for l in open(a.ids) if l.strip() in P] if a.ids else sorted(P)
si, sn = map(int, a.shard.split("/")); ids = ids[si::sn]
rng = random.Random(a.seed or (time.time_ns() ^ os.getpid()))
plan = [(s.split(":")[0], int(s.split(":")[1])) for s in a.plan.split(",")]
fplan = [(s.split(":")[0], int(s.split(":")[1])) for s in a.fresh_plan.split(",")]
# Atakan Aldemir's optimizers (third_party/atakan, Apache-2.0): the argument sets goodpjw2008 used on changed puzzles.
ATK = {"frontier": ["8 4200 9 10 9 {salt} 1500", "14 8000 12 13 9 {salt} 3000", "8 4200 9 10 13 {salt} 1500"],
       "closure": ["90 26 7 {salt} 4150", "92 32 4 {salt} 5250", "100 28 9 {salt} 4950"]}
fplan = [x for x in fplan if x[0] not in ATK or os.path.exists(os.path.join(a.bin, x[0]))] or plan
best = {k: POOL[k] for k in ids}; start_len = {k: len(best[k]) for k in ids}
arch = {k: [] for k in ids}; seen = {k: {json.dumps(best[k])} for k in ids}
if a.archive and os.path.exists(a.archive):
    for ln in open(a.archive):
        try: r = json.loads(ln)
        except Exception: continue
        k = r.get("k")
        if k in arch and json.dumps(r["p"]) not in seen[k]:
            try: replay(*P[k], r["p"])
            except Exception: continue
            seen[k].add(json.dumps(r["p"])); arch[k].append(r["p"])
new_arch = []
log = open(a.log, "a") if a.log else sys.stdout

def atk_job(k, strat, cfg, seed):
    s, t, L = P[k]; tg = sorted(set(t)); prog = best[k]
    case = "\n".join([f"{k} {L} {len(s)} {len(tg)} {len(prog)}", " ".join(map(str, s)), " ".join(map(str, tg))]
                     + [f"{op} {x} {y} {d}" for op, x, y, d in prog])
    argv = [os.path.join(a.bin, strat)] + ATK[strat][cfg % len(ATK[strat])].format(salt=seed).split()
    try:
        r = subprocess.run(argv, input=case + "\n", capture_output=True, text=True, timeout=1800)
    except Exception:
        return k, None
    lines = r.stdout.splitlines()
    if not lines or len(lines[0].split()) < 2: return k, None
    n = int(lines[0].split()[1])
    try:
        out = [[x[0], int(x[1]), int(x[2]), int(x[3])] for x in (l.split() for l in lines[1:1 + n])]
        replay(s, t, L, out)
    except Exception:
        return k, None
    return k, out

def job(k, strat, rounds, seed):
    if strat in ATK: return atk_job(k, strat, rounds, seed)
    s, t, L = P[k]
    r2 = random.Random(seed)
    tapes = [best[k]] + r2.sample(arch[k], min(6, len(arch[k])))
    rows = [f"{L} 48 {rounds} 0 {seed}", " ".join(map(str, (len(s), *s))),
            " ".join(map(str, (len(set(t)), *sorted(set(t))))), str(len(tapes))]
    for tp in tapes:
        rows.append(str(len(tp))); rows += [f"{op} {x} {y} {d}" for op, x, y, d in tp]
    try:
        r = subprocess.run([os.path.join(a.bin, strat)], input="1\n" + "\n".join(rows) + "\n",
                           capture_output=True, text=True, timeout=1800)
    except Exception:
        return k, None
    tok = r.stdout.split()
    if not tok: return k, None
    prog = [[tok[i], int(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])] for i in range(1, len(tok), 4)]
    try: replay(s, t, L, prog)
    except Exception: return k, None
    return k, prog

# Work queue: a fresh queue (puzzles that just improved, re-searched first with new seeds) and a sweep queue
# (every puzzle in turn, cycling through the plan).
deadline = time.time() + a.minutes * 60; t0 = time.time()
fresh, sweep, step = [], [], 0
# A program that just changed still has easy gains near it (goodpjw2008 measured ~12 instructions/hour on fresh
# programs against ~2 on converged ones), so the puzzles that got shorter since the last cycle go first, many times.
if a.fresh_ids and os.path.exists(a.fresh_ids):
    fids = [l.strip() for l in open(a.fresh_ids) if l.strip() in best]
    fresh = [k for _ in range(a.fresh_repeat) for k in fids]
    print(f"fresh ids in this shard: {len(fids)}", file=log, flush=True)
def next_task():
    global sweep, step
    if fresh: k = fresh.pop(0); st, rd = fplan[rng.randrange(len(fplan))]; return k, st, rd
    if not sweep:
        sweep = ids[:]; rng.shuffle(sweep); step += 1
    st, rd = plan[(step - 1) % len(plan)]
    return sweep.pop(), st, rd

done_jobs = 0
with ThreadPoolExecutor(a.jobs) as ex:
    running = set()
    while True:
        while len(running) < a.jobs and time.time() < deadline:
            k, st, rd = next_task()
            running.add(ex.submit(job, k, st, rd, rng.randrange(1, 2**31)))
        if not running: break
        fin, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
        running = set(running)
        for f in fin:
            done_jobs += 1
            k, prog = f.result()
            if prog is None: continue
            key = json.dumps(prog)
            if len(prog) < len(best[k]):
                print(f"[{time.time() - t0:7.0f}s] {k}: {len(best[k])} -> {len(prog)}", file=log, flush=True)
                best[k] = prog; seen[k].add(key)
                fresh += [k] * 3
            elif len(prog) <= len(best[k]) + 2 and key not in seen[k]:
                seen[k].add(key); arch[k].append(prog); new_arch.append((k, prog))
        if done_jobs and done_jobs % 200 == 0:
            tot = sum(start_len[k] - len(best[k]) for k in ids)
            print(f"[{time.time() - t0:7.0f}s] jobs={done_jobs} sweep_step={step} gain_total={tot}", file=log, flush=True)
            imp = {k: best[k] for k in ids if len(best[k]) < start_len[k]}
            write_sub(a.out, imp)

imp = {k: best[k] for k in ids if len(best[k]) < start_len[k]}
write_sub(a.out, imp)
if a.archive_out:
    import csv
    with open(a.archive_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for k, prog in new_arch: w.writerow([k, json.dumps(prog, separators=(",", ":"))])
print(f"done: jobs {done_jobs}, improved {len(imp)} puzzles, gain {sum(start_len[k] - len(best[k]) for k in imp)}",
      file=log, flush=True)

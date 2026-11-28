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
# triwalk only (11-25 GHA run 45599624803, 16 shards x 320 min): triwalk shortened 96 of 12,500 jobs (60 s: 51/5,960,
# 40 s on fresh puzzles: 12/947, 120 s: 24/2,593, 30 s: 9/3,000), Wang Jakie / Atakan 0 of 3,159. Per second, 60 s and 40 s win.
# trilin:S (11-28): triwalk from another lineage (archive programs up to --lin-excess longer than best). goodpjw2008's
# search_stats.csv (11-28): lineages 70 instructions in 7.3 h, best-polish 616 in 35.7 h; our best-polish gains are soon
# matched by public files (our lead over public fell 67 -> 31 in two days), so half the sweeps go to other lineages.
ap.add_argument("--plan", default="triwalk:60,trilin:30,triwalk:60,trilin:30,triwalk:120")
ap.add_argument("--fresh-plan", default="triwalk:40",
                help="strategies for fresh puzzles; frontier:i / closure:i are Atakan's optimizers with config i")
ap.add_argument("--log", default="")
ap.add_argument("--lin-excess", type=int, default=5, help="trilin: start from archive programs at most this much longer than best")
a = ap.parse_args()

P = load_puzzles(a.puzzles); POOL = load_sub(a.pool)
ids = [l.strip() for l in open(a.ids) if l.strip() in P] if a.ids else sorted(P)
si, sn = map(int, a.shard.split("/")); ids = ids[si::sn]
rng = random.Random(a.seed or (time.time_ns() ^ os.getpid()))
plan = [(s.split(":")[0], int(s.split(":")[1])) for s in a.plan.split(",")]
fplan = [(s.split(":")[0], int(s.split(":")[1])) for s in a.fresh_plan.split(",")]
# Atakan Aldemir's optimizers (third_party/atakan, Apache-2.0): the argument sets goodpjw2008 used on changed puzzles.
# goodpjw2008's triwalk (third_party/goodpjw2008, Apache-2.0): anneals the SET of values a program computes. The option
# sets are the defaults of the author's tri_driver.py. "triwalk:S" runs one of them for S seconds.
TW = ["--cap 1 --T 0.2 --rb 0.6 --deep 0.1", "--cap 1 --T 0.2 --rb 0.85 --deep 0.4", "--cap 2 --T 0.2 --rb 0.6 --deep 0.1",
      "--cap 2 --T 0.2 --rb 0.85 --deep 0.4", "--cap 3 --T 0.2 --rb 0.85 --deep 0.2", "--cap 2 --T 0.2 --rb 0.85 --deep 0"]
ATK = {"frontier": ["8 4200 9 10 9 {salt} 1500", "14 8000 12 13 9 {salt} 3000", "8 4200 9 10 13 {salt} 1500"],
       "closure": ["90 26 7 {salt} 4150", "92 32 4 {salt} 5250", "100 28 9 {salt} 4950"]}
have = lambda st: os.path.exists(os.path.join(a.bin, "triwalk" if st == "trilin" else st))
plan = [x for x in plan if have(x[0])] or plan
fplan = [x for x in fplan if have(x[0])] or plan
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

tries = {}   # (k, archive index) -> trilin walks started from that lineage in this run
def tw_job(k, secs, seed, lineage=False):
    s, t, L = P[k]; tg = sorted(set(t)); r2 = random.Random(seed); cfg = r2.randrange(len(TW)); idx = -1
    # "trilin": start from another lineage, i.e. an archive program a little longer than best (from other public files
    # or an earlier walk), fewest tries and smallest excess first. Public teams polish their best programs with the same
    # engine, so gains on our best are soon matched; a different lineage can reach a length the best-polish does not.
    lin = [i for i, p in enumerate(arch[k]) if len(best[k]) < len(p) <= len(best[k]) + a.lin_excess] if lineage else []
    if lin:
        idx = min(lin, key=lambda i: (tries.get((k, i), 0) + 1) * (1 + 0.5 * (len(arch[k][i]) - len(best[k]))) * (0.5 + r2.random()))
        tries[(k, idx)] = tries.get((k, idx), 0) + 1; prog = arch[k][idx]
    else:
        # Half the time start from an equal-length variant (a "V" end state of an earlier walk) so the walk keeps moving.
        alts = [p for p in arch[k] if len(p) == len(best[k])]
        prog = r2.choice(alts) if alts and r2.random() < 0.5 else best[k]
    head = ["P", k, str(len(s))] + [str(x) for x in s] + [str(len(tg))] + [str(x) for x in tg] + [str(L), "48", str(len(prog))]
    case = " ".join(head) + "\n" + "".join(f"{op} {x} {y} {d}\n" for op, x, y, d in prog)
    argv = [os.path.join(a.bin, "triwalk"), "--time", str(secs), "--seed", str(seed)] + TW[cfg].split()
    try:
        r = subprocess.run(argv, input=case, capture_output=True, text=True, timeout=secs * 4 + 60)
    except Exception:
        return k, None, cfg, idx, prog
    lines = r.stdout.splitlines()
    for i, ln in enumerate(lines):
        f = ln.split()
        if len(f) == 4 and f[0] in ("R", "V") and f[1] == k:
            n = int(f[3])
            try:
                out = [[x[0], int(x[1]), int(x[2]), int(x[3])] for x in (l.split() for l in lines[i + 1:i + 1 + n])]
                replay(s, t, L, out)
            except Exception:
                return k, None, cfg, idx, prog
            return k, out, cfg, idx, prog
    return k, list(prog), cfg, idx, prog   # no shorter program and no new variant: same as the input

def job(k, strat, rounds, seed):
    t = time.time()
    if strat in ("triwalk", "trilin"):
        k, prog, cfg, idx, start = tw_job(k, rounds, seed, strat == "trilin")
        name = f"trilin:{rounds}/nolin" if strat == "trilin" and idx < 0 else f"{strat}:{rounds}/c{cfg}"   # nolin: no lineage, walked best
        return k, prog, name, time.time() - t, idx, start
    k, prog = (atk_job if strat in ATK else wj_job)(k, strat, rounds, seed)
    return k, prog, f"{strat}:{rounds}", time.time() - t, -1, None

def wj_job(k, strat, rounds, seed):
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
# Per-strategy diagnostics (lengths only): jobs, failed (no valid program), delta = len(out) - len(best) buckets, seconds.
ST = {}
def stat(name, d, sec, lin=0):
    r = ST.setdefault(name, {"n": 0, "fail": 0, "lt0": 0, "eq0": 0, "p1_2": 0, "p3_10": 0, "gt10": 0, "lin": 0, "sec": 0.0})
    r["lin"] += lin
    r["n"] += 1; r["sec"] += sec
    if d is None: r["fail"] += 1
    elif d < 0: r["lt0"] += 1
    elif d == 0: r["eq0"] += 1
    elif d <= 2: r["p1_2"] += 1
    elif d <= 10: r["p3_10"] += 1
    else: r["gt10"] += 1
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
            k, prog, sname, sec, idx, start = f.result()
            lg = max(0, len(start) - len(prog)) if idx >= 0 and prog is not None else 0
            stat(sname, None if prog is None else len(prog) - len(best[k]), sec, lg)
            if prog is None: continue
            key = json.dumps(prog)
            if idx >= 0 and len(prog) >= len(best[k]):
                # keep walking this lineage from its new state (shorter, or an equal-length move); saved for later cycles
                if len(prog) <= len(arch[k][idx]) and key not in seen[k]:
                    seen[k].add(key); arch[k][idx] = prog; new_arch.append((k, prog))
                continue
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
for name, r in sorted(ST.items()):
    print(f"strat {name}: n={r['n']} fail={r['fail']} better={r['lt0']} same={r['eq0']} +1..2={r['p1_2']} "
          f"+3..10={r['p3_10']} >10={r['gt10']} lin={r['lin']} avg_sec={r['sec'] / max(1, r['n']):.0f}", file=log, flush=True)
print(f"done: jobs {done_jobs}, improved {len(imp)} puzzles, gain {sum(start_len[k] - len(best[k]) for k in imp)}",
      file=log, flush=True)

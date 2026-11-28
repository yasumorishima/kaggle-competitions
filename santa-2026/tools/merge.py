"""Per-puzzle shortest valid program over many program files (data only: parsed with csv+json, replayed exactly).

usage: python3 -I tools/merge.py --puzzles puzzles.csv --out best.csv [--report sources.csv] PATH [PATH ...]
PATH is a CSV with header id,program, or a directory searched recursively for such CSVs.
The first PATH given wins ties (put our pool first so credit and stability stay with it).
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_puzzles, replay, write_sub

ap = argparse.ArgumentParser()
ap.add_argument("--puzzles", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--report", default=""); ap.add_argument("--archive", default="", help="jsonl of distinct programs within --slack of the best")
ap.add_argument("--slack", type=int, default=5)
ap.add_argument("--max-eq", type=int, default=8, help="archive: equal-length variants kept per puzzle (newest first)")
ap.add_argument("--max-lin", type=int, default=10, help="archive: longer programs kept per puzzle, chosen as distinct lineages")
ap.add_argument("--max-ov", type=float, default=0.7, help="archive: a longer program is a new lineage if it shares less than this "
                "fraction of its intermediate values with the best and with every lineage kept so far")
ap.add_argument("paths", nargs="+")
a = ap.parse_args()
P = load_puzzles(a.puzzles)

def files(path):
    if os.path.isfile(path): yield path; return
    for root, _, names in sorted(os.walk(path)):
        for n in sorted(names):
            if n.endswith(".csv") and n not in ("puzzles.csv", "practice_puzzles.csv", "sample_submission.csv"):
                yield os.path.join(root, n)

best, who, allp = {}, {}, {}
for path in a.paths:
    for f in files(path):
        try:
            rd = csv.reader(open(f, encoding="utf-8-sig", newline=""))
            if next(rd, None) != ["id", "program"]: continue
            ok = tot = 0
            for row in rd:
                if len(row) != 2 or row[0] not in P: continue
                try:
                    prog = json.loads(row[1]); n = replay(*P[row[0]], prog)
                except Exception:
                    continue
                ok += 1; tot += n
                allp.setdefault(row[0], {})[json.dumps(prog, separators=(",", ":"))] = n
                if row[0] not in best or n < len(best[row[0]]):
                    best[row[0]], who[row[0]] = prog, f
            print(f"{f}: {ok} valid programs" + (f", total {tot}" if ok == len(P) else ""))
        except Exception as e:
            print(f"{f}: skipped ({type(e).__name__})")
missing = set(P) - set(best)
assert not missing, f"{len(missing)} puzzles without a valid program"
write_sub(a.out, best)
print(f"merged score {sum(len(v) for v in best.values())}")
if a.report:
    with open(a.report, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "instructions", "source"])
        for k in sorted(best): w.writerow([k, len(best[k]), who[k]])
def mids(k, prog):
    """Intermediate values a program computes (not targets): two programs with few in common are different lineages."""
    start, targets, _ = P[k]; regs = list(start) + [None] * 48; T = set(targets); out = set()
    for op, i, j, d in prog:
        x, y = regs[i], regs[j]
        v = x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else x // y
        regs[d] = v
        if v not in T: out.add(v)
    return out

if a.archive:
    m = cnt_eq = cnt_lin = 0
    with open(a.archive, "w") as fh:
        for k in sorted(allp):
            bkey = json.dumps(best[k], separators=(",", ":")); nb = len(best[k])
            items = list(allp[k].items())   # insertion order = path order; later paths are newer (shards, gha, public)
            eq = [key for key, n in items if n == nb and key != bkey][::-1][:a.max_eq]
            longer = sorted(((n, -i, key) for i, (key, n) in enumerate(items) if nb < n <= nb + a.slack))
            reps, lin = [mids(k, best[k])], []
            for n, _, key in longer:
                if len(lin) >= a.max_lin: break
                mm = mids(k, json.loads(key))
                if mm and all(len(mm & r) / max(1, min(len(mm), len(r))) < a.max_ov for r in reps):
                    reps.append(mm); lin.append(key)
            for key in eq + lin: fh.write('{"k":"%s","p":%s}\n' % (k, key)); m += 1
            cnt_eq += len(eq); cnt_lin += len(lin)
    print(f"archive: {m} programs ({cnt_eq} equal-length variants, {cnt_lin} longer lineages within +{a.slack})")

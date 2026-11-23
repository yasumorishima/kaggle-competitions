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
ap.add_argument("--slack", type=int, default=3); ap.add_argument("paths", nargs="+")
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
if a.archive:
    m = 0
    with open(a.archive, "w") as fh:
        for k in sorted(allp):
            for key, n in allp[k].items():
                if n <= len(best[k]) + a.slack and key != json.dumps(best[k], separators=(",", ":")):
                    fh.write('{"k":"%s","p":%s}\n' % (k, key)); m += 1
    print(f"archive: {m} near-length programs")

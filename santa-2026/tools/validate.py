"""python3 -I tools/validate.py PUZZLES.csv SUB.csv -> checks every program, prints the score."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_puzzles, load_sub, replay
P = load_puzzles(sys.argv[1]); S = load_sub(sys.argv[2])
assert set(P) == set(S), "puzzle ids differ"
bad = 0; tot = 0
for pid in sorted(P):
    try: tot += replay(*P[pid], S[pid])
    except AssertionError as e: bad += 1; print("INVALID", pid, e)
print(f"programs {len(S)}  invalid {bad}  score {tot}")
sys.exit(1 if bad else 0)

"""Sum the per-strategy lines ("strat NAME: n=.. fail=.. ...") of many search logs read from stdin."""
import re, sys, collections
A = collections.defaultdict(lambda: collections.Counter())
for l in sys.stdin:
    m = re.match(r"strat (\S+): (.*)", l.strip())
    if not m: continue
    for k, v in re.findall(r"(\S+?)=(\d+)", m.group(2)):
        if k != "avg_sec": A[m.group(1)][k] += int(v)
for name, c in sorted(A.items()):
    print(f"{sys.argv[1] if len(sys.argv) > 1 else ''}strat {name}: " + " ".join(f"{k}={c[k]}" for k in
          ["n", "fail", "better", "same", "+1..2", "+3..10", ">10", "lin"]))

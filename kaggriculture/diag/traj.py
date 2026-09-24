"""Day-by-day state of both farms: money, hands, quadrants, herd, crops, shed,
and the market verbs each side issued that day.

    A=agents/v49_sched.py python diag/traj.py 86000 16

This is what found the router's opening (2026-09-24): 12 melon tiles + 2 cows
+ 2 sheep on day 0, $16,689 on day 10 against our $319.
"""
import collections
import os
import sys

from _common import agent_path, opponent
from kaggle_environments import make

A = agent_path(os.environ.get("A", "agents/v49_sched.py"))
B = opponent()
seed = int(sys.argv[1])
days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
env.run([A, B])
st = env.steps
print("final", [int(s.reward or 0) for s in st[-1]])


def snap(t, p):
    o = st[t][p].observation
    f = o["farms"][p]
    tl = [x for r in f["tiles"] for x in r if isinstance(x, dict)]
    an = collections.Counter(x["animal"] for x in tl if x.get("animal"))
    cr = collections.Counter(x.get("crop") for x in tl if x.get("kind") == "PLANT")
    sh = o["private"]["shed"]
    return "%6d h%-2d q%d an%s cr%s sh%s" % (
        f["money"], len(f["hands"]), len(f["unlocked_quadrants"]), dict(an), dict(cr),
        {k: v for k, v in sh.items() if v})


def verbs(t0, t1, p):
    c = collections.Counter()
    for t in range(t0, min(t1, len(st))):
        for m in (st[t][p].action or {}).get("market", []) or []:
            if m:
                c[m[0]] += 1
    return dict(c)


for d in range(days):
    t = min(d * 24 + 23, len(st) - 1)
    for p in (0, 1):
        print(d, "AB"[p], snap(t, p), verbs(d * 24 + 1, t + 2, p))

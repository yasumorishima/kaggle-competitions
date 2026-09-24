"""Per-product cash ledger for both farms, taken from the actual market fills.

    AGENT=agents/v49_sched.py python diag/ledger.py base '{}' 86000

`_process_market` calls `_commit_unit` as a module global, so wrapping that
attribute records every unit that really changed hands -- orders alone lie,
because a replayed tape issues many that never fill. The farm objects are
identified by rebinding their ids at the top of every market phase.
"""
import collections as C
import json
import os
import sys

from _common import OUT, agent_path, opponent
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as KG
from sweep import build_variant

os.makedirs(OUT, exist_ok=True)
src = open(agent_path(os.environ.get("AGENT", "agents/v49_sched.py")), encoding="utf-8").read()
name, over, seed = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
opp = opponent()
a = build_variant(src, name, over, OUT)

REV = [C.Counter(), C.Counter()]
UNITS = [C.Counter(), C.Counter()]
COST = [C.Counter(), C.Counter()]
CUNITS = [C.Counter(), C.Counter()]
DAY = [0]
FIRST = {}
IDS = {}
_commit = KG._commit_unit
_market = KG._process_market


def process(state, env):
    IDS.clear()
    for i, f in enumerate(state[0].observation["farms"]):
        IDS[id(f)] = i
    return _market(state, env)


def commit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _commit(op, item, price, farm, private, market, shed_capacity)
    if ok:
        pid = IDS.get(id(farm))
        if pid is not None:
            if op == "SELL":
                REV[pid][item] += price
                UNITS[pid][item] += 1
                FIRST.setdefault((pid, item), DAY[0])
            else:
                COST[pid][item] += price
                CUNITS[pid][item] += 1
    return ok


KG._commit_unit = commit
KG._process_market = process

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
_interp = env.interpreter


def interp(state, e=None, *a2, **kw):
    DAY[0] = len(env.steps) // 24
    return _interp(state, e, *a2, **kw)


env.interpreter = interp
env.run([a, opp])
st = env.steps
print(name, seed, "opp", os.path.basename(opp), "final", [s.reward for s in st[-1]])
for pid in (0, 1):
    print("f%d revenue %7d   spend %7d" % (pid, sum(REV[pid].values()), sum(COST[pid].values())))
    for item, v in REV[pid].most_common():
        print("    sell %-11s %7d  units %5d  avg %6.1f  first_day %s"
              % (item, v, UNITS[pid][item], v / max(1, UNITS[pid][item]), FIRST.get((pid, item))))
    for item, v in COST[pid].most_common():
        print("    buy  %-11s %7d  units %5d" % (item, v, CUNITS[pid][item]))

"""A few knob variants on a few seeds, final coins plus money at the end of day 10.

    python diag/probe.py '{"base":{},"m12h":{"open_melon":[[0,12]],"harvest_at_cap":true}}' 86000,82000

A quick look only: one season's margin scatters by ~35,000, so two seeds decide
nothing. Judge with the sweep workflow (twin 96 games / router 48 games).
Check first that "base" reproduces the emitted agent to the coin.
"""
import json
import os
import sys

from _common import OUT, agent_path, opponent
from kaggle_environments import make
from sweep import build_variant

os.makedirs(OUT, exist_ok=True)
opp = opponent()
src = open(agent_path(os.environ.get("AGENT", "agents/v49_sched.py")), encoding="utf-8").read()
variants = json.loads(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split(",")]
for name, over in variants.items():
    a = build_variant(src, name, over, OUT)
    for sd in seeds:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": sd})
        env.run([a, opp])
        st = env.steps
        m10 = st[10 * 24 + 23][0].observation["farms"][0]["money"]
        print(name, sd, [int(s.reward or 0) for s in st[-1]], "day10money", int(m10), flush=True)

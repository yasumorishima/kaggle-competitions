#!/usr/bin/env python
"""Run main.py's policy under a capital calendar.

The calendar decides what the farm owns -- hands, herd, quadrants -- and the
policy decides everything else. See sim/schedule.py for why the split falls
there.

main.py is loaded fresh for each agent, so two agents in the same process (the
two sides of one episode, or a parent and a child) never share the per-season
memo the policy keeps.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "main.py")


def load_main(path=MAIN):
    spec = importlib.util.spec_from_file_location("kagri_main_%d" % id(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(sched, path=MAIN):
    """An agent callable running `sched`. sched=None is plain main.py."""
    mod = load_main(path)
    mod.SCHEDULE = sched

    def agent(obs, config=None):
        return mod.agent(obs, config)

    return agent


def selftest():
    """A schedule must reach the environment, and no schedule must change nothing.

    The second half is the one worth running: the hooks sit in the middle of
    the purchase logic, and a mistake there would quietly move the baseline
    every sweep is measured against.
    """
    plain = load_main()
    assert plain.SCHEDULE is None, "main.py ships with a schedule set"
    assert plain._sched_for(0) is None
    scheduled = load_main()
    scheduled.SCHEDULE = {"0": {"COW": 3, "land": 1}, "6": {"COW": 8, "land": 2}}
    assert scheduled._sched_for(0) == {"COW": 3, "land": 1}
    assert scheduled._sched_for(5) == {"COW": 3, "land": 1}, "an entry must hold"
    assert scheduled._sched_for(9) == {"COW": 8, "land": 2}
    assert plain._sched_for(9) is None, "modules are sharing state"
    print("OK sched_agent")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())

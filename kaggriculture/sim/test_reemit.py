"""A re-emitted agent must play exactly like the frozen one it came from.

Emitted agents are frozen copies of main.py, so adding a knob makes them
unsweepable ("has no such knob") and they have to be rebuilt. That rebuild
quietly carries every change made to main.py since they were frozen, which is
the same failure this project has already paid for twice: a version compared by
its P dict rather than by its behaviour, and a "tie" that turned out to be an
agent without the mechanism in it at all.

So the rebuild is checked the only way that means anything -- both agents are
run over the same scenes and their actions compared byte for byte. A default
that moved shows up here as a differing order, not as a season's money three
hours later.

Usage: python sim/test_reemit.py [frozen.py] [rebuilt.py]
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import knob_bite as kb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def load(path):
    spec = importlib.util.spec_from_file_location(
        "agent_" + os.path.basename(path).replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm(actions):
    return json.dumps(json.loads(actions) if isinstance(actions, str) else actions,
                      sort_keys=True)


def main():
    old = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "agents/v38_sched.py")
    # Rebuild here rather than compare two files on disk: a committed rebuild
    # goes stale the moment main.py moves again, and then this test pins a pair
    # that no longer says anything about the path actually being used.
    import reemit, emit_sched, tempfile
    sched, over = reemit.extract(old)
    source = emit_sched.emit(sched, note="rebuilt by test_reemit")
    if over:
        source += "P.update(" + json.dumps(over, sort_keys=True) + ")\n"
    fd, new = tempfile.mkstemp(suffix="_reemit.py")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    try:
        return compare(load(old), load(new), old)
    finally:
        os.unlink(new)


def compare(a, b, old):
    print(f"rebuilding {os.path.basename(old)} on today's main.py")
    check("the rebuild carries the same calendar",
          json.dumps(a.SCHEDULE, sort_keys=True) == json.dumps(b.SCHEDULE, sort_keys=True),
          f"{len(a.SCHEDULE or {})} days vs {len(b.SCHEDULE or {})}")

    # Knobs the frozen copy has must agree; knobs only the rebuild has are the
    # point of the exercise and are reported, not failed.
    differ = [k for k in a.P if k in b.P and a.P[k] != b.P[k]]
    check("every knob they share has the same value", not differ,
          ", ".join(f"{k}: {a.P[k]!r} vs {b.P[k]!r}" for k in differ))
    added = sorted(set(b.P) - set(a.P))
    print(f"     knobs new in the rebuild: {', '.join(added) if added else '(none)'}")

    scenes = kb.scenes()
    same = 0
    for name, obs in scenes:
        if norm(kb.run(a, obs)) == norm(kb.run(b, obs)):
            same += 1
        else:
            check(f"scene {name!r} plays the same", False)
    check("every scene plays identically", same == len(scenes),
          f"{same}/{len(scenes)}")

    print("\n" + ("FAILED: " + ", ".join(FAILED) if FAILED else "the rebuild is behaviourally identical"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

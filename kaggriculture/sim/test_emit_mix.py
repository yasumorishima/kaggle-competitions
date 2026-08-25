#!/usr/bin/env python
"""Does the split agent actually split where it says it does?

`emit_mix` exists to answer one question -- is a recorded plan's transfer
failure in its labour or in its order book -- and the answer is only worth
anything if each half really comes from where it claims. A wrapper that
silently returned the policy for both halves would produce two identical
scores and read as "the split makes no difference", which is the most
expensive kind of wrong.

So pin it without the environment: build a scene, run the policy alone, run
both emitted agents on the same scene, and check each field against its
claimed source.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util  # noqa: E402

import emit_mix  # noqa: E402
import knob_bite as kb  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def load_src(src, name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_{name}.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.remove(path)


def fake_plan(steps=720):
    """A plan nothing could produce by accident, so its fingerprints show."""
    plan = []
    for i in range(steps):
        plan.append({
            "farmer": ["DIG"],
            "hands": [["DIG"], ["DIG"], ["DIG"], ["DIG"], ["DIG"], ["DIG"]],
            "market": [["SELL", "WHEAT", 7]],
        })
    return plan


def main():
    plan = fake_plan()
    _n, obs = kb.scene("mid season", shed={"WHEAT": 40, "FERTILIZER": 0},
                       day=9, hour=11, step=9 * 24 + 11)

    policy = kb.load_agent()
    base = policy.agent(json.loads(json.dumps(obs)))

    lab = load_src(emit_mix.emit(plan, "labour"), "mix_lab")
    mkt = load_src(emit_mix.emit(plan, "market"), "mix_mkt")
    a = lab.agent(json.loads(json.dumps(obs)))
    b = mkt.agent(json.loads(json.dumps(obs)))

    n_hands = len(obs["farms"][0]["hands"])

    check("labour half takes the farmer from the plan", a["farmer"] == ["DIG"], str(a["farmer"]))
    check("labour half takes the hands from the plan",
          all(h == ["DIG"] for h in a["hands"]), str(a["hands"][:2]))
    check("labour half aligns the roster", len(a["hands"]) == n_hands,
          f"{len(a['hands'])} vs {n_hands}")
    check("labour half takes the market from the policy",
          a["market"] == base["market"], f"{a['market']} vs {base['market']}")

    check("market half takes the market from the plan",
          b["market"] == [["SELL", "WHEAT", 7]], str(b["market"]))
    check("market half takes the farmer from the policy", b["farmer"] == base["farmer"],
          f"{b['farmer']} vs {base['farmer']}")
    check("market half takes the hands from the policy", b["hands"] == base["hands"])

    # The two halves must not collapse into the same agent, which is the
    # failure mode that would read as "the split does not matter".
    check("the two halves differ", a != b)
    check("neither half is just the policy", a != base and b != base)

    # A plan shorter than the season must not wrap around to step 0; the
    # replay clamps to the last step instead.
    short = load_src(emit_mix.emit(plan[:5], "market"), "mix_short")
    c = short.agent(json.loads(json.dumps(obs)))
    check("a short plan clamps rather than wrapping",
          c["market"] == [["SELL", "WHEAT", 7]])

    # And a plan whose steps are malformed must fall all the way back to the
    # policy, not return half an action.
    broken = load_src(emit_mix.emit([{"farmer": None}], "labour"), "mix_broken")
    d = broken.agent(json.loads(json.dumps(obs)))
    check("a broken plan falls back to the whole policy", d == base)

    print()
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("all emit_mix checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

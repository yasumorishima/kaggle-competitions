#!/usr/bin/env python
"""Invariants for optimize.py's mutation operators, with no environment needed.

Two failure modes are worth a test rather than a sweep.

The first is a mutation that produces a plan the farm cannot play -- a hand
that exists at a step where the farm has not hired one, an empty action, a sell
of zero units, an eleventh market order on a turn that takes ten. The replay
would paper over most of that by padding and trimming, so the plan would score
badly for reasons nothing in the log explains.

The second is an operator that never actually edits anything. That one is worse
because it is invisible: the search still runs, still reports generations, and
still concludes the plan cannot be improved -- having spent its entire budget
proposing the incumbent back to itself. `build_shed_weight` was inert at 1.0
for exactly this reason and cost a whole sweep before knob_bite caught it, so
every operator here has to be shown to bite.

Usage:
    python sim/test_plan_ops.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimize as O

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("COW", "SHEEP", "GOOSE")


def fixture(days=30, seed=1):
    """A plan with the shape a real recording has: a roster that grows, units
    that walk then work, and market orders on a minority of steps."""
    rng = random.Random(seed)
    plan = []
    for day in range(days):
        hands = min(11, day // 3)          # the roster grows as it is hired
        for _hour in range(O.TURNS):
            def unit():
                roll = rng.random()
                if roll < 0.45:
                    return [rng.choice(O.MOVES)]
                if roll < 0.60:
                    return [O.IDLE]
                if roll < 0.72:
                    return ["PLANT", rng.choice(CROPS)]
                if roll < 0.84:
                    return ["CARE", rng.choice(ANIMALS)]
                return [rng.choice(("HARVEST", "WATER", "DIG", "COLLECT"))]

            market = []
            if rng.random() < 0.25:
                for _ in range(rng.randrange(1, 4)):
                    market.append([rng.choice(("SELL", "BUY")),
                                   rng.choice(CROPS), rng.randrange(1, 20)])
            plan.append({"farmer": unit(),
                         "hands": [unit() for _ in range(hands)],
                         "market": market})
    return plan


def roster(plan):
    """How many hands exist at each step -- the shape a mutation must preserve."""
    return [len(a.get("hands") or []) for a in plan]


def check_valid(plan, where):
    assert len(plan) == 720, "%s: plan length changed to %d" % (where, len(plan))
    for t, action in enumerate(plan):
        units = [action.get("farmer")] + list(action.get("hands") or [])
        for slot, op in enumerate(units):
            assert isinstance(op, list) and op, \
                "%s: step %d slot %d is not an action: %r" % (where, t, slot, op)
            assert isinstance(op[0], str), \
                "%s: step %d slot %d verb is not a string: %r" % (where, t, slot, op)
        market = action.get("market") or []
        assert len(market) <= O.MARKET_CAP, \
            "%s: step %d has %d market orders" % (where, t, len(market))
        for order in market:
            assert len(order) >= 3, "%s: step %d short order %r" % (where, t, order)
            assert int(order[2]) >= 1, \
                "%s: step %d order of %r units" % (where, t, order[2])


def main():
    base = fixture()
    check_valid(base, "fixture")
    base_roster = roster(base)
    rng = random.Random(11)

    failures = []
    print("operator                 applied  changed  roster-kept")
    for fn, _weight in O.MUTATIONS:
        applied = changed = roster_ok = 0
        trials = 300
        for _ in range(trials):
            plan = [dict(a, hands=[list(h) for h in a["hands"]],
                         farmer=list(a["farmer"]),
                         market=[list(o) for o in a["market"]]) for a in base]
            got = fn(plan, rng)
            if got is None:
                continue
            applied += 1
            check_valid(plan, fn.__name__)
            if roster(plan) == base_roster:
                roster_ok += 1
            if plan != base:
                changed += 1
        print("  %-22s %6d %8d %12d" % (fn.__name__, applied, changed, roster_ok))
        # An operator that cannot fire, or fires without editing, silently
        # eats the search budget; one that grows the roster invents a hand the
        # farm never hired.
        if applied < trials * 0.5:
            failures.append("%s applied only %d/%d times" % (fn.__name__, applied, trials))
        if changed < applied * 0.9:
            failures.append("%s left the plan untouched %d/%d times"
                            % (fn.__name__, applied - changed, applied))
        if roster_ok != applied:
            failures.append("%s changed the roster %d/%d times"
                            % (fn.__name__, applied - roster_ok, applied))

    # mutate() itself: several operators at once must still leave a valid plan.
    for i in range(200):
        child, names = O.mutate(base, random.Random(i), 6)
        check_valid(child, "mutate")
        if roster(child) != base_roster:
            failures.append("mutate changed the roster on trial %d (%s)" % (i, names))
            break
        if child == base:
            failures.append("mutate returned the incumbent unchanged on trial %d" % i)
            break

    # The plan a deep-copying mutation starts from must not be touched, or the
    # incumbent quietly drifts and every score after it is measured against a
    # parent that no longer exists.
    before = fixture()
    O.mutate(before, random.Random(3), 20)
    if before != fixture():
        failures.append("mutate edited its input plan in place")

    print("")
    for line in failures:
        print("FAIL " + line)
    print("VERDICT=" + ("PASS" if not failures else "FAIL"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

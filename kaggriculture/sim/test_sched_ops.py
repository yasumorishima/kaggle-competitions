#!/usr/bin/env python
"""Invariants for the capital calendar, with no environment in the loop.

The same trap as test_plan_ops.py guards: an operator that never fires, or
that produces a calendar the farm cannot follow, would burn a whole climb
without ever saying so. The list-side version of this test earned its keep --
two operators in optimize.py were silently no-ops until it was written -- and
the calendar has the extra hazard that its columns are cumulative, so a
careless edit can ask the farm to return a cow it already owns.

Run: python sim/test_sched_ops.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schedule as sched_mod            # noqa: E402
import optimize_schedule as opt         # noqa: E402

BASE = {
    "0": {"hands": 5, "COW": 1, "SHEEP": 4, "GOOSE": 0, "land": 1},
    "6": {"hands": 4, "COW": 4, "SHEEP": 4, "GOOSE": 0, "land": 2},
    "10": {"hands": 14, "COW": 8, "SHEEP": 4, "GOOSE": 0, "land": 3},
    "20": {"hands": 12, "COW": 8, "SHEEP": 4, "GOOSE": 0, "land": 3},
}

FAILED = []


def check(label, ok, detail=""):
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", label,
                           "" if ok else "  " + detail))
    if not ok:
        FAILED.append(label)


def rows_of(sched):
    rows = sched_mod.expand(sched)
    for row in rows:
        row.setdefault("hands", 0)
        row.setdefault("land", 1)
        for s in opt.SPECIES:
            row.setdefault(s, 0)
    return rows


def main():
    sched_mod.validate(BASE)
    print("operators (each must fire, and each must change the calendar)")
    for op in opt.OPERATORS:
        rng = random.Random(11)
        fired = changed = 0
        trials = 300
        for _ in range(trials):
            rows = rows_of(BASE)
            before = [dict(r) for r in rows]
            if op(rows, rng) is None:
                continue
            fired += 1
            opt._tidy(rows)
            if rows != before:
                changed += 1
        check("%-22s fires" % op.__name__, fired >= trials * 0.5,
              "fired %d/%d" % (fired, trials))
        check("%-22s changes it" % op.__name__, changed >= fired * 0.9,
              "changed %d of %d firings" % (changed, fired))

    print("mutation keeps the calendar followable")
    rng = random.Random(3)
    bad = 0
    sched = dict(BASE)
    for _ in range(2000):
        child, _applied = opt.mutate(sched, rng, ops=3)
        try:
            sched_mod.validate(child)
        except AssertionError as exc:
            bad += 1
            print("     ", exc)
            break
        sched = child                       # walk, so late calendars get tested
    check("2000 chained mutations validate", bad == 0)

    print("cumulative columns never fall")
    rng = random.Random(5)
    falls = 0
    for _ in range(500):
        child, _ = opt.mutate(BASE, rng, ops=4)
        rows = rows_of(child)
        for key in opt.SPECIES + ("land",):
            column = [r[key] for r in rows]
            if any(b < a for a, b in zip(column, column[1:])):
                falls += 1
                break
    check("no species or quadrant is given back", falls == 0,
          "%d of 500 calendars fell" % falls)

    print("bounds hold")
    rng = random.Random(9)
    out_of_range = 0
    for _ in range(500):
        child, _ = opt.mutate(BASE, rng, ops=4)
        for row in rows_of(child):
            if not (0 <= row["hands"] <= opt.MAX_HANDS):
                out_of_range += 1
            if not (1 <= row["land"] <= opt.MAX_LAND):
                out_of_range += 1
    check("hands and land stay in range", out_of_range == 0,
          "%d violations" % out_of_range)

    print("round trip")
    rebuilt = sched_mod.compress({str(d): r for d, r in enumerate(rows_of(BASE))})
    check("expand then compress is identity",
          sched_mod.expand(rebuilt) == sched_mod.expand(BASE))

    print("")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

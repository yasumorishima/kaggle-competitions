#!/usr/bin/env python
"""The capital calendar: what the farm should own, day by day.

Why a calendar and not a plan. Recording this agent and hill-climbing the
720-step list it produces does not transfer -- measured, the same list earned
60k on the seeds it was climbed on and 25k on held-out ones, and 137
generations moved the training win count 7->8 while the held-out count sat at
1 out of 12 the whole way. A published list does not behave like that: 155,980
on the same training seeds and 170,675 held out. The difference is not the
climb, it is what a recording of a reactive policy contains. Most of its
720 steps encode which tile happened to need water at hour 9 of that season,
and none of that is true of the next one.

What did differ between the two farms, read off the two action lists:

    animals            12 (8 cows, 4 sheep)   against 27 (12 cows, 9 geese, 6 sheep)
    care delivered     100% of animal-days    against 62%
    wheat bought/sold  189 / 479  (net +290)  against 329 / 274  (net -55)
    waterings          1,010                  against 595
    walking steps      2,855                  against 4,390
    quadrants bought   2 (days 6 and 10)      against 1 (day 8)

Every line of that is capital, and capital is season-independent: "carry eight
cows and buy the second quadrant on day 6" is as true of one season as the
next, while "water the tile at (3,7) at hour 9" is true of exactly one. So the
calendar is what gets searched and the policy keeps the field work.

A schedule is {day: {hands, COW, SHEEP, GOOSE, land}}, cumulative, holding
until the next entry, with any key omitted falling through to the policy.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_shape  # noqa: E402

TURNS = route_shape.TURNS
SPECIES = ("COW", "SHEEP", "GOOSE")
KEYS = ("hands",) + SPECIES + ("land",)
DAYS = 30


def from_plan(plan):
    """Read the calendar a recorded plan actually ran.

    Hands are counted from the roster the plan carries, animals from its
    BUY_ANIMAL orders and land from its BUY_LAND orders, all accumulated. This
    is the honest starting point for a search: whatever else a strong plan is
    doing, this is the capital it is doing it with.
    """
    hands, bought, land = {}, {s: 0 for s in SPECIES}, 1
    out = {}
    for step, action in enumerate(plan):
        day = step // TURNS
        hands[day] = max(hands.get(day, 0), len(action.get("hands") or []))
        for order in action.get("market") or []:
            if not order:
                continue
            verb = str(order[0])
            if verb == "BUY_ANIMAL":
                species = str(order[1]) if len(order) > 1 else ""
                qty = int(order[2]) if len(order) > 2 else 1
                if species in bought:
                    bought[species] += qty
            elif verb == "BUY_LAND":
                land += 1
        entry = {"hands": hands[day], "land": land}
        entry.update(bought)
        out[str(day)] = dict(entry)
    return compress(out)


def compress(sched):
    """Drop days that repeat the previous day, since entries already hold."""
    out, prev = {}, None
    for key in sorted(sched, key=lambda k: int(k)):
        entry = {k: v for k, v in sched[key].items() if k in KEYS}
        if entry != prev:
            out[key] = entry
            prev = entry
    return out


def expand(sched):
    """Per-day view, for printing and for mutation."""
    rows, live = [], {}
    for day in range(DAYS):
        if str(day) in sched:
            live = dict(sched[str(day)])
        rows.append(dict(live))
    return rows


def validate(sched):
    """A schedule the environment could actually be asked to follow."""
    assert isinstance(sched, dict) and sched, "empty schedule"
    rows = expand(sched)
    for day, row in enumerate(rows):
        for key, value in row.items():
            assert key in KEYS, "unknown key %r" % key
            assert isinstance(value, int), "%s is not an integer" % key
            assert value >= 0, "%s went negative on day %d" % (key, day)
        if "land" in row:
            assert 1 <= row["land"] <= 4, "land %d on day %d" % (row["land"], day)
    # Cumulative targets may not fall: the farm cannot un-buy a quadrant, and a
    # falling head count would ask it to slaughter stock it paid for.
    for key in SPECIES + ("land",):
        seen = [r[key] for r in rows if key in r]
        for a, b in zip(seen, seen[1:]):
            assert b >= a, "%s falls from %d to %d" % (key, a, b)
    return True


def dump(sched, name=""):
    rows = expand(sched)
    print("== %s ==" % (name or "schedule"))
    print("%3s %6s %5s %6s %6s %5s" % ("day", "hands", "COW", "SHEEP", "GOOSE", "land"))
    prev = None
    for day, row in enumerate(rows):
        if row == prev:
            continue
        prev = row
        print("%3d %6s %5s %6s %6s %5s"
              % (day, row.get("hands", "-"), row.get("COW", "-"),
                 row.get("SHEEP", "-"), row.get("GOOSE", "-"), row.get("land", "-")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="a plan JSON or an agent .py with an embedded list")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sched = from_plan(route_shape.load_plan(args.plan))
    validate(sched)
    dump(sched, args.plan)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sched, f, indent=1, sort_keys=True)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

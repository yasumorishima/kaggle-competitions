#!/usr/bin/env python
"""What economy a plan runs, read off the action list alone.

route_shape.py answers how hard a plan's labour works; this answers what it
works on. Both come out of the list without simulating anything, which matters
because the interesting comparison is against a published plan, and a published
plan is a file rather than an agent that can be instrumented.

What it reports, per day and in total: who gets hired and when, which animals
are bought and when, what gets planted, what is sold and in what quantity, and
what is bought back. Those five lines are the farm's whole strategy -- a plan
that ends the season with eight cows made that decision in the first week, and
the day it made it is visible here and nowhere else.

Accepts a plan JSON from record.py or an agent .py with an embedded list.

Usage:
    python sim/plan_economy.py plan.json
    python sim/plan_economy.py plan.json --other opponents/theirs.py
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_shape  # noqa: E402

TURNS = route_shape.TURNS


def orders_by_day(plan):
    """day -> verb -> Counter(item -> quantity)."""
    out = defaultdict(lambda: defaultdict(Counter))
    for step, action in enumerate(plan):
        day = step // TURNS
        for order in action.get("market") or []:
            if not order:
                continue
            verb = str(order[0])
            item = str(order[1]) if len(order) > 1 else ""
            try:
                qty = int(order[2]) if len(order) > 2 else 1
            except (TypeError, ValueError):
                qty = 1
            out[day][verb][item] += qty
    return out


def unit_ops_by_day(plan):
    """day -> verb -> Counter(argument -> count), over farmer and hands."""
    out = defaultdict(lambda: defaultdict(Counter))
    for step, action in enumerate(plan):
        day = step // TURNS
        units = [action.get("farmer") or ["PASS"]]
        units += [h or ["PASS"] for h in (action.get("hands") or [])]
        for op in units:
            verb = str(op[0])
            arg = str(op[1]) if len(op) > 1 else ""
            out[day][verb][arg] += 1
    return out


def roster_by_day(plan):
    """day -> hands present at the last step of that day."""
    out = {}
    for step, action in enumerate(plan):
        out[step // TURNS] = len(action.get("hands") or [])
    return out


def fmt(counter, limit=6):
    items = [(k, v) for k, v in counter.most_common(limit) if v]
    return " ".join("%s:%d" % (k or "-", v) for k, v in items) or "-"


def report(plan, name):
    market = orders_by_day(plan)
    ops = unit_ops_by_day(plan)
    hands = roster_by_day(plan)
    days = max(max(market or [0]), max(ops or [0]), max(hands or [0])) + 1

    print("== %s ==" % name)
    print("%3s %5s  %-26s %-30s %s" % ("day", "hands", "bought", "sold", "planted / cared"))
    for day in range(days):
        m, o = market.get(day, {}), ops.get(day, {})
        bought = Counter()
        for verb in ("BUY_ANIMAL", "BUY_PRODUCT", "BUY_LAND", "BUILD"):
            for k, v in m.get(verb, {}).items():
                bought[(k or verb)] += v
        work = Counter()
        for k, v in o.get("PLANT", {}).items():
            work["plant:" + (k or "?")] += v
        care = sum(o.get("CARE", {}).values())
        if care:
            work["CARE"] = care
        print("%3d %5d  %-26s %-30s %s"
              % (day, hands.get(day, 0), fmt(bought, 3),
                 fmt(m.get("SELL", Counter()), 4), fmt(work, 4)))

    tot_sell, tot_buy, tot_ops = Counter(), Counter(), Counter()
    for day in market:
        for k, v in market[day].get("SELL", {}).items():
            tot_sell[k] += v
        for verb in ("BUY_ANIMAL", "BUY_PRODUCT"):
            for k, v in market[day].get(verb, {}).items():
                tot_buy[verb + " " + k] += v
    hires = sum(sum(market[d].get("HIRE", {}).values()) for d in market)
    for day in ops:
        for verb, args in ops[day].items():
            tot_ops[verb] += sum(args.values())

    print("")
    print("sold      " + fmt(tot_sell, 12))
    print("bought    " + fmt(tot_buy, 12))
    print("hires     %d   peak hands %d" % (hires, max(hands.values() or [0])))
    print("actions   " + fmt(tot_ops, 14))
    print("SHAPE     " + route_shape_line(plan))
    print("")


def route_shape_line(plan):
    import record
    try:
        return record.shape(plan)
    except Exception as exc:                      # layout needs no env, but be safe
        return "unavailable (%s)" % exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--other", default="")
    args = ap.parse_args()
    report(route_shape.load_plan(args.plan), args.plan)
    if args.other:
        report(route_shape.load_plan(args.other), args.other)
    return 0


if __name__ == "__main__":
    sys.exit(main())

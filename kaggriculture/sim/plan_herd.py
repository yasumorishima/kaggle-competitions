#!/usr/bin/env python
"""Make herd-sized variants of a plan, so the size of the herd can be tested
as a starting point rather than as a policy knob.

The knob version of this question came back a tie (+1,535). That test could
only stop the purchases; it had no way to spend the freed labour on anything
else, and the agent went on walking the same routes. On an action list the
same question is a different one, because whatever the removed animals were
costing -- the wheat to feed them, the trips to reach them -- is freed the
moment the order goes, and the climb's other operators can put those turns
somewhere useful.

Measured, on the two plans in hand: theirs runs 8 cows and 4 sheep and gets
100% of the herd's daily care delivered; ours runs 12 cows, 9 geese and 6
sheep and delivers 62%. Ours is a net buyer of wheat (329 in, 274 out) while
theirs is a net seller (189 in, 479 out), and ours waters 595 times against
their 1,010. One herd is being fed by the farm; the other is feeding it.

The CARE and FEED actions aimed at a removed animal are left where they are.
They cost a turn and return nothing, so every variant here is strictly worse
than the same herd planned from scratch -- the point is to find out whether
the herd is worth its keep even carrying that waste, not to produce a
finished plan.

Usage:
    python sim/plan_herd.py in.json --drop GOOSE --out out.json
    python sim/plan_herd.py in.json --keep COW=8,SHEEP=4 --out out.json
"""
import argparse
import copy
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_shape  # noqa: E402


def herd_of(plan):
    got = Counter()
    for action in plan:
        for order in action.get("market") or []:
            if order and str(order[0]) == "BUY_ANIMAL":
                qty = int(order[2]) if len(order) > 2 else 1
                got[str(order[1])] += qty
    return got


def edit(plan, drop=(), keep=None):
    """Remove BUY_ANIMAL orders, in the order the plan issues them.

    keep is a species -> cap map; anything bought beyond the cap goes. A
    partial order is trimmed rather than dropped, so `keep COW=8` on a plan
    that buys two cows at once on day 6 keeps one of them.
    """
    out = copy.deepcopy(plan)
    seen = Counter()
    for action in out:
        orders = action.get("market")
        if not orders:
            continue
        kept = []
        for order in orders:
            if not order or str(order[0]) != "BUY_ANIMAL":
                kept.append(order)
                continue
            species = str(order[1])
            qty = int(order[2]) if len(order) > 2 else 1
            if species in drop:
                continue
            cap = (keep or {}).get(species)
            if cap is not None:
                room = max(0, cap - seen[species])
                if room == 0:
                    continue
                qty = min(qty, room)
                order = list(order)
                if len(order) > 2:
                    order[2] = qty
            seen[species] += qty
            kept.append(order)
        action["market"] = kept
    return out


def parse_keep(text):
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, num = part.partition("=")
        out[name.strip().upper()] = int(num)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--drop", default="", help="species to remove entirely")
    ap.add_argument("--keep", default="", help="SPECIES=N,SPECIES=N caps")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    plan = route_shape.load_plan(args.plan)
    drop = {s.strip().upper() for s in args.drop.split(",") if s.strip()}
    keep = parse_keep(args.keep) if args.keep else None
    out = edit(plan, drop=drop, keep=keep)

    before, after = herd_of(plan), herd_of(out)
    assert len(out) == len(plan), "the plan changed length"
    for species, n in after.items():
        assert n <= before[species], "%s grew" % species
    print("herd  before %s  ->  after %s"
          % (dict(before) or "-", dict(after) or "-"))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

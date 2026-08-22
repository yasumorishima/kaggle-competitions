#!/usr/bin/env python
"""Does the calendar actually change what the farm orders?

The same one-second question knob_bite.py asks of a knob, asked of the three
schedule hooks. A hook that silently fails to bite would not crash and would
not look wrong: the climb would run its full four hours, accept mutations on
season noise, and report a number. Cheaper to find out here.

Environment-free -- these are hand-built observations, so the answer arrives
in under a second and needs nothing installed.

Run: python sim/sched_bite.py
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import knob_bite                        # noqa: E402
import sched_agent                      # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", label, "" if ok else "  " + detail))
    if not ok:
        FAILED.append(label)


def pastures(n, unlocked=("NW",)):
    """n empty pastures, so a scheduled herd has somewhere to stand."""
    tiles = knob_bite.blank_tiles(unlocked)
    for i in range(n):
        tiles[1][i] = {"kind": "PASTURE"}
    return tiles


def orders(sched, obs):
    agent = sched_agent.make(sched)
    return agent(copy.deepcopy(obs), None).get("market") or []


def count(market, verb, item=None):
    total = 0
    for order in market:
        if str(order[0]) != verb:
            continue
        if item is not None and (len(order) < 2 or str(order[1]) != item):
            continue
        total += int(order[2]) if len(order) > 2 else 1
    return total


def main():
    print("herd")
    _n, obs = knob_bite.scene("herd", tiles=pastures(6), money=6000.0, day=0, hour=6)
    plain = orders(None, obs)
    big = orders({"0": {"COW": 5}}, obs)
    none = orders({"0": {"COW": 0, "SHEEP": 0, "GOOSE": 0}}, obs)
    print("     policy=%d  COW:5=%d  all-zero=%d"
          % (count(plain, "BUY_ANIMAL"), count(big, "BUY_ANIMAL", "COW"),
             count(none, "BUY_ANIMAL")))
    check("a scheduled herd is bought", count(big, "BUY_ANIMAL", "COW") > 0)
    check("a bigger target buys more than the policy chose",
          count(big, "BUY_ANIMAL", "COW") >= count(plain, "BUY_ANIMAL", "COW"))
    check("a zero target buys nothing", count(none, "BUY_ANIMAL") == 0,
          "bought %d" % count(none, "BUY_ANIMAL"))

    print("herd already met")
    tiles = pastures(6)
    for i in range(3):
        tiles[2][i] = knob_bite.animal_tile("PASTURE", "COW")
    _n, obs2 = knob_bite.scene("met", tiles=tiles, money=6000.0, day=0, hour=6)
    check("a target already standing in the field is not re-bought",
          count(orders({"0": {"COW": 3}}, obs2), "BUY_ANIMAL", "COW") == 0)
    check("a target above what is standing buys the difference",
          count(orders({"0": {"COW": 5}}, obs2), "BUY_ANIMAL", "COW") > 0)

    print("land")
    _n, obs3 = knob_bite.scene("land", money=20000.0, day=1, hour=6)
    check("a scheduled quadrant is bought early",
          count(orders({"0": {"land": 2}}, obs3), "BUY_LAND") == 1,
          "policy alone bought %d" % count(orders(None, obs3), "BUY_LAND"))
    check("land already owned is not re-bought",
          count(orders({"0": {"land": 1}}, obs3), "BUY_LAND") == 0)

    print("hands")
    _n, obs4 = knob_bite.scene("hands", money=20000.0, day=5, hour=1, hires_today=0)
    few = count(orders({"0": {"hands": 0}}, obs4), "HIRE")
    many = count(orders({"0": {"hands": 20}}, obs4), "HIRE")
    print("     hands:0=%d  hands:20=%d  policy=%d"
          % (few, many, count(orders(None, obs4), "HIRE")))
    check("a bigger roster hires more", many > few)

    print("no schedule changes nothing")
    same = True
    for scene in (obs, obs2, obs3, obs4):
        if orders(None, scene) != orders(None, scene):
            same = False
    check("the unscheduled path is stable", same)

    print("")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("the calendar bites")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

    print("crop dial")
    _n, obs5 = knob_bite.scene("crops", money=9000.0, day=2, hour=6)

    def planted(sched):
        act = sched_agent.make(sched)(copy.deepcopy(obs5), None)
        units = [act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])
        crops = [str(u[1]) for u in units if u and str(u[0]) == "PLANT" and len(u) > 1]
        seeds = count(act.get("market") or [], "BUY_SEED", "STRAWBERRY")
        return crops, seeds

    off_crops, off_seeds = planted({"0": {"STRAWBERRY_pct": 0}})
    on_crops, on_seeds = planted({"0": {"STRAWBERRY_pct": 400}})
    base_crops, base_seeds = planted(None)
    print("     strawberry seeds bought  0%%=%d  policy=%d  400%%=%d"
          % (off_seeds, base_seeds, on_seeds))
    check("the dial reaches the plan",
          (off_seeds, off_crops) != (on_seeds, on_crops),
          "0%% and 400%% produced the same turn")
    check("100 is the policy's own choice",
          planted({"0": {"STRAWBERRY_pct": 100}}) == (base_crops, base_seeds))

    print("labour dial")
    # A mid-season farm with both kinds of work standing open at once: two
    # plants wanting water and two animals wanting care. Under one scoring rule
    # the hands pick whichever price says, and the point of the dial is that
    # the day gets to overrule that. A dial that could not change this turn
    # would be indistinguishable from one that is inert.
    tiles = knob_bite.blank_tiles()
    tiles[4][4] = knob_bite.animal_tile("PASTURE", "COW", fed=True, yield_units=0)
    tiles[4][5] = knob_bite.animal_tile("PASTURE", "COW", fed=True, yield_units=0)
    tiles[3][1] = knob_bite.plant_tile("STRAWBERRY", day=6, watered=False, yield_units=0)
    tiles[3][2] = knob_bite.plant_tile("MELON", day=6, watered=False, yield_units=0)
    _n, obs6 = knob_bite.scene("labour", tiles=tiles, day=9, hour=8, money=1500.0,
                               farmer=[4, 3], hands=[[4, 4], [2, 3], [5, 4]],
                               shed={"WHEAT": 20, "FERTILIZER": 0},
                               hires_today=3)

    def work(sched):
        act = sched_agent.make(sched)(copy.deepcopy(obs6), None)
        units = [act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])
        return sorted(str(u[0]) for u in units if u)

    plain_work = work(None)
    water_up = work({"0": {"WATER_w": 400, "CARE_w": 0}})
    care_up = work({"0": {"CARE_w": 400, "WATER_w": 0}})
    print("     policy=%s" % ",".join(plain_work))
    print("     water-first=%s" % ",".join(water_up))
    print("     care-first=%s" % ",".join(care_up))
    check("the labour dial reaches the hands", water_up != care_up,
          "both extremes produced the same turn")
    check("100 everywhere is the policy's own choice",
          work({"0": {"WATER_w": 100, "CARE_w": 100, "HARVEST_w": 100}}) == plain_work)
    check("an empty calendar entry changes nothing", work({"0": {}}) == plain_work)

    print("the calendar reserves what it is about to buy")
    # knob_bite cannot reach this one: it only exists when a calendar is
    # active, and knob_bite loads the bare policy. Measured on seed 2000, the
    # calendar asked for four sheep on day 3 and got them on day 15 because
    # the seed queue, which runs first, had already spent the money.

    def opening(reserve, money):
        mod = sched_agent.load_main()
        mod.P["sched_reserve"] = reserve
        mod.SCHEDULE = {"0": {"COW": 1, "SHEEP": 4, "GOOSE": 0,
                             "hands": 5, "land": 1}}
        _n, ob = knob_bite.scene("opening", money=money, day=0, hour=6,
                                 tiles=knob_bite.blank_tiles(), seeds={})
        return mod.agent(copy.deepcopy(ob), None).get("market") or []

    tight_off, tight_on = opening(0.0, 1800.0), opening(1.0, 1800.0)
    print("     $1800 sheep bought  reserve0=%d  reserve1=%d"
          % (count(tight_off, "BUY_ANIMAL", "SHEEP"),
             count(tight_on, "BUY_ANIMAL", "SHEEP")))
    check("a tight opening buys more herd with the reserve on",
          count(tight_on, "BUY_ANIMAL", "SHEEP")
          > count(tight_off, "BUY_ANIMAL", "SHEEP"))
    check("the reserve is off by default",
          opening(0.0, 2400.0) == opening(0.0, 2400.0)
          and sched_agent.load_main().P["sched_reserve"] == 0.0)

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

#!/usr/bin/env python
"""Does a knob change what the agent does? Answer it before spending a sweep.

Three of the knobs in the last two rounds came back with a delta of exactly
zero over forty games -- not "no effect worth adopting" but "the same actions,
every step". A sweep is an expensive way to learn that. This asks the agent
directly: build a handful of representative observations, call `agent()` once
per knob setting, and diff the actions.

It is not a simulator and makes no claim about money. It answers a narrower
question: on this turn, with this board, does the setting change what the agent
returns?

Read a NONE carefully -- it has two meanings and they look identical. Either no
scene reaches the decision the knob guards, or the knob only moves the plan and
the plan takes turns to show. `fill_idle` raises a planting target, but the
seed queue is already buying its five a turn, so the first turn is unchanged
while a whole season is not. A knob that bites here definitely reaches the
agent; a knob that does not needs a scene built for it before the NONE means
anything.

Scale matters as much as reachability. `build_shed_weight` at 1.0 is invisible
because it cancels the 1/(1+d) discount exactly for a unit standing at the rim;
it starts changing placements at 3.0. Sweep the value this tool says bites, not
the first one that seemed reasonable.

Usage:
    python sim/knob_bite.py                       # the built-in knob list
    python sim/knob_bite.py '{"finish_tile": 3.0}'
"""
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = 10


def load_agent():
    spec = importlib.util.spec_from_file_location(
        "agent_under_test", os.path.join(HERE, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blank_tiles(unlocked=("NW",)):
    def quad(x, y):
        return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
    return [[None if quad(x, y) in unlocked else "LOCKED"
             for x in range(BOARD)] for y in range(BOARD)]


def scene(name, **over):
    """A plausible mid-turn observation. Defaults describe an opening turn."""
    tiles = over.pop("tiles", None) or blank_tiles(over.pop("unlocked", ("NW",)))
    farm = {
        "money": over.pop("money", 3000.0),
        "tiles": tiles,
        "farmer": over.pop("farmer", [4, 4]),
        "hands": over.pop("hands", [[4, 4], [5, 4], [4, 5]]),
        "unlocked_quadrants": over.pop("quads", ["NW"]),
        "hires_today": over.pop("hires_today", 3),
    }
    opp = copy.deepcopy(farm)
    opp["tiles"] = over.pop("opp_tiles", None) or blank_tiles()
    obs = {
        "player": 0,
        "step": over.pop("step", 0),
        "day": over.pop("day", 0),
        "hour": over.pop("hour", 4),
        "farms": [farm, opp],
        "private": {
            "shed": over.pop("shed", {"WHEAT": 0, "FERTILIZER": 0}),
            "seeds": over.pop("seeds", {"WHEAT": 5, "STRAWBERRY": 2,
                                        "MELON": 2, "TOMATO": 2, "CARROT": 0}),
            "inventories": over.pop("inventories", [{}, {}, {}, {}]),
        },
        "market": {
            "prices": over.pop("prices", {}),
            "inventory": over.pop("inventory", {}),
        },
        "town": {"unlocked_shops": over.pop("shops", ["PIZZA_SHOP", "YARN_STORE"])},
    }
    assert not over, f"unused scene fields: {sorted(over)}"
    return name, obs


def animal_tile(kind, animal, fed=False, cared=False, yield_units=0):
    return {"kind": kind, "animal": animal, "fed_today": fed,
            "cared_today": cared, "yield_units": yield_units,
            "placed_day": 0, "consecutive_unfed": 0,
            "fertilizer_available": True}


def plant_tile(crop, day=0, watered=False, yield_units=1):
    return {"kind": "PLANT", "crop": crop, "planted_day": day,
            "watered_today": watered, "yield_units": yield_units,
            "consecutive_unwatered": 0, "fertilized_until_day": -1,
            "max_lifespan_step": -1}


def scenes():
    """Scenes chosen so that every knob has a turn where it could matter.

    A knob that shows no change here is either inert or untested, and the two
    look identical -- so each scene below exists to make one decision reachable:
    somewhere to site a structure, a herd that needs feeding, an opponent whose
    output the plan defers to, land that could be bought, land standing idle.
    """
    out = [scene("opening")]

    # A farm with animals placed away from the shed and a hand standing on one
    # that is fed but not yet cared for: the case `finish_tile` is meant for.
    tiles = blank_tiles()
    tiles[2][1] = animal_tile("PASTURE", "COW", fed=True, yield_units=2)
    tiles[4][4] = animal_tile("PASTURE", "COW", fed=False)
    tiles[1][3] = plant_tile("STRAWBERRY", day=0, watered=False, yield_units=0)
    tiles[3][3] = plant_tile("MELON", day=0, watered=False, yield_units=0)
    out.append(scene("herd_midgame", tiles=tiles, day=9, hour=8, money=1800.0,
                     farmer=[1, 2], hands=[[4, 4], [3, 1], [0, 0]],
                     shed={"WHEAT": 12, "FERTILIZER": 9, "MILK": 4},
                     hires_today=3))

    # Late season, three quadrants, cash in hand: the siting and selling knobs.
    tiles = blank_tiles(("NW", "NE", "SW"))
    for (x, y) in ((0, 0), (1, 0), (2, 0), (6, 1), (7, 2)):
        tiles[y][x] = plant_tile("STRAWBERRY", day=8, yield_units=3)
    for (x, y) in ((4, 4), (5, 4), (3, 4)):
        tiles[y][x] = animal_tile("PASTURE", "COW", fed=True, yield_units=3)
    out.append(scene("late", tiles=tiles, day=20, hour=6, money=9000.0,
                     quads=["NW", "NE", "SW"],
                     farmer=[5, 5], hands=[[4, 4], [0, 1], [6, 2], [2, 6]],
                     shed={"WHEAT": 30, "FERTILIZER": 24, "MILK": 9,
                           "STRAWBERRY": 6},
                     hires_today=4))

    # Somewhere to put an animal: stock waiting in the shed, empty tiles both
    # beside the shed and out at the rim, and the hands standing at the rim.
    tiles = blank_tiles()
    tiles[0][0] = plant_tile("WHEAT", day=2, yield_units=2)
    out.append(scene("siting", tiles=tiles, day=5, hour=7, money=2200.0,
                     farmer=[0, 1], hands=[[1, 0], [0, 4], [4, 0]],
                     shed={"COW": 2, "SHEEP": 1, "WHEAT": 10},
                     hires_today=3))

    # A herd that has not eaten, with barely any feed in the shed: the shed-run
    # rules and the size of a feed purchase both land here.
    tiles = blank_tiles()
    for (x, y) in ((4, 3), (5, 3), (3, 4), (2, 2), (1, 4)):
        tiles[y][x] = animal_tile("PASTURE", "COW", fed=False)
    out.append(scene("feed_run", tiles=tiles, day=12, hour=3, money=1500.0,
                     farmer=[0, 0], hands=[[1, 1], [4, 4], [2, 5]],
                     shed={"WHEAT": 2, "FERTILIZER": 3},
                     hires_today=3))

    # An opponent already supplying the town: how much of their output the plan
    # subtracts from its own decides what it is willing to grow.
    opp = blank_tiles(("NW", "NE", "SW"))
    for i in range(20):
        opp[i // 5][i % 5] = plant_tile("STRAWBERRY", day=6, yield_units=2)
    for i in range(10):
        opp[5 + i // 5][i % 5] = animal_tile("PASTURE", "COW", fed=True)
    tiles = blank_tiles(("NW", "NE"))
    out.append(scene("rival_supplying", tiles=tiles, opp_tiles=opp, day=11,
                     hour=5, money=4000.0, quads=["NW", "NE"],
                     farmer=[4, 4], hands=[[4, 5], [3, 3]],
                     shed={"WHEAT": 14, "FERTILIZER": 6}, hires_today=2))

    # Land on the table: past the gate day, cash in hand, and the tiles it
    # already owns mostly in use.
    tiles = blank_tiles(("NW", "NE"))
    for i in range(30):
        x, y = (i % 10), (i // 10)
        if y < 5:
            tiles[y][x] = plant_tile("STRAWBERRY", day=4, yield_units=2)
    out.append(scene("land_offer", tiles=tiles, day=11, hour=2, money=6000.0,
                     quads=["NW", "NE"], farmer=[4, 4],
                     hands=[[4, 5], [1, 1], [7, 2]],
                     shed={"WHEAT": 20, "FERTILIZER": 5}, hires_today=3))

    # Three quadrants owned and most of them empty, with cash to plant: this is
    # the turn where planting past the town's cap is either taken or refused.
    tiles = blank_tiles(("NW", "NE", "SW"))
    for (x, y) in ((0, 0), (1, 0), (4, 4), (5, 4)):
        tiles[y][x] = plant_tile("STRAWBERRY", day=6, yield_units=2)
    out.append(scene("idle_land", tiles=tiles, day=13, hour=4, money=7000.0,
                     quads=["NW", "NE", "SW"], farmer=[4, 4],
                     hands=[[0, 2], [6, 1], [2, 6]],
                     shed={"WHEAT": 18, "FERTILIZER": 8}, hires_today=3))
    return out


DEFAULT_KNOBS = [
    {"build_shed_weight": 1.0},
    {"build_shed_weight": 3.0},
    {"build_shed_weight": 8.0},
    {"plant_shed_weight": 0.3},
    {"finish_tile": 3.0},
    {"pickup_min": 4, "pickup_topup": False},
    {"fert_cap": 0},
    {"animal_first": ["WOOL", "MILK", "EGG"]},
    {"feed_buy_days": 3},
    {"seed_priority": ["MELON"]},
    {"wheat_floor_early": 0},
    {"fill_idle": True},
    {"rival_supply": 0.0},
    {"max_quadrants": 2},
]


def run(mod, obs):
    return json.dumps(mod.agent(copy.deepcopy(obs)), sort_keys=True)


def main():
    knobs = [json.loads(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_KNOBS
    mod = load_agent()
    base_p = copy.deepcopy(mod.P)
    cases = scenes()

    baseline = {}
    for name, obs in cases:
        mod._MEM.clear()
        baseline[name] = run(mod, obs)

    print(f"{'knob':<44}{'scenes changed':>16}")
    dead = []
    for knob in knobs:
        mod.P.clear()
        mod.P.update(copy.deepcopy(base_p))
        mod.P.update(knob)
        changed = []
        for name, obs in cases:
            mod._MEM.clear()
            if run(mod, obs) != baseline[name]:
                changed.append(name)
        label = json.dumps(knob, sort_keys=True)
        print(f"{label:<44}{(', '.join(changed) or 'NONE'):>16}")
        if not changed:
            dead.append(label)

    mod.P.clear()
    mod.P.update(base_p)
    if dead:
        print("\nNo scene distinguishes these on a single turn. Before sweeping "
              "one, decide which it is: a decision no scene reaches (build the "
              "scene), a value too small to move anything (try a bigger one), "
              "or a plan-level change that needs a season to show:")
        for label in dead:
            print("  " + label)
    return 0


if __name__ == "__main__":
    sys.exit(main())

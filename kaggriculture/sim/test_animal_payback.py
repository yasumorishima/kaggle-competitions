"""Can the farm refuse a head the calendar orders but the market cannot pay for?

Seed 46000 against the top replay, read off the ledger on 2026-08-27:

    day 13  wool quotes 169   the calendar buys the first sheep
    day 14  wool quotes 172   ...and three more, four head standing
    day 18  wool quotes  55
    day 19  wool quotes   1   the first fleece from those sheep is due here
    season  WOOL sold: 3 units, at a mean quote of 5.7, for $17

The town never opened a yarn store that season, so wool's only buyer was the
town centre at one unit a day, and the rival had four sheep of its own on the
board from day 0. None of that reaches the decision: a calendar entry sets the
head count directly and switches off both the payback test and the feed test,
by design, because whether the herd pays for itself is the question the search
is asking.

The spot payback test would not have caught it either -- at day 13 wool is 169,
so a sheep looks like it returns 169 * 1.33 * 10 producing days against a $600
bar. Only pricing the fleece on the day it is actually cut refuses it.

These are single planning turns; no episode is played.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import knob_bite as kb  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def bought(actions, species):
    if isinstance(actions, str):
        actions = json.loads(actions)
    for o in actions.get("market", []):
        if o[0] == "BUY_ANIMAL" and o[1] == species:
            return o[2]
    return 0


def day13_scene(mod, rival_sheep=4, wool_inv=10030):
    """Our farm on day 13 with a calendar that wants four sheep."""
    tiles = kb.blank_tiles(("NW", "NE"))
    for (x, y) in ((0, 0), (1, 0), (2, 0), (3, 0)):
        tiles[y][x] = kb.animal_tile("PASTURE", "COW", fed=True, yield_units=1)
    for (x, y) in ((0, 2), (1, 2), (2, 2)):
        tiles[y][x] = kb.plant_tile("STRAWBERRY", day=8, yield_units=2)
    opp = kb.blank_tiles(("NW", "NE"))
    for i in range(rival_sheep):
        opp[4][i] = kb.animal_tile("PASTURE", "SHEEP", fed=True, yield_units=2)
    return kb.scene("day13", tiles=tiles, opp_tiles=opp, day=13, hour=4,
                    step=13 * 24 + 4, money=9000.0, quads=["NW", "NE"],
                    shed={"WHEAT": 30, "FERTILIZER": 4},
                    inventory={"WOOL": wool_inv},
                    # The season that was measured: no yarn store ever opened,
                    # so wool's only buyer is the town centre.
                    shops=["SMOOTHIE_SHOP", "ICE_CREAM_SHOP", "PET_CAFE",
                           "BRUNCH_SPOT", "PIZZA_SHOP"])


def main():
    mod = kb.load_agent()
    saved = dict(mod.P)
    try:
        mod.SCHEDULE = {"13": {"SHEEP": 4, "COW": 4}}
        _n, obs = day13_scene(mod)
        spot = mod.price_at("WOOL", 10030)
        check("the scene starts with wool still looking healthy",
              spot > 0.7 * mod.MARKET_PARAMS["WOOL"]["base"],
              f"quote {spot} against base {mod.MARKET_PARAMS['WOOL']['base']}")

        # Today's behaviour, and the reason this file exists.
        base_buy = bought(kb.run(mod, obs), "SHEEP")
        check("as it stands the calendar's sheep are bought", base_buy > 0,
              f"{base_buy} head")

        # The spot payback test cannot refuse them even when it is allowed to.
        mod.P["sched_veto"] = True
        mod.P["animal_payback_rule"] = "spot"
        spot_buy = bought(kb.run(mod, obs), "SHEEP")
        check("the spot test lets them through anyway", spot_buy == base_buy,
              f"{spot_buy} vs {base_buy}")

        # Priced at the day the fleece is actually cut, they are refused.
        mod.P["animal_payback_rule"] = "forward"
        fwd_buy = bought(kb.run(mod, obs), "SHEEP")
        check("the forward test refuses them", fwd_buy == 0, f"{fwd_buy} head")

        # ...and the veto is what makes that reachable: with it off, a calendar
        # entry still overrules the test, which is the search's contract.
        mod.P["sched_veto"] = False
        check("without the veto the calendar still wins",
              bought(kb.run(mod, obs), "SHEEP") == base_buy)

        # The rule must not simply hate sheep. Same day, same calendar, but the
        # town has a yarn store and no rival flock: these head are bought.
        mod.P["sched_veto"] = True
        mod.P["animal_payback_rule"] = "forward"
        _n2, good = day13_scene(mod, rival_sheep=0, wool_inv=9950)
        good["town"]["unlocked_shops"] = ["YARN_STORE", "PIZZA_SHOP", "PET_CAFE"]
        check("a healthy wool market still gets its sheep",
              bought(kb.run(mod, good), "SHEEP") > 0)

        # And a cow, whose milk nobody is flooding, is untouched in both scenes.
        cows_off, cows_on = None, None
        mod.P["sched_veto"] = False
        cows_off = bought(kb.run(mod, obs), "COW")
        mod.P["sched_veto"] = True
        cows_on = bought(kb.run(mod, obs), "COW")
        check("the cows are unaffected", cows_off == cows_on,
              f"{cows_off} vs {cows_on}")

        # The herd cap asks a different question from the payback test: not
        # "will this head pay for itself" but "does the town want what it
        # makes". `plan` already sizes the herd to `market_cap`; a calendar
        # entry replaces that number, and this puts the ceiling back without
        # touching the timing. Both tests are off here so only the cap acts.
        mod.P["sched_veto"] = False
        mod.P["animal_payback_rule"] = "spot"
        mod.P["sched_herd_cap"] = 0
        # Ask for two head the farm does not have yet, or the cow arm of this
        # check cannot fail: the scene already stands four cows against a
        # calendar that wants four.
        mod.SCHEDULE = {"13": {"SHEEP": 4, "COW": 6}}
        ref_sheep = bought(kb.run(mod, obs), "SHEEP")
        ref_cows = bought(kb.run(mod, obs), "COW")
        mod.P["sched_herd_cap"] = True
        cap_sheep = bought(kb.run(mod, obs), "SHEEP")
        cap_cows = bought(kb.run(mod, obs), "COW")
        check("the cap refuses sheep in a town with no yarn store",
              ref_sheep > 0 and cap_sheep == 0, f"{ref_sheep} -> {cap_sheep}")
        check("the cap leaves the cows alone, three shops want their milk",
              cap_cows == ref_cows, f"{ref_cows} -> {cap_cows}")

        # ...and it must not simply hate sheep either.
        _n3, yarn = day13_scene(mod, rival_sheep=0, wool_inv=9950)
        yarn["town"]["unlocked_shops"] = ["YARN_STORE", "PIZZA_SHOP", "PET_CAFE"]
        check("a town with a yarn store still gets its sheep under the cap",
              bought(kb.run(mod, yarn), "SHEEP") > 0)
        mod.P["sched_herd_cap"] = 0
    finally:
        mod.P.clear()
        mod.P.update(saved)

    print("\n" + ("FAILED: " + ", ".join(FAILED) if FAILED
                  else "all animal-payback checks pass"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

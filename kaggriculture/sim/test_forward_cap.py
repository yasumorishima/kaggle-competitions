"""What the per-item tile ceiling does when a rival is flooding the town.

The 146 leaderboard episodes pulled on 2026-08-27 say the whole grade turns on
this regime: against farms above 90k we are 0 for 26, and our own money falls
from 84k to 57k the moment a strong farm is on the board. The ceiling rule is
the code that decides what we do about that, and until now it could only be
watched through a season's final money.

"demand" subtracts the rival's standing supply from our allowance one for one
and reads its stock allowance off the curve's neutral point -- so it is blind
to what is actually in the town. That blindness is the thing to pin: it gives
the same answer whether the quote is $120 or $1.

"forward" measures the stock allowance from where the market will be when the
tile pays. Both rules net the rival off the daily drain; the first draft of
"forward" did not, and this scene caught it planting 54 tiles into a market
whose quote was already the floor.

No episode is played: these are single planning turns.
"""
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


def flood(mod, rival_tiles, inv, crop="STRAWBERRY", day=4):
    """One planning turn with `rival_tiles` of `crop` standing on their farm."""
    opp = kb.blank_tiles(("NW", "NE"))
    n = 0
    for y in range(10):
        for x in range(10):
            if opp[y][x] is None and n < rival_tiles:
                opp[y][x] = kb.plant_tile(crop, day=4, yield_units=2)
                n += 1
    assert n == rival_tiles, f"only placed {n} of {rival_tiles}"
    _name, obs = kb.scene("flood", opp_tiles=opp, inventory={crop: inv},
                          day=day, hour=4, money=4000.0, step=day * 24 + 4)
    kb.run(mod, obs)
    return dict(mod.PLAN_TRACE["caps"]), dict(mod.PLAN_TRACE["target"])


def main():
    mod = kb.load_agent()
    saved = dict(mod.P)
    try:
        # The scene is only worth anything if the crop really does collapse.
        quotes = [mod.price_at("STRAWBERRY", i) for i in (10000, 10040, 10080)]
        check("the flood scene really crashes the quote",
              quotes[0] > 100 and quotes[-1] <= 2, f"{quotes}")

        mod.P["cap_rule"] = "demand"
        d_empty, _ = flood(mod, 0, 10000)
        d_rival, _ = flood(mod, 37, 10000)
        d_crash, _ = flood(mod, 37, 10080)
        check("demand cedes when the rival stands up supply",
              d_rival["STRAWBERRY"] < d_empty["STRAWBERRY"],
              f"{d_empty['STRAWBERRY']} -> {d_rival['STRAWBERRY']}")
        check("demand is blind to what is in the town",
              d_crash["STRAWBERRY"] == d_rival["STRAWBERRY"],
              f"quote 1 gives {d_crash['STRAWBERRY']}, quote 120 gives {d_rival['STRAWBERRY']}")

        mod.P["cap_rule"] = "forward"
        f_empty, _ = flood(mod, 0, 10000)
        f_rival, _ = flood(mod, 37, 10000)
        f_mid, _ = flood(mod, 37, 10040)
        f_crash, _ = flood(mod, 37, 10080)
        check("forward also cedes to a rival's supply",
              f_rival["STRAWBERRY"] < f_empty["STRAWBERRY"],
              f"{f_empty['STRAWBERRY']} -> {f_rival['STRAWBERRY']}")
        check("forward reads the town and tightens as it fills",
              f_empty["STRAWBERRY"] > f_rival["STRAWBERRY"] > f_mid["STRAWBERRY"]
              > f_crash["STRAWBERRY"],
              f"{f_empty['STRAWBERRY']} {f_rival['STRAWBERRY']} "
              f"{f_mid['STRAWBERRY']} {f_crash['STRAWBERRY']}")
        # The bug this file was written for: crediting the whole daily drain
        # let the farm plant into a floor-priced market.
        check("forward does not out-plant demand into a crashed market",
              f_crash["STRAWBERRY"] <= d_crash["STRAWBERRY"],
              f"forward {f_crash['STRAWBERRY']} vs demand {d_crash['STRAWBERRY']}")

        mod.P["forward_rival"] = 0.0
        f_ignore, _ = flood(mod, 37, 10000)
        check("forward_rival 0 stops crediting their farm at all",
              f_ignore["STRAWBERRY"] > f_rival["STRAWBERRY"],
              f"{f_ignore['STRAWBERRY']} vs {f_rival['STRAWBERRY']}")
        mod.P["forward_rival"] = 1.0

        # The reason this matters for the leaderboard: the goods the rival is
        # not growing keep a ceiling far above the constant we hold ourselves
        # to, under either rule. boatlee sells no tomato and no egg all season.
        check("the town would take far more tomato than tomato_cap allows",
              f_rival["TOMATO"] > 3 * saved["tomato_cap"],
              f"market {f_rival['TOMATO']} vs our cap {saved['tomato_cap']}")
        check("...and the rival's strawberry flood does not touch that ceiling",
              f_rival["TOMATO"] == f_empty["TOMATO"],
              f"{f_rival['TOMATO']} vs {f_empty['TOMATO']}")
    finally:
        mod.P.clear()
        mod.P.update(saved)

    print("\n" + ("FAILED: " + ", ".join(FAILED) if FAILED else "all forward-cap checks pass"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

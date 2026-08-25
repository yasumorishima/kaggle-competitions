#!/usr/bin/env python
"""Does the pace fallback obey a price floor, and does anything else change?

The reserve was never a floor. `sellable_qty` stops at
`max(base * reserve_frac, now * slice_frac)`, but the line under it is
`qty = max(qty, pace)`, which looks at no price at all -- so the farm sells
into any crater the pace fallback can reach. Measured on seed 3000 against the
top published plan it put five units of wool on the wire at a mean quote of
3.8, against a base of 200 and an environment price floor of 1.

`knob_bite` says NONE for `pace_floor_frac` because none of its scenes has a
crashed market; that is the "no scene reaches it" reading of NONE, not the
"does nothing" one. This builds the scene.
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


def sell_of(actions, item):
    if isinstance(actions, str):
        actions = json.loads(actions)
    for o in actions.get("market", []):
        if o[0] == "SELL" and o[1] == item:
            return o[2]
    return 0


def crashed_market_scene(**over):
    """Wool at the bottom: the town is stuffed, and the shed holds fleece.

    MARKET_I0 is the neutral inventory; far above it the quote collapses, which
    is the only state in which the pace fallback and the reserve disagree.
    """
    over.setdefault("shed", {"WOOL": 12, "WHEAT": 0, "FERTILIZER": 0})
    return kb.scene(
        "wool crater",
        inventory={"WOOL": 10250},
        day=18, hour=20, step=18 * 24 + 20,
        **over)


def main():
    mod = kb.load_agent()
    base = mod.MARKET_PARAMS["WOOL"]["base"]
    _n, obs = crashed_market_scene()

    quote = mod.price_at("WOOL", obs["market"]["inventory"]["WOOL"])
    check("the scene really is a crater",
          quote < 0.1 * base, f"quote {quote} against base {base}")

    off = kb.run(mod, obs)
    sold_off = sell_of(off, "WOOL")
    check("with the floor off the farm sells into the crater",
          sold_off > 0, f"{sold_off} units at {quote}")

    saved = dict(mod.P)
    try:
        mod.P["pace_floor_frac"] = 0.5
        on = kb.run(mod, obs)
        sold_on = sell_of(on, "WOOL")
        check("with the floor on it does not",
              sold_on == 0, f"{sold_on} units")

        # The floor must not become a trap. The shed discards what it cannot
        # hold at nightfall, so an overflowing shed still dumps.
        _n2, full = crashed_market_scene(
            shed={"WOOL": 60, "WHEAT": 30, "MILK": 20, "FERTILIZER": 0})
        check("an overflowing shed still dumps",
              sell_of(kb.run(mod, full), "WOOL") > 0)

        # And a good whose price is healthy is untouched by any of this.
        _n3, ok = kb.scene("healthy wheat", shed={"WHEAT": 20, "FERTILIZER": 0},
                           inventory={"WHEAT": 10000}, day=18, hour=20,
                           step=18 * 24 + 20)
        mod.P["pace_floor_frac"] = 0.0
        a = sell_of(kb.run(mod, ok), "WHEAT")
        mod.P["pace_floor_frac"] = 0.5
        b = sell_of(kb.run(mod, ok), "WHEAT")
        check("a healthy price is unaffected", a == b, f"{a} vs {b}")
    finally:
        mod.P.clear()
        mod.P.update(saved)

    check("default is off, so every earlier measurement still stands",
          saved["pace_floor_frac"] == 0.0)

    print()
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("all pace-floor checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

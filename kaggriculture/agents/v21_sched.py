"""Kaggriculture agent.

The economics, measured rather than assumed (sim/analyze.py over a full
episode of the top public agent):

* The town drains the market far faster than one farm can refill it, so the
  live regime is *scarcity*, not glut. In a measured season strawberry sold at
  a mean $222 (base $120) and milk at $290 (base $160) while their prices rose
  all season. Only genuinely oversupplied goods crash -- wool ended at $5.
* So the quantity that matters is the town's drain rate per product, and it is
  exactly computable: every unlocked shop instance pulls 1 unit of each product
  it wants every 4 turns (single-product shops pull 2x), i.e. 6 or 12 per day,
  plus 1 per product per day from the town center.
* Tomato ended a measured season at $216 because none of the published agents
  plant it. Whatever the town wants and nobody supplies is where the money is.
* Care is not a nicety: a cared-for cow yields 3 milk per 2 days instead of 1,
  which at scarcity prices is the single most valuable action on the farm.

The agent therefore sizes production against `town_demand - opponent_supply`,
ranks tiles by revenue per tile-day at the live price, and sells in slices that
never push a price under its own reserve.
"""

import math

# ---------------------------------------------------------------- constants
# Mirrors kaggriculture.py; kept local so the agent has no import dependency
# on the environment internals.
CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",  "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",  "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",  "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear","below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
TURNS_PER_DAY = 24
LAND_PRICES = [1000, 2000, 4000]

# Per-episode scratch: the module stays loaded across the 720 turns, so unit
# assignments can persist between calls.
_MEM = {}

# A capital schedule, or None for the policy's own judgement. Set by
# sim/sched_agent.py and left alone everywhere else.
#
# Recording this agent and hill-climbing the resulting 720-step list did not
# transfer: the same list earned 60k on the seeds it was climbed on and 25k on
# held-out ones, while a published list scored 156k and 171k on the same two
# sets. A recording of a reactive policy is welded to its own season, because
# most of what it stores is which tile happened to need water at hour 9.
#
# What is not welded to the season is the capital: how many hands to carry,
# how big a herd, when to buy the second quadrant. Those are the whole measured
# difference between the two farms -- 12 animals against 27, land on days 6 and
# 10 against day 8 alone, 1,010 waterings against 595 -- and they are decisions
# a calendar can hold. So the calendar is what gets searched, and the policy
# keeps the field work it is already good at.
#
# Format: {"<day>": {"hands": n, "COW": n, "SHEEP": n, "GOOSE": n, "land": n}}.
# Entries are cumulative targets that hold until the next entry, and any key
# left out falls through to the policy. A target is a target, not an order: the
# farm still has to afford it and still has to have somewhere to put it.
# The crops the calendar may dial. Livestock is not here: the herd is set by
# head count above, and MILK/WOOL/EGG targets are how the farm sizes pasture,
# not a crop decision.
CROP_KEYS = ("WHEAT", "STRAWBERRY", "MELON", "TOMATO", "CARROT")
SPECIES_ORDER = ("COW", "SHEEP", "GOOSE")

# The labour side of the calendar: how much a day's hands should want each
# kind of work, as a percentage of what the policy already wants.
#
# Why this exists. The capital calendar answers "what should the farm own",
# and searching it transferred -- seventeen generations took both the training
# and the held-out seeds to 12 wins out of 12, where climbing the 720-step
# action list had moved the held-out count not at all in 137. But the capital
# family is now swept out: herd, land, hands and the crop dial were all
# measured against the plan near the top of the ladder and came back tie or
# worse, and the farm still turns roughly the same capital into 58k where that
# plan turns it into 129k. What is left is not what the farm owns, it is what
# its hands spend the day doing.
#
# Every candidate job in jobs_for is already scored `value / (1 + dist*d)`,
# with `value` derived once and for all from price and units. That is a single
# rule for a thirty-day game whose first week (clear, sow, build) and last week
# (harvest, sell) have nothing in common, and no global knob can say so: a
# sweep can only pick one number for the whole season. A per-day multiplier
# can, and it is season-independent in exactly the way capital is -- "on day 3
# care matters more than digging" is as true of one season as the next, while
# "water the tile at (3,7)" is true of exactly one.
#
# 100 is today's behaviour and an absent key is the same thing, the same
# discipline the crop dial uses. Ops arrive either as a bare string or as a
# tuple whose head names the family, so PLANT/PICKUP/PLACE are keyed by head.
TASK_KEYS = ("WATER", "HARVEST", "CARE", "FEED", "FERTILIZE",
             "COLLECT_FERTILIZER", "PLANT", "DIG", "PLACE", "PICKUP",
             "DROP", "BUILD_COOP", "BUILD_PASTURE")
TASK_SUFFIX = "_w"

SCHEDULE = None


def _sched_for(day):
    """The schedule entry in force on `day`, or None."""
    if not SCHEDULE:
        return None
    live = None
    for key in sorted(SCHEDULE, key=lambda k: int(k)):
        if int(key) <= day:
            live = SCHEDULE[key]
    return live


def _task_weights(day):
    """The day's labour multipliers, or None when the calendar is silent.

    Returned as plain floats keyed by op family so the hot loop does a dict
    lookup and a multiply, not a string parse per candidate per hand per turn.
    """
    sched = _sched_for(day)
    if not sched:
        return None
    out = {}
    for name in TASK_KEYS:
        mult = sched.get(name + TASK_SUFFIX)
        if mult is None or int(mult) == 100:
            continue
        out[name] = int(mult) / 100.0
    return out or None


def _struct_target(day):
    """How many pastures and coops the calendar wants standing on `day`.

    Empty dict when the calendar is silent about structures, which is the
    condition the build-ahead path is gated on -- so a calendar written before
    this column existed leaves the farm behaving exactly as it did.
    """
    sched = _sched_for(day)
    if not sched:
        return {}
    out = {}
    for name in ("PASTURE", "COOP"):
        want = sched.get(name)
        if want is not None and int(want) > 0:
            out[name] = int(want)
    return out


def _op_family(op):
    """WATER -> WATER, ("PLANT", "MELON") -> PLANT."""
    return op[0] if isinstance(op, tuple) else op


def _weigh(out, weights):
    """Scale a candidate list by the day's labour multipliers."""
    if not weights:
        return out
    return [(value * weights.get(_op_family(op), 1.0), tile, op)
            for value, tile, op in out]


def _species(tile):
    """Which animal stands on a tile, tolerating either shape of the field."""
    a = tile.get("animal")
    if isinstance(a, dict):
        a = a.get("kind") or a.get("type") or a.get("species")
    return str(a) if a else ""

# Units per tile per day. Crops carry their *measured* rate, which fertilizer
# lifts above the unfertilized table value; livestock keeps the table rate even
# though real episodes deliver 0.71-0.76 milk against the table's 1.50. That
# asymmetry was swept head-to-head, 20 games a side: crops measured + livestock
# tabular scored 28,170 against 18,720 for all-tabular. The herd is sized as
# demand/rate, so believing the low real rate buys more cattle, and every coin
# in cattle is a coin the crops never see.
RATE = {"WHEAT": 0.80, "CARROT": 0.75, "TOMATO": 0.50, "STRAWBERRY": 0.35,
        "MELON": 0.50, "EGG": 2.00, "MILK": 1.50, "WOOL": 1.33}
PRODUCER = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}

# Beyond the town's drain rate there is a one-off stock allowance: the units a
# farm can pour into the market before the price sags under three quarters of
# base. It is read straight off the curve instead of guessed, and it is why the
# strong opening is sheep and melon -- wool clears ~29 units and melon ~79 at
# healthy prices even with no shop demanding them.
ALLOW_FRAC = 0.75


def _stock_allowance(item):
    base = MARKET_PARAMS[item]["base"]
    x = 0
    while x < 600 and price_at(item, MARKET_I0 + x + 1) >= base * ALLOW_FRAC:
        x += 1
    return x

# Precomputed once: item -> units sellable above ALLOW_FRAC of base.
ALLOW = {}


P = {
    "max_hands": 12,
    "hands_early": 4,
    "hands_min": 3,
    "jobs_per_hand": 7,   # a hand clears roughly this many jobs in a 24-turn day
    "wheat_floor": 16,      # feed tiles kept even before the herd exists (swept)
    "cow_cap": 12,
    "sheep_cap": 5,
    "goose_cap": 6,
    "tomato_cap": 6,
    "carrot_cap": 8,
    "melon_cap": 8,
    "animal_buy_last_day": 22,
    "animal_grace_day": 0,     # days on which the feed test is waived (0 = off; measured inert)
    "herd_cap": 0,             # 0 = no ceiling; else max head the farm will own
    "animal_order": "fixed",   # "fixed" = MILK,EGG,WOOL; "roi" = payback per coin
    "plant_last_day": {"WHEAT": 26, "CARROT": 27, "MELON": 17,
                       "TOMATO": 21, "STRAWBERRY": 19},
    "cash_buffer": 120,
    "reserve_frac": 1.00,      # never sell under base: scarcity lifts prices all season
    "slice_frac": 0.92,        # ...nor push the live price below this of itself
    "dump_day": 29,
    # (earliest day, cash floor) per quadrant. Land is what caps the whole farm,
    # so the gates are early and the agent saves toward the next one instead of
    # spending its last coin on livestock.
    "land_gate": [(2, 1200), (6, 2600), (10, 5200)],
    "tile_margin": 1.15,       # plan slightly past the tiles we own
    # A measured season (seed 2000 against boatlee V16-RC5) ended with 42 of
    # 100 tiles empty or under weed while the opponent worked all 75 of its
    # own -- and outsold this farm on strawberry, the very product the cap had
    # closed. The five knobs below expose that finding as switchable
    # mechanism. Measured against boatlee V16-RC5 on seeds 3000+, paired
    # within each season draw, the three adopted below are worth +17,928
    # together against v15. The rest were tested and are NOT adopted:
    # planting past the town-demand cap onto idle land (-3,648, CI +/-1,896),
    # reserving opening cash for the herd (-7,557), suspending the land fund
    # on days 0-1 (-3,399), a lower standing wheat floor (-3,833), and
    # ignoring the opponent's supply entirely (-3,898). They stay switchable
    # so the next round does not have to re-derive that they were tried.
    "max_quadrants": 3,        # quadrants the farm is allowed to own (swept: +6,233 over four)
    "fert_cap": 8,             # fertilizer held back from market for the field (swept: +1,377)
    "fert_span": 3,            # ...spread over this many days of application
    "opening_days": 1,         # days on which the herd outranks the seed line
    "opening_animal_reserve": 0,   # cash the seed line may not touch until then
    "land_save_from_day": -1,  # save toward the next quadrant only after this day
    "wheat_floor_early": 6,    # feed tiles planned during the opening days (swept: +11,695)
    "fill_idle": False,        # plant past the town-demand cap onto idle land
    "fill_floor": 0.75,        # ...while the product still clears this of base
    "fill_cash_floor": 800,    # ...and only with this much cash in hand
    # How much of the opponent's nameplate output is treated as already
    # feeding the town. At 1.0 the farm defers a tile for every tile they own,
    # which is what held strawberry to 21 tiles in the measured season while
    # they worked 36 -- and the mean price stayed at $239 against a $120 base,
    # so the town was never actually filled. Their tiles also count at full
    # rate from the day they are sown, though a strawberry yields nothing for
    # its first ten days, so the subtraction runs high on top of that.
    "rival_supply": 1.0,
    # Fraction of the calendar's outstanding herd cost held back from the
    # seed queue. 0 = off, which is what every earlier measurement was taken
    # under.
    "sched_reserve": 0.0,
    # Read straight off the top public agent's own plan, which ships as a
    # decoded 720-step action list. Its labour goes CARE 967 / FEED 290 /
    # PICKUP 135 against this farm's 169 / 122 / 408, and its opening is four
    # sheep on day 0 with the feed *bought* rather than grown. Defaults below
    # reproduce v16; each is switchable so the sweep decides.
    "animal_first": "",        # "" = P["animal_order"]; else a species order
    "animal_first_days": 1,    # ...applied while day <= this
    "feed_buy_days": 1,        # days of rations to hold in the shed
    "seed_priority": (),       # crops moved to the head of the seed queue
    # This farm issued 408 PICKUP actions against the public plan's 135 while
    # feeding 122 times against its 290. The shed rarely holds more than a
    # bushel or two -- wheat is grown thin and sold -- so a hand walks the whole
    # way for one ration, feeds one animal and walks back. Two rules stop that:
    # do not make the trip for less than a useful load, and do not top up a
    # hand that is already carrying.
    "pickup_min": 1,           # bushels that must be in the shed to justify a trip
    "pickup_topup": True,      # may a hand already carrying wheat go back for more
    # Where the labour actually goes. Measured off the top public plan with
    # sim/route_shape.py: it does 2.07 jobs every time it stops and walks 1.99
    # steps between stops, and 58% of its trips are a single step. That is a
    # farm laid out in a block, not a farm whose tiles were placed wherever a
    # hand happened to be standing. Three levers, all inert at their defaults:
    # pull new structures and new crops toward the shed when siting them, and
    # finish the tile a unit is already standing on before walking off it.
    "build_shed_weight": 0.0,  # how hard a new coop/pasture is pulled shedward
    "plant_shed_weight": 0.0,  # ...and a new crop
    "finish_tile": 1.0,        # bonus for a job on the tile the unit is on
    # The wage is fib(n) for the n-th hire of a day, so the roster is cheap
    # until it is suddenly not: three hands cost $4 a day, eight cost $54, and
    # twelve cost $376. Sizing the roster to the work available puts twelve
    # hands on the farm from day 2, which is $3,008 over days 2-9 -- while the
    # top public plan spends $123 across the same stretch by hiring
    # 5,1,2,3,4,3,4,7,6,7 and only reaching fourteen on day 10 once its melons
    # and wool are selling. The difference is seven cows, and it is spent in
    # exactly the days this farm is measured sitting at $2 unable to buy one.
    "hands_cap_by_day": (),    # ((until_day, cap), ...) applied before max_hands
    "hands_cash_floor": 0,     # below this, hire only hands_min
    # The calendar's `land` line is the one place its genome still sits below
    # the published plan: the climbed calendar buys a second quadrant on day 7
    # and stops, while the top plan buys on day 6 and again on day 10 and works
    # three. Land is tiles and tiles are the crop line, so this is capacity,
    # not preference -- but every measurement that rejected more of it was
    # taken against that same top plan, whose own supply is subtracted from the
    # town's appetite before this farm is allowed to plant. Against an opponent
    # of leaderboard-median size the subtraction is far smaller and the extra
    # quadrant has somewhere to sell. A floor rather than a rewrite of the
    # calendar, so the climb keeps ownership of the rest of the line.
    "sched_land_floor": (),    # ((from_day, quadrants), ...) raising the calendar's land
    # Late in the season the good crops stop being plantable -- strawberry by
    # day 19, melon by 17 -- and every tile they vacate stands empty for the
    # rest of the run. Wheat is the only thing that still finishes: sown, it
    # yields on day 2 and tops out on day 4. The top public plan turns its farm
    # over to it, going 24 -> 56 wheat tiles across days 21-27 and selling 309
    # bushels for $11,364 against this farm's 192 for $7,013. Off by default.
    "wheat_fill_from_day": 99,
    # The one difference that measurement actually found. Same物差し on both
    # plans: this farm does 1.33 jobs every time it stops and walks 2.53 steps
    # to the next one; the top public plan does 2.07 and walks 1.99. The farms
    # are laid out alike -- 51% of this one's work happens within two tiles of
    # the shed against 55% of theirs -- so it is not siting. It is that a unit
    # here is drawn to the single best job, and the single best job is usually
    # alone on its tile. An animal is worth three visits a day (feed, care,
    # collect) and a watered plant one, so a tile's pull should be what is
    # waiting there in total, not just its best item.
    "bundle_weight": 0.0,      # weight on the other jobs waiting on a tile
    # A crop past its last sowing day still takes budget in the plan, and
    # strawberry is taken last with `budget` itself as its cap -- so from day 20
    # the whole spare farm is allotted to a crop that can no longer be sown,
    # `pick_crop` skips it for being out of season, and the tiles stay bare.
    # The measured season ends with 42 of 100 tiles empty or under weed. With
    # this on, a crop that cannot be sown is planned only for what is already
    # in the ground, and the budget flows past it.
    "respect_last_sow_day": False,
    "stickiness": 1.6,         # bonus for keeping a hand on the tile it set out for
    "dist_weight": 1.0,        # how steeply travel discounts a job; higher keeps hands local
    "planner": "greedy",       # "greedy" = per-turn pick, "route" = day rounds
    # 1 = a unit gets first refusal on its own tile. Turned on 2026-08-22 after
    # the capital changed under it: at 27 animals this was +662, a tie, and it
    # was shelved. With the calendar holding the herd at twelve it is
    # +16,529 +/- 5,897 on seeds 6000-6007 and +11,683 +/- 6,829 on 7000-7013,
    # both against the top published plan, both paired within the season.
    #
    # It is walking that it buys back. Counted off the two action lists, this
    # farm spends 4,390 actions moving to their 2,855 and picks items up 425
    # times to their 135, while doing *fewer* productive actions from a larger
    # roster -- the hands were crossing the farm past work they could have done
    # standing still.
    #
    # wheat_floor 16->4 measured +12,101 on its own and then vanished once this
    # was on (+653): the same bottleneck reached by two roads, not two gains.
    # Units are served in roster order and each one claims the tile it picked,
    # so the farmer and the low-numbered hands take the best tiles on the whole
    # farm before the hand standing next to one is asked. Measured on this
    # farm's own season, 56% of 5,910 actions are steps, and care reaches 216
    # of 304 animal-days -- the labour to close that gap is the labour spent
    # walking past work. "global" scores every (unit, job) pair first and fills
    # from the best pair down, which is the same information the roster rule
    # already has, read in an order that does not let unit 0 outbid a unit that
    # is already standing there. `stand_first` is a partial patch on the same
    # hole -- it hands a unit its own tile before the roster can take it -- so
    # under "global" the pre-pass still names the tile but no longer locks it.
    # Hands are dismissed every night and respawn at the shed doorway; the
    # farmer keeps where it stood. So a hand that ends the day at the far edge
    # loses nothing by it, while a farmer that ends the day at the shed throws
    # its one piece of persistence away -- it will start tomorrow on ground the
    # hands arrive on for free. Nothing in the agent uses that asymmetry. This
    # tilts the farmer's last few choices of the day toward the tiles the
    # morning's hands would have to walk out to.
    "farmer_far_bias": 0.0,    # 0 = off; else weight on the farmer's distance from the shed
    "farmer_far_turns": 4,     # ...applied only in this many closing turns of the day
    "assign_rule": "roster",   # "roster" = in order, first come first served; "global" = best pair first
    "stand_first": 1,
    "care_repeat": 0,          # 1 = offer CARE again on an animal already cared today
    "drop_load": 6,            # carry this many items before a shed run is worth it
    "drop_urgency": 0.10,      # weight on the value carried when scoring that run
}


# ------------------------------------------------------------------ helpers
def g(o, key, default=None):
    """Observation objects behave like dicts but are sometimes Structs."""
    if o is None:
        return default
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price_at(item, inv):
    """Exact copy of the environment's price curve, so we can plan sales."""
    p = MARKET_PARAMS[item]
    base, T = p["base"], p["T"]
    x = abs(inv - MARKET_I0)
    if inv < MARKET_I0:
        func, target, sign = p["below_func"], p["below_target"], 1.0
    else:
        func, target, sign = p["above_func"], p["above_target"], -1.0
    denom = _shape(func, T, T)
    amp = target * base / denom if denom else 0.0
    return max(PRICE_FLOOR, round(base + sign * amp * _shape(func, x, T)))


def sellable_qty(item, inv, have, reserve):
    """Units sellable before the marginal price drops under `reserve`.

    The market re-prices after every single unit, so this walks the curve
    instead of trusting the quote.
    """
    n, cur = 0, inv
    while n < have:
        if price_at(item, cur) < reserve:
            break
        n += 1
        cur += 1
    return n


ALLOW.update({item: _stock_allowance(item) for item in MARKET_PARAMS})


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(fr, to):
    if fr[0] < to[0]:
        return "EAST"
    if fr[0] > to[0]:
        return "WEST"
    if fr[1] < to[1]:
        return "SOUTH"
    if fr[1] > to[1]:
        return "NORTH"
    return None


def quadrant_of(x, y, n):
    h = n // 2
    return ("NW" if x < h else "NE") if y < h else ("SW" if x < h else "SE")


def shed_tiles(n):
    h = n // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


# Expected drain a *future* shop unlock adds per product per day: shops are
# drawn uniformly with replacement, so one unlock is worth
# (sum over shop types that want the item of 6, doubled for single-product) / 8.
PER_UNLOCK = {}
for _name, _items in SHOPS.items():
    for _item in _items:
        PER_UNLOCK[_item] = PER_UNLOCK.get(_item, 0.0) + (12.0 if len(_items) == 1 else 6.0) / 8.0

# How much of that expectation to act on. A crop takes 8-16 days to pay, so it
# has to be sown against the town the season will have, not the one it has
# today. An animal can be bought the day its shop appears, so it barely needs
# to speculate -- and sheep are the most expensive bet on the board.
LOOKAHEAD = {"WHEAT": 1.0, "CARROT": 1.0, "TOMATO": 1.0, "STRAWBERRY": 1.0,
             "MELON": 1.0, "MILK": 0.5, "EGG": 0.5, "WOOL": 0.25}


def town_demand(shops, day):
    """Units per day the town will remove from the market, per product.

    Each unlocked instance pulls one unit of each product it wants every 4
    turns (24 turns/day -> 6/day); a single-product shop pulls double. The town
    centre takes one of every non-fertilizer product per day on top. Shops keep
    unlocking every 3 days up to 8 instances, so the slots not yet drawn are
    added at their expectation -- otherwise day-0 demand reads as 1/day and
    nothing slow ever gets planted in time.
    """
    d = {item: 1.0 for item in MARKET_PARAMS if item != "FERTILIZER"}
    d["FERTILIZER"] = 0.0
    have = 0
    for name in shops or []:
        items = SHOPS.get(name)
        if not items:
            continue
        have += 1
        mult = 2 if len(items) == 1 else 1
        for item in items:
            d[item] = d.get(item, 0.0) + 6.0 * mult
    # Unlocks land on days 3, 6, ... until 8 instances exist.
    to_come = max(0, min(8, max(0, (29 - day)) // 3 + 1) - 0)
    to_come = min(to_come, 8 - have)
    for item, per in PER_UNLOCK.items():
        d[item] = d.get(item, 0.0) + to_come * per * LOOKAHEAD.get(item, 1.0)
    return d


def extra_from_fertilizer(tile, cd, day):
    """Extra units one FERTILIZE buys on this plant (it lasts day..day+2).

    Ongoing crops double a scheduled production from 1 to 2 on days they are
    also watered; one-time crops get +2 instead of +1 per watered day inside
    their bonus window. Either way the bonus is capped by the plant's max yield
    and by how much of its life is left.
    """
    age = day - tile["planted_day"]
    if cd["ongoing"]:
        interval = max(1, cd["interval"])
        # Productions that fall inside the 3-day window.
        window = sum(1 for k in range(3)
                     if (age + k - cd["first_yield_day"]) >= 0
                     and (age + k - cd["first_yield_day"]) % interval == 0)
        fired = max(0, (age - cd["first_yield_day"]) // interval + 1) if age >= cd["first_yield_day"] else 0
        left = max(0, cd["max_yield"] - fired)
        return min(window, left)
    lo = (cd["max_yield_day"] + 1) // 2
    days = [a for a in range(age, age + 3) if lo <= a <= cd["max_yield_day"]]
    headroom = max(0, cd["max_yield"] - tile.get("yield_units", 0))
    return min(len(days), headroom)


def census(tiles):
    """Count producing tiles of a farm by the product they yield."""
    out = {}
    for row in tiles or []:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                out[t["crop"]] = out.get(t["crop"], 0) + 1
            elif "animal" in t:
                prod = ANIMALS[t["animal"]]["product"]
                out[prod] = out.get(prod, 0) + 1
    return out


# --------------------------------------------------------------- the agent
def agent(obs, config=None):
    me = g(obs, "player", 0)
    day = int(g(obs, "day", 0))
    hour = int(g(obs, "hour", 0))
    farms = g(obs, "farms", []) or []
    if not farms or me >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[me]
    opp = farms[1 - me] if len(farms) > 1 else None
    private = g(obs, "private", {}) or {}
    market = g(obs, "market", {}) or {}
    prices = dict(g(market, "prices", {}) or {})
    inventory = dict(g(market, "inventory", {}) or {})
    town = g(obs, "town", {}) or {}
    shops = list(g(town, "unlocked_shops", []) or [])
    shed = dict(g(private, "shed", {}) or {})
    seeds = dict(g(private, "seeds", {}) or {})
    invs = g(private, "inventories", []) or []
    tiles = g(farm, "tiles", []) or []
    n = len(tiles)
    money = float(g(farm, "money", 0))
    unlocked = list(g(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    hires_today = int(g(farm, "hires_today", 0))
    hands = [tuple(h) for h in (g(farm, "hands", []) or [])]
    units = [tuple(g(farm, "farmer", [0, 0]))] + hands
    last_day = 29
    days_left = max(1, last_day - day)

    def price(item):
        inv = inventory.get(item)
        return price_at(item, int(inv)) if inv is not None else MARKET_PARAMS[item]["base"]

    # ---------------------------------------------------------- farm census
    animals, plants, weeds, empty_struct, empty_tiles = [], [], [], [], []
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t is None:
                empty_tiles.append((x, y))
                continue
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "PLANT":
                plants.append((x, y, t))
            elif kind == "WEED":
                weeds.append((x, y, t))
            elif kind in ("COOP", "PASTURE"):
                if "animal" in t:
                    animals.append((x, y, t))
                else:
                    empty_struct.append((x, y, t))

    mine = census(tiles)
    theirs = census(g(opp, "tiles", []) if opp else [])
    herd = len(animals)
    shed_animals = sum(shed.get(a, 0) for a in ANIMALS)
    carried_animals = sum(sum(iv.get(a, 0) for a in ANIMALS)
                          for iv in invs if isinstance(iv, dict))
    tiles_owned = 25 * len(unlocked)
    tiles_used = len(plants) + len(animals) + len(empty_struct)

    # ------------------------------------------------- production plan
    # Size each product against what the town removes and the opponent already
    # supplies, then spend the tiles we own on whatever pays most per day.
    demand = town_demand(shops, day)
    pending_animals = shed_animals + carried_animals

    def market_cap(item):
        """Tiles whose output the market can absorb without the price sagging.

        Two parts: what the town takes every day, and the one-off stock the
        curve will swallow, spread over the days that are left. The opponent's
        own tiles are subtracted -- their supply eats the same allowance.
        """
        room = (demand.get(item, 1.0)
                + ALLOW[item] / max(6.0, float(days_left))
                - theirs.get(item, 0) * RATE[item] * P["rival_supply"])
        return max(0, int(room / RATE[item]))

    # An explicit build, in the order the measured economy pays for it. A
    # ranking over live prices was tried first and proved unstable: livestock
    # outbids every crop per tile, so it ate the whole budget and the farm
    # starved. Feed and land are what actually gate the season.
    budget = int(tiles_owned * P["tile_margin"])
    target = {}

    def take(item, want):
        nonlocal budget
        if (P["respect_last_sow_day"] and item in P["plant_last_day"]
                and day > P["plant_last_day"][item]):
            want = min(want, mine.get(item, 0))
        want = max(0, min(int(want), budget))
        target[item] = want
        budget -= want

    # Sixteen feed tiles on day 0 is a farm sized for a herd it has not bought
    # yet: measured, the opening put 16 of its 25 tiles and all of their
    # watering into wheat while holding one cow.
    wheat_floor = P["wheat_floor_early"] if day <= P["opening_days"] else P["wheat_floor"]
    take("WHEAT", max(wheat_floor, math.ceil((herd + pending_animals) * 1.3 / RATE["WHEAT"])))
    take("MILK", min(market_cap("MILK"), P["cow_cap"]))
    # Wool pays superbly and crashes hardest: only farm it once a yarn store is
    # actually open (it is the only shop that wants wool, and it wants 12/day).
    take("WOOL", min(market_cap("WOOL"), P["sheep_cap"]))
    take("EGG", min(market_cap("EGG"), P["goose_cap"] if demand.get("EGG", 1) >= 10 else 2))
    # Tomato and carrot sit on hinge curves: when the town drains them and
    # nobody plants them, their price runs away ($216 measured for tomato).
    take("TOMATO", min(market_cap("TOMATO"), P["tomato_cap"]))
    take("CARROT", min(market_cap("CARROT"), P["carrot_cap"]) if "PET_CAFE" in shops else 0)
    take("MELON", min(market_cap("MELON"), P["melon_cap"]))
    take("STRAWBERRY", min(market_cap("STRAWBERRY"), budget))
    # Wheat as the closing crop, not as feed: this is a different question from
    # `fill_idle`, which ranks every crop by revenue per tile-day and late in
    # the season picks whatever is nominally dearest even though it can no
    # longer ripen. Here the only claim is that a tile which cannot grow
    # anything else should grow wheat.
    if (budget > 0 and day >= P["wheat_fill_from_day"]
            and day <= P["plant_last_day"]["WHEAT"]):
        target["WHEAT"] = target.get("WHEAT", 0) + budget
        budget = 0
    # Idle land is not neutral, it is a loss: the season is fixed, weeds seed
    # themselves onto empty tiles, and every tile the hands walk past is one
    # they walked further for. `market_cap` is a ceiling on *supply*, but the
    # price curve is smooth, not a wall -- 400 combined units of strawberry
    # were sold in the measured season and the mean price stayed at $239
    # against a $120 base. So once the plan fits inside the town's drain, the
    # tiles that are still spare go to whatever pays most per tile-day, as
    # long as it clears its floor and the herd's cash is not the thing paying.
    if P["fill_idle"] and budget > 0 and money >= P["fill_cash_floor"]:
        spare = budget
        best = None
        for crop in ("STRAWBERRY", "MELON", "TOMATO", "CARROT", "WHEAT"):
            if day > P["plant_last_day"][crop]:
                continue
            if price(crop) < MARKET_PARAMS[crop]["base"] * P["fill_floor"]:
                continue
            val = price(crop) * RATE[crop]
            if best is None or val > best[1]:
                best = (crop, val)
        if best:
            target[best[0]] = target.get(best[0], 0) + spare
            budget = 0
    # The calendar's crop dial, applied last so it moves the finished plan
    # rather than competing with the budget arithmetic above.
    #
    # Percentages, not tile counts. The measured crop gap is large -- 1,679
    # units sold against 913, strawberry 300 against 58 -- but writing the
    # other farm's tile counts in here would be copying a surface statistic,
    # which has failed three times on this problem and the fourth time turned
    # out to have been measuring something else entirely. A multiplier starts
    # at 100, which is exactly today's behaviour, and lets the search move it
    # while every step is scored.
    sched = _sched_for(day)
    if sched:
        for _crop in CROP_KEYS:
            mult = sched.get(_crop + '_pct')
            if mult is None:
                continue
            base = target.get(_crop, 0)
            if base == 0 and int(mult) > 100:
                base = 1          # a dial above 100 may open a crop the plan skipped
            target[_crop] = max(0, int(round(base * int(mult) / 100.0)))

    # True while any product is still short of its target: the farm then wants
    # every tile it can clear, weeds included.
    want_more_tiles = any(target.get(i, 0) > mine.get(i, 0) for i in target)

    def deficit(item):
        # `mine`/`theirs` are keyed by product, so an animal tile counts under
        # EGG/MILK/WOOL rather than under GOOSE/COW/SHEEP.
        return target.get(item, 0) - mine.get(item, 0)

    # ------------------------------------------------------- market actions
    # Only 10 orders per turn are processed, so they are built in priority
    # groups: hiring must land at dawn or the day's labour is lost, and sales
    # run before the purchases they fund.
    hire_orders, sell_orders, buy_orders = [], [], []
    liquidate = day >= P["dump_day"]

    # 1. Sell in slices. In the scarcity regime the price climbs back between
    #    turns as the town keeps buying, so dumping a whole shed at once is
    #    strictly worse than trickling.
    wheat_keep = 0 if liquidate else herd * min(3, days_left)
    # Fertilizer is held back for the field, not sold: one unit on a tomato is
    # worth roughly three fruit. Only the surplus over what the standing crop
    # can absorb in the days left goes to market.
    fert_targets = sum(1 for _, _, t in plants
                       if CROPS[t["crop"]]["ongoing"] and t.get("fertilized_until_day", -1) < day)
    # Held-back fertilizer only pays if the hands actually apply it. Measured:
    # 153 units collected, 104 applied, and exactly 1 sold all season, while
    # the opponent turned the same by-product into $13,857. The cap is what
    # pins it -- the shed never reaches 40 units, so the surplus never reaches
    # the market and is discarded at the 100-item shed cap instead.
    fert_keep = 0 if liquidate else min(
        P["fert_cap"], int(fert_targets * max(1, days_left) / P["fert_span"]) + fert_targets)
    for item in ("MILK", "STRAWBERRY", "WOOL", "MELON", "TOMATO", "EGG",
                 "CARROT", "FERTILIZER", "WHEAT"):
        have = int(shed.get(item, 0))
        if item == "WHEAT":
            have = max(0, have - wheat_keep)
        elif item == "FERTILIZER":
            have = max(0, have - fert_keep)
        if have <= 0:
            continue
        if liquidate:
            qty = have
        else:
            now = price(item)
            reserve = max(MARKET_PARAMS[item]["base"] * P["reserve_frac"],
                          now * P["slice_frac"])
            qty = sellable_qty(item, int(inventory.get(item, MARKET_I0)), have, reserve)
            # Holding is not free: the shed caps at 100 items and discards the
            # rest at nightfall, and production never stops. So clear the day's
            # stock at a steady pace regardless of the reserve -- measured
            # against the top agent, under-selling cost more than price impact
            # (67 sell orders against their 196, with milk piling up unsold).
            turns_left = max(1, TURNS_PER_DAY - hour)
            pace = -(-have // turns_left)
            qty = max(qty, pace)
            if sum(shed.values()) > 80:
                qty = max(qty, have // 2)
            qty = min(qty, have)
        if qty > 0:
            sell_orders.append(["SELL", item, qty])

    # 2. Hire. A dozen hands cost ~$376 for the day and add 288 actions.
    if hour <= 2 and not liquidate:
        # Spread over three turns: only 10 orders clear per turn, and hiring
        # them all at dawn would crowd out the sales that pay for them. (A cap
        # of 6/turn over two turns silently pinned the roster at 12, which made
        # the max_hands knob unsweepable.)
        # Size the roster to the work, not to the wage. Measured against the top
        # agent this farm idled 834 actions on PASS to their 323: with fewer
        # producing tiles, a fixed dozen hands stand around while the coins they
        # cost are exactly what the third quadrant and the strawberry seed are
        # short of.
        jobs_today = (sum(1 for _, _, t in plants if not t.get("watered_today"))
                      + sum(1 for _, _, t in plants if t.get("yield_units", 0) > 0)
                      + 3 * len(animals)          # feed, care, collect
                      + len(empty_tiles) + len(weeds) + len(empty_struct))
        want = max(P["hands_min"], min(P["max_hands"], -(-jobs_today // P["jobs_per_hand"])))
        if day < 2:
            want = min(want, P["hands_early"])
        for until_day, cap in P["hands_cap_by_day"]:
            if day <= until_day:
                want = max(P["hands_min"], min(want, cap))
                break
        if P["hands_cash_floor"] and money < P["hands_cash_floor"]:
            want = min(want, P["hands_min"])
        sched = _sched_for(day)
        if sched and sched.get('hands') is not None:
            want = max(P['hands_min'], int(sched['hands']))
        room = max(0, want - hires_today)
        for _ in range(min(room, 5)):
            hire_orders.append(["HIRE"])

    # 3. Land, gated on both a day and a cash floor.
    extra = len(unlocked) - 1
    saving_for_land = 0.0
    if extra < min(len(LAND_PRICES), P["max_quadrants"] - 1) and not liquidate:
        sched = _sched_for(day)
        if sched and sched.get('land') is not None:
            # The calendar has decided; the gate is what it replaces.
            want_land = int(sched['land'])
            for from_day, quadrants in P["sched_land_floor"]:
                if day >= from_day:
                    want_land = max(want_land, int(quadrants))
            if len(unlocked) < want_land and money >= LAND_PRICES[extra]:
                buy_orders.append(["BUY_LAND"])
                money -= LAND_PRICES[extra]
            min_day, min_money = 99, 10 ** 9
        else:
            min_day, min_money = P["land_gate"][extra]
        if day >= min_day and money >= min_money and tiles_used > 0.55 * tiles_owned:
            buy_orders.append(["BUY_LAND"])
            money -= LAND_PRICES[extra]
        elif (day >= min_day - 2 and money >= 0.5 * LAND_PRICES[extra]
              and day > P["land_save_from_day"]):
            # Hold back the price of the next quadrant instead of sinking the
            # last coin into livestock -- but only once the quadrant is within
            # reach, or the farm saves itself out of seed money on day 0.
            saving_for_land = min(LAND_PRICES[extra], min_money)
    # Seeds are exempt from the land fund: wheat costs $10 and the herd starves
    # without it, which is how v4 lost every cow by day 9.
    spendable = money - P["cash_buffer"] - saving_for_land
    # The opening $3000 is fought over by the seed line and the herd. A tile of
    # seed pays from its first harvest; an animal pays on every day it lives,
    # so the days bought earliest are the most valuable ones on the farm. The
    # measured opening still held $977 idle on day 1 and reached four cows only
    # on day 7, against an opponent sitting at $7 cash with nine head by day 8.
    seed_budget = money - P["cash_buffer"]
    if day <= P["opening_days"]:
        seed_budget -= P["opening_animal_reserve"]
    # A calendar that cannot be paid for is a wish. Measured on seed 2000:
    # the calendar asked for four sheep on day 3 and the farm had them on day
    # 15, asked for nine cows on day 7 and had them on day 13, while the
    # opening went into eighteen tiles of seed. The opponent ends day 1 with
    # $7 and four sheep already standing; this farm ended it with $1,092 and
    # none, then sat between $21 and $927 for the next ten days.
    #
    # So the seed queue, which runs first, hands back what the herd the
    # calendar has already committed to still costs.
    if P["sched_reserve"]:
        sched_now = _sched_for(day)
        if sched_now:
            owed = 0
            for _a in SPECIES_ORDER:
                _tgt = sched_now.get(_a)
                if _tgt is None:
                    continue
                _have = sum(1 for _x, _y, _t in animals if _species(_t) == _a)
                _have += int(shed.get(_a, 0))
                owed += max(0, int(_tgt) - _have) * ANIMALS[_a]['cost']
            seed_budget -= owed * P["sched_reserve"]

    # 4. Seeds first, animals second. Livestock outranks every crop per tile,
    #    so if the purchases run the other way the herd eats the whole budget
    #    and the farm ends up buying its own feed at $47 a bushel all season.
    if not liquidate:
        # Melon sits last in this queue, so on the opening day -- when the cash
        # runs out partway down it -- its seed is the one that never gets
        # bought. The public plan buys eight melon seeds before anything else
        # and sells 39 melons on day 10, which is what pays for its herd.
        seed_queue = ["WHEAT", "STRAWBERRY", "TOMATO", "CARROT", "MELON"]
        for crop in reversed(list(P["seed_priority"])):
            if crop in seed_queue:
                seed_queue.remove(crop)
                seed_queue.insert(0, crop)
        for crop in seed_queue:
            if day > P["plant_last_day"][crop]:
                continue
            short = deficit(crop) - seeds.get(crop, 0)
            short = min(short, 5)
            cost = CROPS[crop]["seed"]
            if short > 0 and seed_budget >= short * cost:
                buy_orders.append(["BUY_SEED", crop, short])
                money -= short * cost
                seed_budget -= short * cost
                spendable = min(spendable, money - P["cash_buffer"])

    # 5. Animals, best payer first. An animal bought on day 22 still has time
    #    to return its price once; later than that it is a donation. The herd
    #    may not outgrow the feed the farm can actually grow.
    feed_capacity = mine.get("WHEAT", 0) * RATE["WHEAT"]
    if day <= P["animal_buy_last_day"] and not liquidate:
        pending = shed_animals + carried_animals
        room = len(empty_tiles) + len(empty_struct) - pending
        # A ceiling on the herd, because sizing it by demand/rate is circular.
        # Measured off the two action lists: the top plan keeps twelve animals
        # and spends 967 CARE actions on them -- about eighty visits per head
        # in a season -- while this farm buys twenty-seven and spends 320,
        # which is twelve each. The care bonus is exactly what lifts an animal
        # from the 0.71 milk a real episode delivers to the table's 1.50, so
        # believing the low rate buys more cattle, which spreads the same
        # labour thinner, which keeps the rate low. Capping the herd is the
        # only way to test whether the rate is a fact about the environment or
        # a consequence of the herd's size.
        if P["herd_cap"] and not _sched_for(day):
            room = min(room, P["herd_cap"] - herd - pending)
        # Buy in order of return on the coin, not in a fixed species order.
        # With a fixed MILK-EGG-WOOL order the opening spends itself out on
        # cattle (first yield day 8) and never reaches sheep (day 6, and wool is
        # worth more per unit) -- which is exactly the opening the strong public
        # agents use. Sorting by payback per coin makes the choice follow the
        # live price and the days that are left.
        def roi(item):
            a = ANIMALS[PRODUCER[item]]
            producing = max(0, days_left - a["first_yield_day"])
            return price(item) * RATE[item] * producing / a["cost"]

        if P["animal_first"] and day <= P["animal_first_days"]:
            # Wool starts paying on day 6 against the cow's day 8, and the
            # public plan turns four sheep into $3,400 of wool on day 7 -- which
            # is the money its cows are bought with. Buying by payback per coin
            # picks the cow first and never reaches that opening.
            order = tuple(P["animal_first"])
        elif P["animal_order"] == "roi":
            order = sorted(("MILK", "EGG", "WOOL"), key=roi, reverse=True)
        else:
            order = ("MILK", "EGG", "WOOL")
        sched = _sched_for(day)
        for item in order:
            a = PRODUCER[item]
            # A scheduled head count replaces the demand estimate, and the
            # guards that protect the estimate go with it: whether the herd
            # pays for itself is the question the search is asking, so the
            # policy must not answer it here and refuse to buy.
            sched_forced = bool(sched and sched.get(a) is not None)
            if sched_forced:
                have = sum(1 for _x, _y, _t in animals if _species(_t) == a)
                need = int(sched[a]) - have - int(shed.get(a, 0)) - pending
            else:
                need = deficit(item) - pending
            if need <= 0 or room <= 0:
                continue
            cost = ANIMALS[a]["cost"]
            payback = price(item) * RATE[item] * max(0, days_left - ANIMALS[a]["first_yield_day"])
            if payback < cost * 1.2 and not sched_forced:
                continue
            # Each extra head eats one wheat a day; buying that on the market
            # costs more than the animal earns once the price climbs.
            wheat_px = prices.get("WHEAT", 25)
            headroom = int(feed_capacity) - (herd + pending)
            # Buy anyway when the feed is genuinely affordable: a month of
            # rations for one head is 30 wheat.
            can_buy_feed = money > cost + 30 * wheat_px
            # An animal's worth is its producing days, and the opening days are
            # the cheapest ones to buy: measured against the top agent this farm
            # ran 153 animal-days to their 312. In the first days wheat is still
            # near base and the first yield is days away, so the feed test only
            # delays the herd.
            grace = day <= P["animal_grace_day"] or sched_forced
            if headroom <= 0 and wheat_px > 32 and not can_buy_feed and not grace:
                continue
            k = 0
            while k < need and k < room and spendable >= cost and k < 4:
                if headroom <= k and wheat_px > 32 and not can_buy_feed and not grace:
                    break
                k += 1
                spendable -= cost
                money -= cost
            if k:
                buy_orders.append(["BUY_ANIMAL", a, k])
                pending += k
                room -= k

    # 6. Emergency feed: an escaped animal costs far more than dear wheat.
    if herd and not liquidate:
        # One day of rations is hand to mouth: the plan has to survive the turn
        # the price spikes, and a tile of wheat costs a season of watering that
        # a bushel off the market does not. The public plan buys 14 bushels on
        # day 0 for five animals and keeps buying all season.
        need = herd * P["feed_buy_days"] - int(shed.get("WHEAT", 0))
        if need > 0 and prices.get("WHEAT", 25) <= 80 and money > P["cash_buffer"]:
            k = int(min(need, (money - P["cash_buffer"]) // max(1, prices.get("WHEAT", 25))))
            if k > 0:
                buy_orders.append(["BUY_PRODUCT", "WHEAT", k])

    orders = (hire_orders + sell_orders + buy_orders)[:10]

    # ---------------------------------------------------------- unit actions
    claimed = set()
    sheds = shed_tiles(n)
    shed_here = set(sheds)
    fert_price = price("FERTILIZER")
    task_w = _task_weights(day)
    # Structures standing today, occupied or not. The calendar's target is
    # counted against this rather than against empty ones, because a pasture
    # with a cow on it is a pasture the farm already paid for and does not
    # need again.
    have_past = (sum(1 for _, _, t in animals if t.get("kind") == "PASTURE")
                 + sum(1 for _, _, t in empty_struct if t.get("kind") == "PASTURE"))
    have_coop = (sum(1 for _, _, t in animals if t.get("kind") == "COOP")
                 + sum(1 for _, _, t in empty_struct if t.get("kind") == "COOP"))
    struct_target = _struct_target(day)

    def can_act(pos):
        return quadrant_of(pos[0], pos[1], n) in unlocked

    def unit_inv(i):
        return invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}

    def plant_ready(t):
        cd = CROPS[t["crop"]]
        age = day - t["planted_day"]
        if cd["ongoing"]:
            return t.get("yield_units", 0) > 0 and age >= cd["first_yield_day"]
        return t.get("yield_units", 0) > 0 and age >= cd["max_yield_day"]

    # PLANT is validated atomically: if the hands ask for more seeds of a crop
    # than the shed holds, the environment drops *every* plant request for that
    # crop -- not just the surplus. With a dozen hands all aiming at the same
    # single seed this fired constantly: a measured episode issued 894 PLANT
    # actions against 243 harvests, so most of the sowing never happened.
    planted_this_turn = {}

    def seed_left(crop):
        return seeds.get(crop, 0) - planted_this_turn.get(crop, 0)

    def pick_crop():
        """Sow whatever is furthest under plan, weighted by what it earns."""
        # Feed first, always: wheat is the worst tile on the farm by revenue and
        # the only one whose absence kills the herd.
        need_feed = math.ceil((herd + shed_animals + carried_animals) * 1.3 / RATE["WHEAT"])
        if (mine.get("WHEAT", 0) < need_feed and seed_left("WHEAT") > 0
                and day <= P["plant_last_day"]["WHEAT"]):
            return "WHEAT"
        best, best_val = None, 0.0
        for crop in ("STRAWBERRY", "TOMATO", "MELON", "CARROT", "WHEAT"):
            if seed_left(crop) <= 0 or day > P["plant_last_day"][crop]:
                continue
            if deficit(crop) <= 0:
                continue
            val = price(crop) * RATE[crop]
            if val > best_val:
                best, best_val = crop, val
        return best

    def jobs_for(pos, inv):
        out = []
        wheat_held = inv.get("WHEAT", 0)
        held_animal = next((a for a in ANIMALS if inv.get(a, 0) > 0), None)

        for (x, y, t) in animals:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            a = ANIMALS[t["animal"]]
            unit_price = price(a["product"])
            if not t.get("fed_today") and wheat_held > 0:
                # Two unfed nights and the animal is gone for good.
                urgency = 4.0 if t.get("consecutive_unfed", 0) >= 1 else 1.5
                out.append((unit_price * urgency / (1 + P['dist_weight'] * d), (x, y), "FEED"))
            if t.get("yield_units", 0) > 0:
                val = t["yield_units"] * unit_price
                if t["yield_units"] >= a["max_held"]:
                    val *= 2  # production is being thrown away while it sits full
                out.append((val / (1 + P['dist_weight'] * d), (x, y), "HARVEST"))
            # The gate below caps care at one visit per animal per day, which
            # bounds a twelve-head farm at 360 CARE actions in a season. The
            # top plan issues 967 on twelve animals -- 2.7 per animal-day. So
            # either the bonus accrues per visit and this gate throws most of
            # it away, or two thirds of their labour is landing on an animal
            # already cared for and costs them nothing. `care_repeat` is how
            # that gets decided rather than argued.
            if t.get("fed_today") and (P["care_repeat"] or not t.get("cared_today")):
                # One care day = one extra unit on the next production.
                out.append((unit_price * 0.9 / (1 + P['dist_weight'] * d), (x, y), "CARE"))
            if t.get("fertilizer_available"):
                out.append((fert_price / (1 + P['dist_weight'] * d), (x, y), "COLLECT_FERTILIZER"))

        fert_held = inv.get("FERTILIZER", 0)
        for (x, y, t) in plants:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            cd = CROPS[t["crop"]]
            unit_price = price(t["crop"])
            if fert_held > 0 and t.get("fertilized_until_day", -1) < day:
                # Fertilizer is worth far more applied than sold: it doubles an
                # ongoing crop's scheduled yield for three days (tomato fires
                # daily, so one unit buys ~3 extra fruit) and doubles the daily
                # watering bonus on a one-time crop inside its window.
                extra = extra_from_fertilizer(t, cd, day)
                if extra > 0:
                    out.append((unit_price * extra / (1 + P['dist_weight'] * d), (x, y), "FERTILIZE"))
            if plant_ready(t):
                out.append((t["yield_units"] * unit_price / (1 + P['dist_weight'] * d), (x, y), "HARVEST"))
            elif not t.get("watered_today"):
                age = day - t["planted_day"]
                window = (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]
                if t.get("consecutive_unwatered", 0) >= 1:
                    val = unit_price * 3.0        # dies tonight otherwise
                elif window or cd["ongoing"]:
                    val = unit_price * 1.0        # this watering is a unit of yield
                else:
                    val = unit_price * 0.3
                out.append((val / (1 + P['dist_weight'] * d), (x, y), "WATER"))

        if held_animal:
            struct = ANIMALS[held_animal]["structure"]
            for (x, y, t) in empty_struct:
                if (x, y) in claimed or not can_act((x, y)) or t["kind"] != struct:
                    continue
                val = price(ANIMALS[held_animal]["product"]) * RATE[ANIMALS[held_animal]["product"]] * 4
                out.append((val / (1 + P['dist_weight'] * dist(pos, (x, y))), (x, y), ("PLACE", held_animal)))

        crop = pick_crop()
        pending_animals = shed_animals + carried_animals
        need_coop = sum(1 for _, _, t in empty_struct if t["kind"] == "COOP")
        need_past = sum(1 for _, _, t in empty_struct if t["kind"] == "PASTURE")
        # Offer the build-ahead job on at most as many tiles as the target is
        # short. Every empty tile would otherwise carry the offer, and with a
        # dozen hands bidding, a shortfall of two pastures buys twelve.
        #
        # Which tiles, and not just how many: the shortlist is the sites
        # nearest the shed, because siting is the one decision a tile never
        # revisits -- an animal placed across the farm is walked to every day
        # for the rest of the season. Taking the first N in scan order would
        # put the herd in whatever corner the loop happens to start in.
        short_past = max(0, struct_target.get("PASTURE", 0) - have_past)
        short_coop = max(0, struct_target.get("COOP", 0) - have_coop)
        build_ahead = set()
        if short_past or short_coop:
            sites = [(x, y) for (x, y) in empty_tiles if can_act((x, y))]
            if sheds:
                sites.sort(key=lambda p: min(dist(p, st) for st in sheds))
            build_ahead = set(sites[:max(short_past, short_coop)])
        for (x, y) in empty_tiles:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            # Siting is the one decision a tile never gets to revisit: an
            # animal placed across the farm is fed, cared for and collected
            # from every day for the rest of the season, and its feed comes out
            # of the shed. Scoring it only by how near the placing hand happens
            # to be is what scatters the farm.
            ds = min(dist((x, y), st) for st in sheds) if sheds else 0
            near_build = 1 + P['dist_weight'] * d + P["build_shed_weight"] * ds
            near_plant = 1 + P['dist_weight'] * d + P["plant_shed_weight"] * ds
            if pending_animals > 0:
                if shed.get("GOOSE", 0) + inv.get("GOOSE", 0) > need_coop:
                    out.append((price("EGG") * RATE["EGG"] * 2 / near_build, (x, y), "BUILD_COOP"))
                if shed.get("COW", 0) + shed.get("SHEEP", 0) > need_past:
                    out.append((price("MILK") * RATE["MILK"] * 2 / near_build, (x, y), "BUILD_PASTURE"))
            # Building ahead of the herd, when the calendar asks for it.
            #
            # The gate above is the whole reason this farm's animals are idle.
            # A pasture is only ever offered once animals are *already* sitting
            # in the shed, so the season serialises: buy the cow, let it wait,
            # walk a hand over, build, walk back, carry, place. Measured, that
            # costs the farm a third of its herd's working life -- 229
            # animal-days against the published plan's 312, which is 64% of
            # what a twelve-head farm can deliver against their 87% -- and an
            # animal-day lost is lost for good, because the milk that day is
            # simply never produced. It also explains why BUILD_PASTURE_w
            # measured positive but heterogeneous across seed groups: a weight
            # can only reorder jobs that are offered, and this one is offered
            # in some seasons and not others.
            #
            # So the structure target joins the calendar as its own cumulative
            # column, next to the head count it serves. Absent, nothing below
            # fires and the farm behaves exactly as before.
            if (x, y) in build_ahead:
                if short_coop > 0:
                    out.append((price("EGG") * RATE["EGG"] * 2 / near_build, (x, y), "BUILD_COOP"))
                if short_past > 0:
                    out.append((price("MILK") * RATE["MILK"] * 2 / near_build, (x, y), "BUILD_PASTURE"))
            if crop:
                out.append((price(crop) * RATE[crop] * 1.5 / near_plant, (x, y), ("PLANT", crop)))

        # A weed is worth clearing exactly as much as whatever would be planted
        # on the tile it is squatting on. Gating this on "almost no empty tiles
        # left" let weeds reach 26 of 100 tiles by day 27 -- a quarter of the
        # farm sitting idle while hands walked past it.
        weed_val = 25.0
        if crop and want_more_tiles:
            weed_val = price(crop) * RATE[crop] * 1.3
        for (x, y, t) in weeds:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            out.append((weed_val / (1 + P['dist_weight'] * dist(pos, (x, y))), (x, y), "DIG"))

        # Shed trips: fetch feed, fetch an animal, or unload a full pack.
        unfed = sum(1 for _, _, t in animals if not t.get("fed_today"))
        carried = sum(v for v in inv.values() if isinstance(v, int))
        carried_wheat = sum(iv.get("WHEAT", 0) for iv in invs if isinstance(iv, dict))
        carried_fert = sum(iv.get("FERTILIZER", 0) for iv in invs if isinstance(iv, dict))
        for st in sheds:
            d = dist(pos, st)
            stock = int(shed.get("WHEAT", 0))
            worth_the_walk = stock >= min(unfed, P["pickup_min"]) if unfed else stock > 0
            may_top_up = P["pickup_topup"] or wheat_held <= 0
            if (unfed > carried_wheat and unfed > wheat_held and stock > 0
                    and worth_the_walk and may_top_up):
                out.append((price("MILK") * 1.2 / (1 + P['dist_weight'] * d), st,
                            ("PICKUP", "WHEAT", min(14, stock))))
            if fert_targets > carried_fert and fert_held == 0 and shed.get("FERTILIZER", 0) > 0:
                out.append((price("STRAWBERRY") * 1.5 / (1 + P['dist_weight'] * d), st,
                            ("PICKUP", "FERTILIZER", min(10, shed.get("FERTILIZER", 0)))))
            if shed_animals > carried_animals and not held_animal:
                a = next(a for a in ANIMALS if shed.get(a, 0) > 0)
                out.append((price(ANIMALS[a]["product"]) * 3 / (1 + P['dist_weight'] * d), st, ("PICKUP", a, 1)))
            if carried >= P["drop_load"]:
                # Produce only earns once it is in the shed, because SELL draws
                # from the shed and nothing else. Waiting for the automatic
                # nightly drop leaves the market orders with nothing to sell --
                # measured at 80 sell orders against the top agent's 195 -- and
                # risks the 100-item cap discarding the day's harvest.
                worth = sum(price(k) * v for k, v in inv.items()
                            if k in MARKET_PARAMS and isinstance(v, int))
                out.append((worth * P["drop_urgency"] / (1 + P['dist_weight'] * d), st, "DROP"))

        # Applied before bundling, so a tile's queue is credited with what the
        # day actually wants done there rather than with the unweighted total.
        out = _weigh(out, task_w)

        if P["bundle_weight"] and out:
            # Credit each candidate with what else is waiting where it stands.
            # The unit that walks there will still be there next turn, so the
            # rest of that tile's queue costs it no steps at all -- which is
            # exactly the difference between 1.33 jobs per arrival and 2.07.
            by_tile = {}
            for value, tile, _op in out:
                by_tile[tile] = by_tile.get(tile, 0.0) + value
            out = [(value + P["bundle_weight"] * (by_tile[tile] - value), tile, op)
                   for value, tile, op in out]
        return out

    def resolve(pos, target_tile, op):
        if pos != target_tile:
            mv = step_toward(pos, target_tile)
            return [mv] if mv else ["PASS"]
        return list(op) if isinstance(op, tuple) else [op]

    # ---- route planner ---------------------------------------------------
    # Picking the single best job every turn makes hands cross the farm: 55% of
    # this agent's actions were steps against the top agent's 43%, and 834 were
    # idle PASSes against their 323. Planning a whole day's round per hand -- a
    # cheapest-insertion route over the day's tasks -- lets one hand water a
    # column, harvest on the way back and hit the shed once.
    def tile_of(t):
        return t[0]

    def task_still_valid(tile, op):
        x, y = tile
        if tile in shed_here and isinstance(op, tuple) and op[0] in ("PICKUP", "DROP"):
            return True
        t = tiles[y][x] if 0 <= y < n and 0 <= x < n else None
        name = op[0] if isinstance(op, tuple) else op
        if name in ("PLANT", "BUILD_COOP", "BUILD_PASTURE"):
            return t is None
        if name == "DIG":
            return isinstance(t, dict) and t.get("kind") == "WEED"
        if name == "PLACE":
            return isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and "animal" not in t
        if not isinstance(t, dict):
            return False
        if name == "WATER":
            return t.get("kind") == "PLANT" and not t.get("watered_today")
        if name == "HARVEST":
            return t.get("yield_units", 0) > 0
        if name == "FEED":
            return "animal" in t and not t.get("fed_today")
        if name == "CARE":
            return "animal" in t and t.get("fed_today") and not t.get("cared_today")
        if name == "COLLECT_FERTILIZER":
            return "animal" in t and t.get("fertilizer_available")
        if name == "FERTILIZE":
            return t.get("kind") == "PLANT" and t.get("fertilized_until_day", -1) < day
        return True

    def plan_routes():
        """Cheapest-insertion assignment of the day's tasks to hands."""
        budget = max(1, TURNS_PER_DAY - hour)
        # One pooled task list, scored without distance so value decides the
        # order and geography decides the owner.
        pool, seen = [], set()
        for pos in units[:1] + [(n // 2, n // 2)]:
            for val, tile, op in jobs_for(pos, {}):
                sig = (tile, repr(op))
                if sig in seen or tile in shed_here:
                    continue
                seen.add(sig)
                pool.append((val * (1 + P["dist_weight"] * dist(pos, tile)), tile, op))
        # Feeding and fertilizing need cargo, which jobs_for only offers to a
        # hand already carrying it; add them here against the shed stock.
        wheat_stock = int(shed.get("WHEAT", 0))
        fert_stock = int(shed.get("FERTILIZER", 0))
        # Weighted like the rest: a day that wants less feeding has to mean it
        # here too, or the route planner quietly restores what jobs_for cut.
        feed_w = (task_w or {}).get("FEED", 1.0)
        fert_w = (task_w or {}).get("FERTILIZE", 1.0)
        for (x, y, t) in animals:
            if not t.get("fed_today") and wheat_stock > 0:
                pool.append((price(ANIMALS[t["animal"]]["product"]) * 2.0 * feed_w, (x, y), "FEED"))
                wheat_stock -= 1
        for (x, y, t) in plants:
            cd = CROPS[t["crop"]]
            if fert_stock > 0 and t.get("fertilized_until_day", -1) < day:
                extra = extra_from_fertilizer(t, cd, day)
                if extra > 0:
                    pool.append((price(t["crop"]) * extra * fert_w, (x, y), "FERTILIZE"))
                    fert_stock -= 1

        routes = {i: [] for i in range(len(units))}
        ends = {i: units[i] for i in range(len(units))}
        spent = {i: 0 for i in range(len(units))}
        for _, tile, op in sorted(pool, key=lambda c: -c[0]):
            best, best_cost = None, None
            for i in range(len(units)):
                c = spent[i] + dist(ends[i], tile) + 1
                if c <= budget and (best_cost is None or c < best_cost):
                    best, best_cost = i, c
            if best is None:
                continue
            routes[best].append((tile, op))
            spent[best] = best_cost
            ends[best] = tile

        # Prefix a shed stop for whatever cargo the route needs.
        for i, route in routes.items():
            need_wheat = sum(1 for _, op in route if op == "FEED")
            need_fert = sum(1 for _, op in route if op == "FERTILIZE")
            inv = unit_inv(i)
            stop = []
            if need_wheat > inv.get("WHEAT", 0) and shed.get("WHEAT", 0) > 0:
                stop.append(("PICKUP", "WHEAT", min(need_wheat, int(shed.get("WHEAT", 0)))))
            if need_fert > inv.get("FERTILIZER", 0) and shed.get("FERTILIZER", 0) > 0:
                stop.append(("PICKUP", "FERTILIZER", min(need_fert, int(shed.get("FERTILIZER", 0)))))
            if stop:
                st = min(sheds, key=lambda s: dist(units[i], s))
                routes[i] = [(st, op) for op in stop] + route
        return routes

    key = (me, day)
    if _MEM.get("key") != key:
        _MEM["key"] = key
        _MEM["routes"] = {}
        _MEM["assign"] = {}

    actions = []
    if P["planner"] == "route":
        routes = _MEM.get("routes") or {}
        # Replan at dawn, once the roster has finished filling, and whenever the
        # farm has run out of planned work.
        if (hour in (3, 9, 15, 21) or len(routes) != len(units)
                or not any(routes.get(i) for i in range(len(units)))):
            routes = plan_routes()
            _MEM["routes"] = routes
        for i, pos in enumerate(units):
            route = routes.get(i) or []
            while route and not task_still_valid(route[0][0], route[0][1]):
                route.pop(0)
            if not route:
                actions.append(["PASS"])
                continue
            tile, op = route[0]
            act = resolve(pos, tile, op)
            if pos == tile:
                route.pop(0)
            actions.append(act)
            routes[i] = route
    else:
        prev_assign = _MEM["assign"]
        new_assign = {}
        # First refusal on the ground under your feet.
        #
        # A unit already standing on a tile is the cheapest possible worker for
        # whatever is left there -- an animal wants feeding, then care, then its
        # fertilizer collected, and none of that costs a step. But units are
        # served in roster order and each one claims its tile, so the farmer and
        # the low-numbered hands take the best tiles on the whole farm first,
        # and a hand parked on a half-finished animal is told to walk away from
        # it. `finish_tile` cannot reach this: it is a weight applied while a
        # unit sorts its candidates, and a tile another unit has already claimed
        # never becomes a candidate at all -- which is why raising it past 3.0
        # changes nothing (measured with knob_bite: it bites at 1.5 and
        # saturates at 3.0).
        #
        # Measured off the two action lists, this farm chains 1.49 jobs per
        # arrival against the top plan's 2.07, and walks 60% of its actions
        # against 43%.
        prestaked = {}
        if P["stand_first"]:
            for i, pos in enumerate(units):
                if not can_act(pos):
                    continue
                here = [c for c in jobs_for(pos, unit_inv(i)) if c[1] == pos]
                if not here:
                    continue
                here.sort(key=lambda c: -c[0])
                prestaked[i] = here[0]
                if pos not in shed_here and P["assign_rule"] != "global":
                    claimed.add(pos)

        far_now = (P["farmer_far_bias"] > 0
                   and hour >= TURNS_PER_DAY - P["farmer_far_turns"] and sheds)

        def _rank(pos, val, tile, op, was, i=None):
            """The same score the roster rule sorts by, as a function so both
            rules read the candidate identically."""
            r = (val
                 * (P["stickiness"] if was == (tile, repr(op)) else 1.0)
                 * (P["finish_tile"] if tile == pos else 1.0))
            if far_now and i == 0:
                r *= 1.0 + P["farmer_far_bias"] * min(dist(tile, st) for st in sheds) / n
            return r

        if P["assign_rule"] == "global":
            # Every candidate of every unit, scored before anything is claimed,
            # then filled from the best pair down. A unit that loses a tile to a
            # closer one simply falls through to its own next candidate, which
            # is what roster order cannot do: there the loser has already been
            # served and the winner is the one left walking.
            pairs = []
            for i, pos in enumerate(units):
                cand = jobs_for(pos, unit_inv(i))
                if i in prestaked:
                    cand = [prestaked[i]] + cand
                was = prev_assign.get(i)
                for val, tile, op in cand:
                    pairs.append((_rank(pos, val, tile, op, was, i), i, tile, op))
            pairs.sort(key=lambda c: -c[0])
            picked = {}
            for _, i, tile, op in pairs:
                if i in picked or (tile in claimed and tile not in shed_here):
                    continue
                sowing = (isinstance(op, tuple) and op[0] == "PLANT"
                          and units[i] == tile)
                # The environment throws away *every* request for a crop whose
                # turn total exceeds the seeds held, so the reservation has to
                # happen while the pairs are being taken, not afterwards.
                if sowing and seed_left(op[1]) <= 0:
                    continue
                picked[i] = (tile, op)
                if tile not in shed_here:
                    claimed.add(tile)
                if sowing:
                    planted_this_turn[op[1]] = planted_this_turn.get(op[1], 0) + 1
            for i, pos in enumerate(units):
                if i not in picked:
                    actions.append(["PASS"])
                    continue
                tile, op = picked[i]
                new_assign[i] = (tile, repr(op))
                actions.append(resolve(pos, tile, op))
            _MEM["assign"] = new_assign
            return {"farmer": actions[0] if actions else ["PASS"],
                    "hands": actions[1:],
                    "market": orders}

        for i, pos in enumerate(units):
            cand = jobs_for(pos, unit_inv(i))
            # The pre-pass claimed this unit's own tile, so jobs_for no longer
            # offers it; hand it back, and let the ordinary sort (with
            # finish_tile) decide whether standing still beats walking.
            if i in prestaked:
                cand = [prestaked[i]] + cand
            if not cand:
                actions.append(["PASS"])
                continue
            was = prev_assign.get(i)
            # A job on the tile under a unit's feet costs no steps, so it is
            # already favoured by the 1/(1+d) discount -- but only by the width
            # of that discount, which a richer job two tiles away can beat. The
            # top plan chains 2.07 jobs per stop against this farm's walking
            # 63% of the time, and an animal wants three visits a day: feed,
            # care, collect.
            cand.sort(key=lambda c: -_rank(pos, c[0], c[1], c[2], was, i))
            _, tile, op = cand[0]
            if tile not in shed_here:
                claimed.add(tile)
            # Reserve the seed the moment a hand is aimed at a tile, so the next
            # hand is offered a different crop instead of a request the
            # environment will throw away.
            if isinstance(op, tuple) and op[0] == "PLANT" and pos == tile:
                planted_this_turn[op[1]] = planted_this_turn.get(op[1], 0) + 1
            new_assign[i] = (tile, repr(op))
            actions.append(resolve(pos, tile, op))
        _MEM["assign"] = new_assign

    return {"farmer": actions[0] if actions else ["PASS"],
            "hands": actions[1:],
            "market": orders}


# ---------------------------------------------------------------------
# Capital calendar, bound after the definitions above so it is in force
# before the first turn. Generated by sim/emit_sched.py -- edit the
# calendar, not this file.
# 同じカレンダー。assign_rule と farmer_far_bias を足した main.py で載せ直したもの
SCHEDULE = {"0": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 1, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 100, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 0, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 5, "land": 1}, "1": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 1, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 0, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 1, "land": 1}, "10": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 5, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 14, "land": 2}, "11": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 5, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 10, "land": 2}, "12": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 10, "land": 2}, "13": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 8, "land": 2}, "14": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 100, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 10, "land": 2}, "15": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 40, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 75, "hands": 8, "land": 2}, "16": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 40, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 75, "hands": 12, "land": 2}, "17": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 40, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 75, "hands": 5, "land": 2}, "18": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 100, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 6, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 75, "hands": 8, "land": 2}, "19": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 100, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 100, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 75, "hands": 10, "land": 2}, "2": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 1, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 0, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 2, "land": 1}, "20": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 160, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 75, "hands": 11, "land": 2}, "21": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 160, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 75, "hands": 10, "land": 2}, "22": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 60, "PASTURE": 0, "PICKUP_w": 120, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 75, "hands": 7, "land": 2}, "23": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 60, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 80, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 140, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 100, "hands": 10, "land": 2}, "24": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 80, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 100, "hands": 10, "land": 2}, "25": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 80, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 100, "hands": 11, "land": 2}, "27": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 20, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 100, "hands": 10, "land": 2}, "28": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 60, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 60, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 100, "hands": 11, "land": 2}, "29": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 140, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 8, "DIG_w": 100, "DROP_w": 100, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 1, "HARVEST_w": 120, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 100, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 8, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 75, "WHEAT_pct": 100, "hands": 11, "land": 2}, "3": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 3, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 3, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 4, "land": 1}, "4": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 5, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 3, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 4, "land": 1}, "5": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 6, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 3, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 4, "land": 1}, "6": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 6, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 5, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 4, "land": 1}, "7": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 5, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 7, "land": 2}, "8": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 5, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 6, "land": 2}, "9": {"BUILD_COOP_w": 100, "BUILD_PASTURE_w": 125, "CARE_w": 100, "CARROT_pct": 100, "COLLECT_FERTILIZER_w": 100, "COOP": 0, "COW": 7, "DIG_w": 100, "DROP_w": 125, "FEED_w": 100, "FERTILIZE_w": 100, "GOOSE": 0, "HARVEST_w": 100, "MELON_pct": 100, "PASTURE": 0, "PICKUP_w": 75, "PLACE_w": 100, "PLANT_w": 100, "SHEEP": 5, "STRAWBERRY_pct": 100, "TOMATO_pct": 100, "WATER_w": 100, "WHEAT_pct": 100, "hands": 7, "land": 2}}

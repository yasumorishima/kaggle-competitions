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
    "sched_hands_scale": 1.0,  # multiplier on the calendar's hand count (it overrides the two above)
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
    "reserve_by_item": {},     # ...except where it does not: {"FERTILIZER": 0.0, "MELON": 0.0}
    # 0 = off, and off is what every sweep of `reserve_frac` was measured
    # under. The reserve is not a floor: `qty = max(qty, pace)` below runs
    # whatever the price is, and the reserve itself is
    # `max(base * frac, now * slice_frac)` -- the second term follows the
    # falling price down, so neither term is anchored to anything absolute.
    # Measured on seed 3000 against the top plan, this farm put five units of
    # wool on the wire at a mean quote of 3.8 against a base of 200, which is
    # the price floor of the environment: not patience lost, goods given away.
    # This is the one direction the six sell-floor sweeps never went -- they
    # all loosened `reserve_frac`, a number the pace fallback overrides.
    "pace_floor_frac": 0.0,    # pace may not sell under base * this
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
    # 1.0 = off. Stickiness was a patch on roster order -- a unit that had been
    # aimed at a tile kept it rather than being outbid by the next unit served.
    # Under "global" the assignment is already settled before anyone moves, so
    # the bonus only stops a better pair from being taken: with it at 1.6 the
    # global rule is worth +848 (a tie), at 1.0 it is worth +2,657. Off the
    # global rule -- that is, under "roster" -- turning it off is -295, a tie,
    # so this is not a change that stands on its own.
    "stickiness": 1.0,         # bonus for keeping a hand on the tile it set out for
    # How steeply travel discounts a job -- a job d tiles away is worth
    # value/(1 + dist_weight * d). Lowered from 1.0 on 2026-08-25 after three
    # disjoint seed bands against the ladder-representative opponent, 96 games
    # each, put the same interior peak in the same place: +4,626 +/- 2,144
    # (4700), +5,438 +/- 2,753 (4900), +2,411 +/- 2,964 (5300), inverse-variance
    # +4,313 +/- 1,469. The neighbours are not a plateau -- 0.5 and 0.9 sit
    # about 3,000 below 0.7 in both bands that measured them -- so this is a
    # located optimum and not a maximum re-picked out of noise. Nor is it a
    # licence to ignore distance: 0.2 is -9,549 and 0.0 is -45,939, the agent
    # simply thrashing. Under the global assignment rule the value is what
    # decides whether the matcher may send a hand across the farm for a big
    # job, which is why the old measurement (taken under the roster rule, and
    # bundled with stand_first off) never settled.
    "dist_weight": 0.7,
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
    # Measured over three disjoint seed bands of 96 games each against a
    # leaderboard-median opponent: +6,098 / +5,551 / +4,554, pooled by inverse
    # variance to +5,441 +/- 1,461 (t=7.30). "roster" is kept switchable
    # because every measurement in this project before 2026-08-25 was taken
    # under it, and a verdict is conditional on the rule it was measured with.
    "assign_rule": "global",   # "roster" = in order, first come first served; "global" = best pair first
    # Under "global" every pair is scored before any is taken, so all units are
    # offered the same crop and the ones past the seed count fall through to
    # something that is not sowing at all -- the farm sowed 112 times against
    # the roster rule's 211 in a measured season. Off, that halving stands and
    # has to be read as part of whatever the rule is worth; on, the tile keeps
    # its turn and only the crop is re-asked. Switchable because the first
    # measurement of "global" was taken with it off.
    "global_resow": False,
    # Measured 2026-08-25: when the seed for the first-choice crop runs out and
    # the hand does farm work instead of sowing a second choice, the season is
    # worth +3,441 -- and the control that merely sows fewer tiles is worth
    # -1,243, so it is not "plant less", it is "this particular sowing was
    # worth less than the watering and the care it displaced". That says the
    # sowing candidate is scored too high against the rest. The calendar owns a
    # PLANT dial already, but it has sat at 100 for every day of every climb, so
    # nothing has ever tested the claim. This scales the calendar's dials
    # instead of replacing them, which keeps the climb's ownership of the line.
    "task_w_scale": {},        # {"PLANT": 0.5, ...} applied on top of the calendar's dials
    # The same idea for the crop mix. The calendar carries a per-day `*_pct`
    # dial per crop; this multiplies it, so a whole mix can be moved in one
    # sweep entry without rewriting thirty days of calendar. It exists because
    # the crop question is a *set*: the third quadrant lost 5,741 when it was
    # bought and filled with wheat, and a side-by-side against the top replay
    # on seed 5100 shows the difference is not the land but what stands on it
    # -- 36 strawberry plants against our 17, 243 units sold against 110, at a
    # price the town held at 238 for both of us.
    "crop_pct_scale": {},      # {"STRAWBERRY": 2.0, "WHEAT": 0.5, ...}
    # Order the tile budget is handed to the cash crops in. A list names it
    # explicitly; "value" sorts by price * yield at today's quote. The default
    # is the order that was written by hand, kept as the default only because
    # every verdict before 2026-08-25 was measured under it.
    "crop_order": ["TOMATO", "CARROT", "MELON", "STRAWBERRY"],
    "stand_first": 1,
    "care_repeat": 0,          # 1 = offer CARE again on an animal already cared today
    "care_urgency": 1.0,       # multiplier on CARE inside the last `care_deadline` hours (1.0 = off)
    "care_deadline": 6,
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
    # The four cash crops, in an order the tile budget is spent in. The order
    # is the whole question: `take` hands out a fixed budget in sequence, so
    # whatever comes last gets the remainder, and strawberry -- the dearest
    # thing the farm can grow, $250 against tomato's $67 -- was written last.
    # A season measured against the top replay on seed 5100 shows what that
    # costs: 11-17 strawberry plants standing against their 36, 110 units sold
    # against 243, in a town holding the price at $238 for both of us. Tomato
    # took its cap ahead of it and returned 29 units for $1,617 all season.
    #
    # "value" sorts them by what a tile of each is worth at today's price
    # instead. Livestock is deliberately not in the sort: it outbids every
    # crop per tile and eating the whole budget with it starved the farm when
    # that was tried.
    cash_crops = {
        "TOMATO": lambda: min(market_cap("TOMATO"), P["tomato_cap"]),
        "CARROT": lambda: (min(market_cap("CARROT"), P["carrot_cap"])
                           if "PET_CAFE" in shops else 0),
        "MELON": lambda: min(market_cap("MELON"), P["melon_cap"]),
        "STRAWBERRY": lambda: min(market_cap("STRAWBERRY"), budget),
    }
    order = P["crop_order"]
    if order == "value":
        order = sorted(cash_crops, key=lambda c: -price(c) * RATE[c])
    for _crop in order:
        take(_crop, cash_crops[_crop]())
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
    if sched or P["crop_pct_scale"]:
        for _crop in CROP_KEYS:
            mult = (sched or {}).get(_crop + '_pct')
            scale = P["crop_pct_scale"].get(_crop)
            if mult is None and scale is None:
                continue
            mult = 100.0 if mult is None else float(mult)
            if scale is not None:
                mult *= float(scale)
            base = target.get(_crop, 0)
            if base == 0 and mult > 100:
                base = 1          # a dial above 100 may open a crop the plan skipped
            target[_crop] = max(0, int(round(base * mult / 100.0)))

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
            # `reserve_frac` says never sell under the opening price, on the
            # grounds that scarcity lifts prices all season. Printing the
            # town's quote day by day says that is true of seven goods and
            # false of two: fertilizer falls from 100 to 45 without ever
            # turning, and melon from 250 to 131. For those two the floor is
            # not patience, it is a stop order that fires on day three and
            # never lifts -- `sellable_qty` returns nothing from then on, and
            # what does get sold trickles out through the pace fallback at
            # whatever the price has fallen to. Measured on seed 5100 this farm
            # moved 85 units of fertilizer at a mean of 61.7 while the top plan
            # moved 300 at 70.3. A single global fraction cannot be right for
            # both kinds of good, which is the likeliest reason sweeping it
            # read as noise; this lets the two be named.
            frac = P["reserve_by_item"].get(item, P["reserve_frac"])
            reserve = max(MARKET_PARAMS[item]["base"] * frac,
                          now * P["slice_frac"])
            qty = sellable_qty(item, int(inventory.get(item, MARKET_I0)), have, reserve)
            # Holding is not free: the shed caps at 100 items and discards the
            # rest at nightfall, and production never stops. So clear the day's
            # stock at a steady pace regardless of the reserve -- measured
            # against the top agent, under-selling cost more than price impact
            # (67 sell orders against their 196, with milk piling up unsold).
            turns_left = max(1, TURNS_PER_DAY - hour)
            pace = -(-have // turns_left)
            # The shed still overflows at nightfall whatever the price is, so
            # the floor holds the pace back but never the overflow dump: the
            # choice there is between a bad price and no price at all.
            hard = MARKET_PARAMS[item]["base"] * P["pace_floor_frac"]
            if not hard or now >= hard:
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
            # The calendar's hand count replaces everything above it, which
            # makes `max_hands` and `jobs_per_hand` dead letters in any agent
            # carrying a calendar -- and every agent does. The scale is here so
            # the one question five failed mechanism readings kept circling can
            # be asked directly: is this farm short of labour or short of
            # demand? It sells about half of what the top plan sells in every
            # product at once, which is what either constraint looks like.
            want = int(round(int(sched['hands']) * P["sched_hands_scale"]))
            want = max(P['hands_min'], min(20, want))
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
    if P["task_w_scale"]:
        task_w = dict(task_w or {})
        for _name, _mult in P["task_w_scale"].items():
            task_w[_name] = task_w.get(_name, 1.0) * float(_mult)
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
                care_val = unit_price * 0.9
                # ...and a care day that ends unspent is gone for good, while a
                # watering can be done next turn. `FEED` already says so with an
                # urgency of 1.5 rising to 4.0, and `WATER` with 3.0 for a plant
                # that dies tonight; `CARE` said nothing, and lost to whatever
                # was nearer whenever the day ran short. What that costs is
                # readable off one season: eight cows opposite seven produced
                # 320 units of milk against 145 -- 1.33 a head a day against
                # 0.69, the published table rate against half of it -- which on
                # seed 5100 is $42,400, the largest single line in the gap.
                if P["care_urgency"] > 1.0 and TURNS_PER_DAY - 1 - hour <= P["care_deadline"]:
                    care_val *= P["care_urgency"]
                out.append((care_val / (1 + P['dist_weight'] * d), (x, y), "CARE"))
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
                #
                # Every pair was scored before any of them was taken, so all
                # units were offered the same crop -- the one `pick_crop` liked
                # with nothing yet reserved. Dropping the ones past the seed
                # count would halve the farm's sowing, because the second-choice
                # crop was never generated as a candidate at all. Re-ask instead:
                # the tile is still the right tile, only the crop is stale.
                if sowing and seed_left(op[1]) <= 0:
                    alt = pick_crop() if P["global_resow"] else None
                    if not alt or seed_left(alt) <= 0:
                        continue
                    op = ("PLANT", alt)
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


# ---------------------------------------------------------------------------
# Half a recorded plan, bound after the definitions above. Generated by
# sim/emit_mix.py -- edit the plan, not this file.
#
# HALF = "labour": the farmer and hands come from the plan, the market orders
#        from the policy, which reads the shed this turn.
# HALF = "market": the reverse, as the control.
import json as _json          # main.py imports math and nothing else
_POLICY_AGENT = agent
HALF = "market"
_PLAN = _json.loads(r'''[{"farmer":["PASS"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_SEED","WHEAT",5],["BUY_SEED","TOMATO",5],["BUY_SEED","MELON",4],["BUY_ANIMAL","COW",4]]},{"farmer":["BUILD_PASTURE"],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["BUILD_PASTURE"]],"market":[["BUY_SEED","WHEAT",2],["BUY_SEED","TOMATO",1]]},{"farmer":["NORTH"],"hands":[["WEST"],["NORTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["PLACE","COW"],["PLACE","COW"],["NORTH"],["PLANT","WHEAT"]],"market":[["SELL","MELON",1]]},{"farmer":["NORTH"],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["WEST"],["NORTH"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["PICKUP","WHEAT",1],["PICKUP","WHEAT",1],["EAST"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["FEED"],["WEST"],["WEST"],["NORTH"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["CARE"],["EAST"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["WEST"],["NORTH"],["PLANT","WHEAT"]],"market":[]},{"farmer":["NORTH"],"hands":[["PLANT","WHEAT"],["PLANT","WHEAT"],["WATER"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["WATER"],["NORTH"],["PLANT","WHEAT"]],"market":[]},{"farmer":["WEST"],"hands":[["PLANT","WHEAT"],["WEST"],["WATER"],["NORTH"]],"market":[]},{"farmer":["PLANT","MELON"],"hands":[["NORTH"],["PLANT","MELON"],["WEST"],["PLANT","MELON"]],"market":[]},{"farmer":["WATER"],"hands":[["PLANT","MELON"],["WATER"],["NORTH"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["NORTH"],["PLANT","TOMATO"],["WEST"]],"market":[["BUY_SEED","STRAWBERRY",1]]},{"farmer":["NORTH"],"hands":[["WEST"],["PLANT","STRAWBERRY"],["WATER"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["PLANT","TOMATO"],["WATER"],["WEST"],["PLANT","TOMATO"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["NORTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["PLANT","TOMATO"],"hands":[["WEST"],["PLANT","TOMATO"],["WEST"],["PLANT","TOMATO"]],"market":[]},{"farmer":["WATER"],"hands":[["PASS"],["WATER"],["WATER"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","WHEAT",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["CARE"],["NORTH"],["CARE"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["NORTH"],["FERTILIZE"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["NORTH"],["WEST"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["NORTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["NORTH"],["WEST"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["WATER"],["WATER"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["EAST"],["EAST"],["EAST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["SOUTH"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["SOUTH"],["SOUTH"],["SOUTH"],["SOUTH"]],"market":[]},{"farmer":["PASS"],"hands":[["WATER"],["SOUTH"],["SOUTH"],["SOUTH"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["SOUTH"],["WATER"],["SOUTH"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["WATER"],["PASS"],["WATER"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_SEED","WHEAT",5]]},{"farmer":["PICKUP","WHEAT",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["BUY_SEED","WHEAT",4]]},{"farmer":["FEED"],"hands":[["WEST"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["CARE"],["NORTH"],["CARE"],["CARE"],["CARE"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["NORTH"],["WEST"],["WEST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["WATER"],["FERTILIZE"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["WEST"],["WEST"],["WATER"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["WATER"],["WATER"],["WEST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WATER"],["WATER"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["WEST"],["WATER"],["NORTH"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["PLANT","WHEAT"],["EAST"],["WEST"],["WATER"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["WATER"],["WATER"],["NORTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["NORTH"],["NORTH"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["PLANT","WHEAT"],["PLANT","WHEAT"],["WATER"],["NORTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["PASS"],"hands":[["WATER"],["WATER"],["PASS"],["PLANT","WHEAT"],["PASS"],["PLANT","WHEAT"]],"market":[]},{"farmer":["WEST"],"hands":[["PASS"],["SOUTH"],["PASS"],["WATER"],["PASS"],["PASS"]],"market":[]},{"farmer":["WATER"],"hands":[["PASS"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PLANT","WHEAT"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["WATER"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PICKUP","SHEEP",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","WHEAT",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"]]},{"farmer":["FEED"],"hands":[["WEST"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"],["WEST"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["CARE"],["NORTH"],["CARE"],["CARE"],["CARE"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["WATER"],["NORTH"],["WATER"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["WEST"],["WEST"],["FERTILIZE"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["WATER"],["WATER"],["WATER"],["WEST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["NORTH"],["WATER"],["WATER"],["NORTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["SOUTH"],["NORTH"],["WEST"],["WEST"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["NORTH"],["WEST"],["WEST"],["WEST"],["WEST"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["WATER"],["WEST"],["WEST"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["WATER"],["PASS"],["WATER"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["WEST"],["WATER"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["WATER"],["PASS"],["WATER"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["WATER"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","WHEAT",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"]]},{"farmer":["FEED"],"hands":[["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"],["WEST"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["NORTH"],["NORTH"],["CARE"],["CARE"],["CARE"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["HARVEST"],["WEST"],["WEST"],["WEST"],["WEST"],["NORTH"]],"market":[]},{"farmer":["HARVEST"],"hands":[["NORTH"],["NORTH"],["WEST"],["WEST"],["WEST"],["NORTH"],["NORTH"]],"market":[["BUY_SEED","WHEAT",1],["BUY_SEED","WHEAT",2]]},{"farmer":["PLANT","WHEAT"],"hands":[["HARVEST"],["HARVEST"],["HARVEST"],["NORTH"],["WEST"],["NORTH"],["NORTH"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WATER"],"hands":[["PLANT","WHEAT"],["PLANT","WHEAT"],["WEST"],["FERTILIZE"],["NORTH"],["HARVEST"],["HARVEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["WATER"],["WATER"],["WATER"],["WATER"],["NORTH"],["NORTH"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["NORTH"],"hands":[["NORTH"],["SOUTH"],["EAST"],["NORTH"],["NORTH"],["WATER"],["WATER"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["PLANT","WHEAT"],["PLANT","WHEAT"],["PLANT","WHEAT"],["WATER"],["WATER"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["WATER"],["WATER"],["NORTH"],["NORTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["EAST"],["WEST"],["WATER"],["NORTH"],["WEST"],["WATER"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["DROP"],["WEST"],["WEST"],["WATER"],["WEST"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["PASS"],["WATER"],["WATER"],["WEST"],["WEST"],["PASS"]],"market":[["SELL","WHEAT",6]]},{"farmer":["WATER"],"hands":[["WEST"],["PASS"],["NORTH"],["PASS"],["WATER"],["PASS"],["PASS"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["PASS"],["WATER"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["EAST"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["EAST"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["EAST"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["SOUTH"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["SOUTH"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["DROP"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PICKUP","COW",1],"hands":[["PASS"],["PICKUP","COW",1],["EAST"],["EAST"],["EAST"],["EAST"],["EAST"]],"market":[["SELL","WHEAT",5]]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","WHEAT",13]]},{"farmer":["PICKUP","WHEAT",3],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"]]},{"farmer":["FEED"],"hands":[["WEST"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"],["WEST"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["CARE"],["NORTH"],["CARE"],["CARE"],["CARE"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["NORTH"],["WEST"],["WEST"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["NORTH"],["NORTH"],["WEST"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["NORTH"],["NORTH"],["FERTILIZE"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["WEST"],["WATER"],["WATER"],["WATER"],["WATER"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["WEST"],["WEST"],["WEST"],["WEST"],["WATER"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["WEST"],["WEST"],["WATER"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["NORTH"],["WATER"],["WATER"],["NORTH"],["WATER"],["WEST"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["EAST"],["EAST"],["WATER"],["EAST"],["WEST"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["EAST"],["EAST"],["EAST"],["EAST"],["EAST"],["WATER"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["SOUTH"],["SOUTH"],["EAST"],["EAST"],["PASS"],["PASS"]],"market":[]},{"farmer":["WATER"],"hands":[["SOUTH"],["WATER"],["SOUTH"],["SOUTH"],["EAST"],["PASS"],["PASS"]],"market":[]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["PASS"],["WATER"],["WATER"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["WATER"],"hands":[["PASS"],["PASS"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["WATER"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","WHEAT",3],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"]]},{"farmer":["FEED"],"hands":[["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["WEST"],["WEST"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["WEST"],["NORTH"],["NORTH"],["CARE"],["WEST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["NORTH"],["WEST"],["NORTH"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["PASS"],"hands":[["WEST"],["NORTH"],["WATER"],["NORTH"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["WATER"],["WEST"],["NORTH"],["WEST"],["WATER"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["NORTH"],["WEST"],["FERTILIZE"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["WATER"],["NORTH"],["WATER"],["HARVEST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["HARVEST"],["WEST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["NORTH"],"hands":[["WATER"],["WATER"],["NORTH"],["WEST"],["NORTH"],["NORTH"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WATER"],"hands":[["WEST"],["EAST"],["HARVEST"],["WATER"],["HARVEST"],["NORTH"],["WATER"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["EAST"],["PLANT","WHEAT"],["EAST"],["PLANT","WHEAT"],["HARVEST"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["WEST"],"hands":[["NORTH"],["WATER"],["WATER"],["SOUTH"],["WATER"],["PLANT","WHEAT"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WATER"],"hands":[["HARVEST"],["SOUTH"],["EAST"],["SOUTH"],["EAST"],["WATER"],["WATER"]],"market":[]},{"farmer":["PASS"],"hands":[["PLANT","WHEAT"],["WATER"],["EAST"],["WATER"],["EAST"],["EAST"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["WATER"],["SOUTH"],["PASS"],["SOUTH"],["EAST"],["EAST"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["WATER"],["PASS"],["WATER"],["EAST"],["EAST"],["PASS"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["SOUTH"],["EAST"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["SOUTH"],["SOUTH"],["PASS"]],"market":[]},{"farmer":["WATER"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["DROP"],["SOUTH"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["SOUTH"],["PASS"]],"market":[["SELL","WHEAT",4]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["DROP"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[["SELL","WHEAT",6]]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","WHEAT",1]]},{"farmer":["PICKUP","WHEAT",10],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"],["SELL","WHEAT",1]]},{"farmer":["FEED"],"hands":[["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["WEST"],["WEST"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["WEST"],["NORTH"],["WEST"],["CARE"],["EAST"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["WEST"],["WEST"],["WEST"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["NORTH"],["WEST"],["NORTH"],["WATER"],["NORTH"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["WATER"],["WEST"],["FERTILIZE"],["NORTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WATER"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["WATER"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["WATER"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["NORTH"],["WATER"],["NORTH"],["WATER"],["EAST"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["NORTH"],["NORTH"],["NORTH"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["WEST"],["NORTH"],["WATER"],["WATER"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WEST"],["WATER"],["WEST"],["WEST"],["WEST"],["WEST"],["WATER"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WATER"],["WEST"],["SOUTH"],["WEST"],["PASS"],["WEST"],["PASS"]],"market":[]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["WATER"],["WATER"],["PASS"],["PASS"],["WEST"],["PASS"]],"market":[]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["PASS"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["DROP"],"hands":[["WATER"],["PASS"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["WATER"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[["SELL","WHEAT",3]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[["SELL","WHEAT",1]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[["SELL","WHEAT",1]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[["SELL","WHEAT",1]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[["SELL","WHEAT",1]]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","COW",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","WHEAT",3],"hands":[["PICKUP","WHEAT",3],["PICKUP","WHEAT",3],["PICKUP","WHEAT",3],["PICKUP","WHEAT",3],["PICKUP","WHEAT",3],["PICKUP","WHEAT",3],["PICKUP","WHEAT",3],["PICKUP","WHEAT",3]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["CARE"],"hands":[["CARE"],["WEST"],["NORTH"],["WEST"],["CARE"],["WEST"],["NORTH"],["CARE"]],"market":[]},{"farmer":["DROP"],"hands":[["NORTH"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["WEST"],["WEST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["NORTH"],["HARVEST"],["NORTH"],["HARVEST"],["NORTH"],["WEST"],["NORTH"]],"market":[["SELL","MILK",6]]},{"farmer":["BUILD_PASTURE"],"hands":[["NORTH"],["WATER"],["BUILD_PASTURE"],["NORTH"],["WEST"],["WATER"],["HARVEST"],["HARVEST"]],"market":[["BUY_LAND"],["BUY_SEED","WHEAT",2]]},{"farmer":["NORTH"],"hands":[["SOUTH"],["EAST"],["NORTH"],["FERTILIZE"],["PLANT","WHEAT"],["NORTH"],["NORTH"],["NORTH"]],"market":[["BUY_SEED","WHEAT",2],["BUY_SEED","STRAWBERRY",5],["BUY_SEED","MELON",1]]},{"farmer":["EAST"],"hands":[["SOUTH"],["SOUTH"],["PLANT","MELON"],["WATER"],["WATER"],["WATER"],["EAST"],["NORTH"]],"market":[["BUY_SEED","STRAWBERRY",5]]},{"farmer":["EAST"],"hands":[["PLACE","COW"],["PLACE","COW"],["WATER"],["EAST"],["NORTH"],["NORTH"],["HARVEST"],["WATER"]],"market":[]},{"farmer":["BUILD_PASTURE"],"hands":[["SOUTH"],["EAST"],["EAST"],["EAST"],["EAST"],["EAST"],["BUILD_PASTURE"],["EAST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["SOUTH"],["SOUTH"],["EAST"],["EAST"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["PLANT","STRAWBERRY"],"hands":[["WEST"],["EAST"],["EAST"],["PLANT","STRAWBERRY"],["FEED"],["HARVEST"],["WEST"],["PLANT","STRAWBERRY"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["WATER"],["WEST"],["WATER"],["CARE"],["EAST"],["SOUTH"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["PLACE","COW"],["EAST"],["EAST"],["EAST"],["NORTH"],["WEST"],["FEED"],["NORTH"]],"market":[]},{"farmer":["PLANT","STRAWBERRY"],"hands":[["NORTH"],["EAST"],["PLANT","STRAWBERRY"],["PLANT","STRAWBERRY"],["NORTH"],["SOUTH"],["CARE"],["PLANT","STRAWBERRY"]],"market":[]},{"farmer":["WATER"],"hands":[["PLANT","STRAWBERRY"],["PLANT","STRAWBERRY"],["WATER"],["WATER"],["HARVEST"],["FEED"],["WEST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["WATER"],["EAST"],["NORTH"],["PLANT","STRAWBERRY"],["CARE"],["NORTH"],["EAST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["WEST"],["NORTH"],["PLANT","WHEAT"],["PLANT","WHEAT"],["WATER"],["NORTH"],["NORTH"],["PLANT","WHEAT"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["HARVEST"],["WATER"],["NORTH"],["WATER"],["WEST"],["WEST"],["NORTH"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["EAST"],["PLANT","WHEAT"],["EAST"],["NORTH"],["WEST"],["HARVEST"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["NORTH"],["PLANT","WHEAT"],["WATER"],["PLANT","WHEAT"],["HARVEST"],["HARVEST"],["WATER"],["PLANT","WHEAT"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["NORTH"],["WATER"],["EAST"],["WATER"],["WATER"],["WATER"],["WEST"],["WATER"]],"market":[]},{"farmer":["PICKUP","WHEAT",14],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","TOMATO",4],["SELL","WHEAT",4],["BUY_SEED","WHEAT",1]]},{"farmer":["FEED"],"hands":[["WATER"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["WATER"]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_SEED","STRAWBERRY",2]]},{"farmer":["CARE"],"hands":[["WEST"],["WEST"],["WEST"],["CARE"],["WEST"],["NORTH"],["WEST"],["NORTH"],["WEST"],["CARE"]],"market":[["HIRE"],["HIRE"]]},{"farmer":["NORTH"],"hands":[["WEST"],["NORTH"],["NORTH"],["WEST"],["WEST"],["NORTH"],["EAST"],["WEST"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["NORTH"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["EAST"],["NORTH"],["EAST"],["NORTH"],["WATER"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["WATER"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["EAST"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["WEST"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["NORTH"],["EAST"],["NORTH"],["WATER"],["WATER"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["WATER"],["WATER"],["WATER"],["WATER"],["EAST"],["WATER"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WATER"],["WEST"],["EAST"],["WEST"],["WEST"],["WATER"],["EAST"],["WEST"],["WATER"],["WEST"],["WEST"]],"market":[]},{"farmer":["FEED"],"hands":[["NORTH"],["WATER"],["WATER"],["FERTILIZE"],["NORTH"],["WEST"],["EAST"],["WEST"],["EAST"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["CARE"],"hands":[["NORTH"],["EAST"],["WEST"],["NORTH"],["HARVEST"],["WEST"],["DIG"],["WEST"],["EAST"],["WATER"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["NORTH"],["WEST"],["FERTILIZE"],["WATER"],["WEST"],["PLANT","STRAWBERRY"],["FERTILIZE"],["PLANT","STRAWBERRY"],["WEST"],["NORTH"]],"market":[]},{"farmer":["DROP"],"hands":[["NORTH"],["NORTH"],["HARVEST"],["HARVEST"],["WEST"],["WEST"],["WATER"],["NORTH"],["WATER"],["EAST"],["HARVEST"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["WATER"],["WATER"],["WATER"],["HARVEST"],["WEST"],["NORTH"],["FERTILIZE"],["NORTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["NORTH"],["EAST"],["EAST"],["EAST"],["WATER"],["EAST"],["PLANT","WHEAT"],["HARVEST"],["NORTH"],["EAST"],["SOUTH"]],"market":[["SELL","WHEAT",2]]},{"farmer":["EAST"],"hands":[["WATER"],["EAST"],["EAST"],["EAST"],["EAST"],["EAST"],["WATER"],["WATER"],["WEST"],["NORTH"],["SOUTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["PASS"],["EAST"],["EAST"],["EAST"],["PASS"],["EAST"],["WEST"],["PASS"],["WATER"],["WATER"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["PASS"],["PASS"],["WATER"],["EAST"],["PASS"],["PASS"],["WATER"],["PASS"],["SOUTH"],["EAST"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["SOUTH"],["PASS"],["WATER"],["SOUTH"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["WATER"],["PASS"],["PASS"],["WATER"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","TOMATO",8],["SELL","FERTILIZER",1]]},{"farmer":["PICKUP","WHEAT",12],"hands":[["PICKUP","WHEAT",12],["PICKUP","WHEAT",12],["PICKUP","WHEAT",12],["PICKUP","WHEAT",12],["PICKUP","WHEAT",12]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_SEED","MELON",1],["BUY_ANIMAL","COW",1]]},{"farmer":["PICKUP","COW",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"]]},{"farmer":["EAST"],"hands":[["WATER"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["WATER"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["WATER"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["WEST"],["NORTH"],["WEST"],["EAST"],["WEST"],["EAST"],["NORTH"],["NORTH"],["NORTH"]],"market":[["SELL","MILK",5]]},{"farmer":["PLACE","COW"],"hands":[["WATER"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["WEST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["FEED"],"hands":[["NORTH"],["NORTH"],["WATER"],["WEST"],["WATER"],["NORTH"],["WATER"],["NORTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["NORTH"],["WATER"],["NORTH"],["WATER"],["EAST"],["NORTH"],["EAST"],["NORTH"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["NORTH"],["WATER"],["NORTH"],["EAST"],["HARVEST"],["EAST"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["HARVEST"],["NORTH"],["WATER"],["NORTH"],["NORTH"],["WATER"],["WATER"],["WATER"],["WATER"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["NORTH"],["WATER"],["EAST"],["WATER"],["HARVEST"],["WEST"],["SOUTH"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["HARVEST"],["WEST"],["SOUTH"],["WEST"],["WATER"],["WATER"],["WATER"],["HARVEST"],["WATER"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["WATER"],["WEST"],["FERTILIZE"],["WATER"],["NORTH"],["NORTH"],["EAST"],["WATER"],["WEST"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["NORTH"],["WEST"],["WATER"],["NORTH"],["HARVEST"],["WATER"],["EAST"],["WEST"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WEST"],["HARVEST"],["WEST"],["WEST"],["WATER"],["EAST"],["WEST"],["FERTILIZE"],["WEST"],["HARVEST"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["WATER"],["HARVEST"],["WEST"],["NORTH"],["EAST"],["WATER"],["WATER"],["WEST"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["NORTH"],["WEST"],["WATER"],["HARVEST"],["WATER"],["EAST"],["EAST"],["NORTH"],["HARVEST"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["HARVEST"],["WEST"],["PASS"],["EAST"],["PASS"],["SOUTH"],["EAST"],["WATER"],["EAST"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["EAST"],["SOUTH"],["PASS"],["EAST"],["PASS"],["SOUTH"],["WATER"],["PASS"],["EAST"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["EAST"],["EAST"],["EAST"],["EAST"],["WEST"],["SOUTH"],["WEST"],["WEST"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["DROP"],"hands":[["PASS"],["WEST"],["PASS"],["PASS"],["PASS"],["DROP"],["PASS"],["PASS"],["PASS"],["EAST"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["EAST"]],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",2],["SELL","TOMATO",2]]},{"farmer":["WEST"],"hands":[["PASS"],["SOUTH"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["EAST"]],"market":[["BUY_SEED","WHEAT",5]]},{"farmer":["PASS"],"hands":[["WEST"],["HARVEST"],["WEST"],["WEST"],["EAST"],["EAST"],["NORTH"],["EAST"],["WEST"],["FERTILIZE"]],"market":[]},{"farmer":["PICKUP","WHEAT",14],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","TOMATO",7],["SELL","WHEAT",4],["BUY_SEED","WHEAT",1]]},{"farmer":["FEED"],"hands":[["PICKUP","FERTILIZER",1],["PICKUP","FERTILIZER",1],["PICKUP","FERTILIZER",1],["PICKUP","FERTILIZER",1],["PICKUP","FERTILIZER",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["CARE"],"hands":[["WATER"],["NORTH"],["NORTH"],["CARE"],["WATER"],["NORTH"],["NORTH"],["CARE"],["WATER"],["NORTH"]],"market":[["HIRE"],["HIRE"]]},{"farmer":["NORTH"],"hands":[["EAST"],["COLLECT_FERTILIZER"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["COLLECT_FERTILIZER"],["NORTH"],["COLLECT_FERTILIZER"],["EAST"],["COLLECT_FERTILIZER"],["NORTH"],["WEST"]],"market":[["BUY_PRODUCT","WHEAT",4]]},{"farmer":["FEED"],"hands":[["WATER"],["WEST"],["WEST"],["NORTH"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["WEST"],["EAST"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["EAST"],["WEST"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["NORTH"],["NORTH"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["NORTH"],["WEST"],["NORTH"],["WEST"],["NORTH"],["WATER"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["EAST"],["WATER"],["WATER"],["WATER"],["WEST"],["WATER"],["NORTH"],["NORTH"],["WATER"],["NORTH"],["NORTH"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["EAST"],["WEST"],["NORTH"],["WEST"],["EAST"],["WEST"],["WATER"],["WATER"],["EAST"],["WATER"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WATER"],["WATER"],["NORTH"],["WEST"],["EAST"],["HARVEST"],["WEST"],["WEST"],["EAST"],["EAST"],["WATER"],["WATER"]],"market":[["SELL","MELON",1]]},{"farmer":["FEED"],"hands":[["NORTH"],["WEST"],["HARVEST"],["HARVEST"],["NORTH"],["WATER"],["SOUTH"],["HARVEST"],["WATER"],["SOUTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["FERTILIZE"],["PLANT","WHEAT"],["WATER"],["WATER"],["COLLECT_FERTILIZER"],["NORTH"],["WATER"],["WATER"],["NORTH"],["SOUTH"],["WEST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["WATER"],["NORTH"],["NORTH"],["EAST"],["WEST"],["EAST"],["WEST"],["PLANT","WHEAT"],["COLLECT_FERTILIZER"],["EAST"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["NORTH"],["FERTILIZE"],["NORTH"],["HARVEST"],["EAST"],["PLANT","WHEAT"],["EAST"],["WEST"],["WATER"],["EAST"],["EAST"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["PLANT","WHEAT"],["NORTH"],["HARVEST"],["WATER"],["EAST"],["WATER"],["EAST"],["WEST"],["NORTH"],["EAST"],["EAST"],["NORTH"]],"market":[]},{"farmer":["FEED"],"hands":[["WATER"],["PLANT","WHEAT"],["WATER"],["SOUTH"],["EAST"],["WEST"],["FERTILIZE"],["PLANT","WHEAT"],["EAST"],["EAST"],["WATER"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["WATER"],["PASS"],["SOUTH"],["EAST"],["PASS"],["WATER"],["WATER"],["PASS"],["EAST"],["EAST"],["WATER"]],"market":[]},{"farmer":["SOUTH"],"hands":[["PASS"],["PASS"],["PASS"],["SOUTH"],["FERTILIZE"],["PASS"],["PASS"],["PASS"],["PASS"],["SOUTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["DROP"],"hands":[["PASS"],["PASS"],["PASS"],["SOUTH"],["WEST"],["PASS"],["PASS"],["PASS"],["PASS"],["FERTILIZE"],["PASS"],["WATER"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["WATER"],["NORTH"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["NORTH"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["FERTILIZE"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["PASS"],"hands":[["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"],["PASS"]],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","TOMATO",9]]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["PICKUP","WHEAT",14],["WEST"],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_SEED","TOMATO",5]]},{"farmer":["WEST"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WEST"],["WATER"],["WEST"],["COLLECT_FERTILIZER"],["WATER"],["EAST"],["NORTH"]],"market":[["HIRE"],["HIRE"],["BUY_SEED","TOMATO",1],["BUY_PRODUCT","WHEAT",5]]},{"farmer":["NORTH"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WEST"],["EAST"],["WEST"],["NORTH"],["NORTH"],["EAST"],["EAST"],["WEST"],["WEST"]],"market":[]},{"farmer":["HARVEST"],"hands":[["NORTH"],["NORTH"],["FEED"],["HARVEST"],["WATER"],["WEST"],["NORTH"],["NORTH"],["EAST"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["PLANT","MELON"],"hands":[["NORTH"],["NORTH"],["CARE"],["NORTH"],["EAST"],["NORTH"],["FERTILIZE"],["WATER"],["NORTH"],["WATER"],["NORTH"],["NORTH"]],"market":[["BUY_SEED","MELON",2]]},{"farmer":["WATER"],"hands":[["HARVEST"],["HARVEST"],["WEST"],["HARVEST"],["WATER"],["PLANT","MELON"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["HARVEST"],["COLLECT_FERTILIZER"]],"market":[["BUY_SEED","MELON",1],["BUY_ANIMAL","COW",1]]},{"farmer":["EAST"],"hands":[["BUILD_PASTURE"],["BUILD_PASTURE"],["SOUTH"],["WEST"],["WEST"],["WATER"],["SOUTH"],["SOUTH"],["WEST"],["WEST"],["BUILD_PASTURE"],["EAST"]],"market":[["BUY_SEED","WHEAT",1],["BUY_SEED","MELON",2],["BUY_SEED","MELON",1]]},{"farmer":["EAST"],"hands":[["SOUTH"],["EAST"],["PICKUP","COW",1],["EAST"],["WEST"],["EAST"],["SOUTH"],["SOUTH"],["WEST"],["SOUTH"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WATER"],["EAST"],["WEST"],["WATER"],["EAST"],["WEST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["EAST"],["EAST"],["FEED"],["WEST"]],"market":[]},{"farmer":["DROP"],"hands":[["WEST"],["WATER"],["WEST"],["NORTH"],["NORTH"],["WEST"],["WEST"],["NORTH"],["NORTH"],["EAST"],["CARE"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["EAST"],["PLACE","COW"],["DIG"],["EAST"],["PLANT","MELON"],["WEST"],["NORTH"],["WATER"],["HARVEST"],["NORTH"],["NORTH"]],"market":[["SELL","MILK",3],["SELL","MELON",6]]},{"farmer":["WEST"],"hands":[["DIG"],["HARVEST"],["FEED"],["PLANT","MELON"],["PLANT","MELON"],["WATER"],["WEST"],["NORTH"],["EAST"],["WEST"],["FEED"],["WATER"]],"market":[["BUY_SEED","WHEAT",1],["BUY_ANIMAL","COW",1],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["EAST"],"hands":[["PLANT","MELON"],["PLANT","MELON"],["EAST"],["WATER"],["WATER"],["EAST"],["FERTILIZE"],["SOUTH"],["WEST"],["WEST"],["EAST"],["EAST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["WATER"],["WATER"],["EAST"],["EAST"],["WEST"],["EAST"],["EAST"],["SOUTH"],["WEST"],["SOUTH"],["SOUTH"],["SOUTH"]],"market":[]},{"farmer":["PICKUP","COW",1],"hands":[["EAST"],["WEST"],["PICKUP","COW",1],["EAST"],["WEST"],["EAST"],["EAST"],["SOUTH"],["SOUTH"],["PICKUP","COW",1],["PICKUP","COW",1],["SOUTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["WEST"],["FEED"],["EAST"],["EAST"],["WEST"],["CARE"],["EAST"],["EAST"],["WEST"],["FEED"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WATER"],["SOUTH"],["CARE"],["SOUTH"],["WEST"],["CARE"],["EAST"],["WEST"],["WEST"],["CARE"],["CARE"],["SOUTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["SOUTH"],["EAST"],["SOUTH"],["EAST"],["NORTH"],["NORTH"],["EAST"],["EAST"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["NORTH"],"hands":[["HARVEST"],["FEED"],["EAST"],["DROP"],["EAST"],["NORTH"],["WATER"],["EAST"],["EAST"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["PLACE","COW"],"hands":[["EAST"],["CARE"],["EAST"],["WEST"],["NORTH"],["DIG"],["EAST"],["EAST"],["EAST"],["EAST"],["NORTH"],["EAST"]],"market":[["SELL","STRAWBERRY",2],["SELL","MELON",6],["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["WEST"],["SOUTH"],["EAST"],["WEST"],["HARVEST"],["PLANT","TOMATO"],["EAST"],["HARVEST"],["NORTH"],["EAST"],["NORTH"],["EAST"]],"market":[["BUY_LAND"],["BUY_ANIMAL","COW",4],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["WEST"],"hands":[["BUILD_PASTURE"],["PICKUP","COW",1],["WEST"],["SOUTH"],["BUILD_PASTURE"],["WEST"],["WEST"],["BUILD_PASTURE"],["EAST"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["PASS"],"hands":[["WEST"],["WEST"],["WEST"],["EAST"],["WEST"],["EAST"],["SOUTH"],["WEST"],["WEST"],["WEST"],["SOUTH"],["WEST"]],"market":[["BUY_SEED","WHEAT",4]]},{"farmer":["PICKUP","COW",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MELON",1],["SELL","WHEAT",18],["BUY_SEED","TOMATO",1]]},{"farmer":["WEST"],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MELON",1],["BUY_SEED","STRAWBERRY",5]]},{"farmer":["NORTH"],"hands":[["EAST"],["PICKUP","WHEAT",14],["EAST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["SELL","MELON",1],["BUY_SEED","STRAWBERRY",7]]},{"farmer":["NORTH"],"hands":[["EAST"],["EAST"],["EAST"],["FEED"],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3],["PICKUP","FERTILIZER",3]],"market":[["SELL","MELON",1],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["NORTH"],"hands":[["NORTH"],["EAST"],["EAST"],["NORTH"],["WATER"],["CARE"],["NORTH"],["EAST"],["CARE"],["WATER"],["NORTH"],["WATER"]],"market":[["SELL","MELON",1],["BUY_PRODUCT","WHEAT",6]]},{"farmer":["PLACE","COW"],"hands":[["NORTH"],["NORTH"],["NORTH"],["FEED"],["EAST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["NORTH"],["EAST"]],"market":[["SELL","MELON",1],["BUY_SEED","WHEAT",3]]},{"farmer":["EAST"],"hands":[["PLACE","COW"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["NORTH"],["WEST"],["EAST"],["WEST"],["EAST"],["NORTH"],["NORTH"]],"market":[["SELL","MELON",1],["BUY_SEED","STRAWBERRY",5]]},{"farmer":["PASS"],"hands":[["EAST"],["NORTH"],["PLACE","COW"],["NORTH"],["NORTH"],["CARE"],["COLLECT_FERTILIZER"],["EAST"],["NORTH"],["WATER"],["WATER"],["WATER"]],"market":[["SELL","MELON",1],["BUY_PRODUCT","WHEAT",2]]},{"farmer":["WATER"],"hands":[["WATER"],["NORTH"],["EAST"],["NORTH"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["WEST"]],"market":[["SELL","MELON",1],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["EAST"],"hands":[["NORTH"],["NORTH"],["HARVEST"],["FEED"],["WATER"],["WEST"],["COLLECT_FERTILIZER"],["WATER"],["WATER"],["WATER"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[["SELL","MELON",1],["BUY_SEED","STRAWBERRY",5]]},{"farmer":["WATER"],"hands":[["WATER"],["PLACE","COW"],["WEST"],["CARE"],["WEST"],["WEST"],["WEST"],["NORTH"],["EAST"],["WEST"],["WATER"],["WEST"]],"market":[["SELL","MELON",1],["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["EAST"],["FEED"],["FEED"],["WEST"],["EAST"],["WEST"],["WATER"],["NORTH"],["WATER"],["WEST"],["WEST"],["WEST"]],"market":[["SELL","MELON",1],["BUY_SEED","STRAWBERRY",4],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["WATER"],"hands":[["PLANT","WHEAT"],["CARE"],["CARE"],["SOUTH"],["EAST"],["WATER"],["SOUTH"],["WATER"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["WATER"],["WEST"],["WEST"],["FEED"],["EAST"],["WEST"],["COLLECT_FERTILIZER"],["SOUTH"],["WEST"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["DIG"],"hands":[["NORTH"],["SOUTH"],["NORTH"],["CARE"],["NORTH"],["WATER"],["WEST"],["PLANT","WHEAT"],["WEST"],["WEST"],["WATER"],["WEST"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["PLANT","WHEAT"],["SOUTH"],["FEED"],["SOUTH"],["FERTILIZE"],["NORTH"],["WATER"],["WATER"],["WATER"],["WEST"],["WEST"],["SOUTH"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["SOUTH"],["CARE"],["SOUTH"],["EAST"],["FERTILIZE"],["SOUTH"],["WEST"],["NORTH"],["EAST"],["WEST"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["FEED"],["WEST"],["FEED"],["SOUTH"],["WATER"],["PLANT","WHEAT"],["SOUTH"],["WATER"],["NORTH"],["WEST"],["SOUTH"]],"market":[]},{"farmer":["PASS"],"hands":[["DIG"],["CARE"],["WEST"],["CARE"],["FERTILIZE"],["EAST"],["WATER"],["DIG"],["WEST"],["DIG"],["FERTILIZE"],["PLANT","WHEAT"]],"market":[]},{"farmer":["DIG"],"hands":[["PLANT","WHEAT"],["WEST"],["WEST"],["SOUTH"],["NORTH"],["NORTH"],["FERTILIZE"],["PLANT","WHEAT"],["PLANT","WHEAT"],["PLANT","WHEAT"],["WATER"],["WATER"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["WATER"],["WEST"],["SOUTH"],["FEED"],["FERTILIZE"],["FERTILIZE"],["EAST"],["WATER"],["WATER"],["WATER"],["EAST"],["FERTILIZE"]],"market":[["BUY_ANIMAL","COW",1]]},{"farmer":["WATER"],"hands":[["WEST"],["EAST"],["SOUTH"],["SOUTH"],["WEST"],["SOUTH"],["BUILD_PASTURE"],["WEST"],["SOUTH"],["SOUTH"],["SOUTH"],["SOUTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["SOUTH"],["PICKUP","COW",1],["EAST"],["WEST"],["EAST"],["EAST"],["WEST"],["EAST"],["EAST"],["EAST"],["EAST"]],"market":[["HIRE"]]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["WEST"],["PLANT","WHEAT"],["WEST"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["SOUTH"],["DIG"],["WEST"]],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["PICKUP","COW",1],"hands":[["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1],["PICKUP","COW",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_ANIMAL","GOOSE",1]]},{"farmer":["WEST"],"hands":[["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["WEST"],"hands":[["WEST"],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2],["PICKUP","FERTILIZER",2]],"market":[]},{"farmer":["SOUTH"],"hands":[["WEST"],["NORTH"],["NORTH"],["EAST"],["WATER"],["DIG"],["NORTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["PLACE","COW"],"hands":[["WEST"],["FEED"],["EAST"],["WEST"],["WEST"],["PLANT","STRAWBERRY"],["NORTH"],["EAST"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["FEED"],["SOUTH"],["WATER"],["SOUTH"],["SOUTH"],["WATER"],["WEST"],["WEST"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["CARE"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["SOUTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["CARE"],["WATER"],["CARE"],["CARE"],["CARE"],["CARE"],["CARE"],["CARE"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["FEED"],["NORTH"],["NORTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["NORTH"],"hands":[["CARE"],["FEED"],["WATER"],["WEST"],["NORTH"],["WEST"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["NORTH"],["EAST"],["WEST"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["EAST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["FEED"],["FEED"],["WATER"],["WEST"],["WATER"],["NORTH"],["EAST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["EAST"],"hands":[["CARE"],["CARE"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["NORTH"],["EAST"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["NORTH"],["EAST"],["WATER"],["HARVEST"],["WATER"],["WATER"],["EAST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["NORTH"],["EAST"],["EAST"],["WATER"],["NORTH"],["SOUTH"],["EAST"],["WATER"],["EAST"]],"market":[]},{"farmer":["DROP"],"hands":[["FEED"],["NORTH"],["WATER"],["NORTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["WATER"],["EAST"],["WATER"]],"market":[]},{"farmer":["PICKUP","FERTILIZER",3],"hands":[["CARE"],["FEED"],["WEST"],["WATER"],["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"]],"market":[["SELL","MILK",3]]},{"farmer":["WEST"],"hands":[["EAST"],["CARE"],["WEST"],["EAST"],["WATER"],["WATER"],["NORTH"],["EAST"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",3],["BUY_ANIMAL","GOOSE",1],["BUY_PRODUCT","WHEAT",10]]},{"farmer":["PICKUP","GOOSE",1],"hands":[["NORTH"],["EAST"],["WEST"],["NORTH"],["SOUTH"],["EAST"],["WEST"],["FERTILIZE"],["WATER"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["WEST"],"hands":[["FEED"],["SOUTH"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"],["WEST"],["WEST"],["NORTH"],["EAST"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["CARE"],["FEED"],["WEST"],["WATER"],["SOUTH"],["SOUTH"],["COLLECT_FERTILIZER"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["BUILD_COOP"],"hands":[["EAST"],["CARE"],["FERTILIZE"],["NORTH"],["FERTILIZE"],["COLLECT_FERTILIZER"],["NORTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["PASS"],"hands":[["EAST"],["COLLECT_FERTILIZER"],["WEST"],["DIG"],["WEST"],["WEST"],["FERTILIZE"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["PICKUP","WHEAT",14],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","STRAWBERRY",2],["SELL","FERTILIZER",1]]},{"farmer":["FEED"],"hands":[["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["BUY_ANIMAL","GOOSE",1],["BUY_PRODUCT","WHEAT",2]]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["PICKUP","GOOSE",1],["CARE"],["WATER"],["WATER"],["PICKUP","GOOSE",1],["CARE"],["WATER"],["WATER"]],"market":[["HIRE"],["HIRE"]]},{"farmer":["WEST"],"hands":[["NORTH"],["WEST"],["WEST"],["WEST"],["EAST"],["NORTH"],["EAST"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["NORTH"],["WEST"],["SOUTH"],["WEST"],["WATER"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["FERTILIZE"],["NORTH"],["BUILD_COOP"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["EAST"],["COLLECT_FERTILIZER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["WATER"],["WATER"],["PLACE","GOOSE"],["WEST"],["WATER"],["NORTH"],["WATER"],["NORTH"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["NORTH"],["WEST"],["WATER"],["NORTH"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["FEED"],"hands":[["WATER"],["WATER"],["NORTH"],["NORTH"],["WATER"],["WATER"],["WATER"],["WEST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["HARVEST"],["EAST"],["WEST"],["EAST"],["WEST"],["WATER"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["COLLECT_FERTILIZER"],["NORTH"],["WEST"],["EAST"],["EAST"],["WATER"],["WATER"],["FERTILIZE"],["EAST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["HARVEST"],["COLLECT_FERTILIZER"],["EAST"],["WATER"],["WEST"],["NORTH"],["SOUTH"],["COLLECT_FERTILIZER"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["WATER"],["EAST"],["WEST"],["EAST"],["NORTH"],["WATER"],["WATER"],["EAST"],["EAST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["WEST"],["EAST"],["WATER"],["FEED"],["HARVEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WATER"],["EAST"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["EAST"],["WEST"],["CARE"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["EAST"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["NORTH"],["WATER"],["EAST"],["WEST"],["FERTILIZE"],["EAST"],["WATER"],["FERTILIZE"],["WATER"]],"market":[]},{"farmer":["CARE"],"hands":[["HARVEST"],["FEED"],["SOUTH"],["EAST"],["NORTH"],["WATER"],["FERTILIZE"],["WEST"],["WATER"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["CARE"],["PLANT","WHEAT"],["FEED"],["FEED"],["EAST"],["NORTH"],["WEST"],["SOUTH"],["FERTILIZE"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["COLLECT_FERTILIZER"],["WATER"],["CARE"],["CARE"],["HARVEST"],["HARVEST"],["PLANT","WHEAT"],["PLANT","WHEAT"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["PLANT","WHEAT"],["SOUTH"],["FERTILIZE"],["EAST"],["SOUTH"],["NORTH"],["WEST"],["WATER"],["WATER"],["WEST"]],"market":[]},{"farmer":["FEED"],"hands":[["WATER"],["SOUTH"],["EAST"],["EAST"],["WATER"],["DIG"],["WEST"],["NORTH"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["SOUTH"],["PLANT","STRAWBERRY"],["EAST"],["EAST"],["PLANT","STRAWBERRY"],["FEED"],["PLANT","STRAWBERRY"],["SOUTH"],["FERTILIZE"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["SOUTH"],["WATER"],["FEED"],["SOUTH"],["WEST"],["CARE"],["WATER"],["PLANT","STRAWBERRY"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["FEED"],["EAST"],["CARE"],["SOUTH"],["PLANT","STRAWBERRY"],["WEST"],["SOUTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",2]]},{"farmer":["HARVEST"],"hands":[["WEST"],["WEST"],["NORTH"],["WEST"],["WEST"]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_SEED","WHEAT",5]]},{"farmer":["WEST"],"hands":[["HARVEST"],["HARVEST"],["HARVEST"],["NORTH"],["EAST"],["WEST"],["WEST"],["WEST"],["NORTH"],["WEST"]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_SEED","WHEAT",2]]},{"farmer":["PASS"],"hands":[["EAST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["HARVEST"],"hands":[["PICKUP","FERTILIZER",8],["FEED"],["FEED"],["FEED"],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8]],"market":[["BUY_PRODUCT","WHEAT",4]]},{"farmer":["EAST"],"hands":[["CARE"],["CARE"],["CARE"],["CARE"],["CARE"],["CARE"],["WATER"],["CARE"],["CARE"],["WATER"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["DROP"],["NORTH"],["WEST"],["WEST"],["WEST"],["NORTH"],["NORTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["NORTH"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["PICKUP","FERTILIZER",8],["FEED"],["FEED"],["NORTH"],["EAST"],["SOUTH"],["PICKUP","FERTILIZER",8],["EAST"],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["WATER"]],"market":[["SELL","MILK",6]]},{"farmer":["EAST"],"hands":[["EAST"],["CARE"],["CARE"],["FEED"],["WEST"],["NORTH"],["NORTH"],["FERTILIZE"],["WEST"],["EAST"],["NORTH"]],"market":[["BUY_ANIMAL","SHEEP",3]]},{"farmer":["WATER"],"hands":[["PICKUP","SHEEP",1],["SOUTH"],["EAST"],["EAST"],["EAST"],["SOUTH"],["SOUTH"],["PICKUP","SHEEP",1],["EAST"],["PICKUP","SHEEP",1],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["EAST"],["WEST"],["WEST"],["EAST"],["WEST"],["WEST"],["NORTH"],["EAST"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["FERTILIZE"],["WEST"],["WEST"],["FEED"],["NORTH"],["WEST"],["NORTH"],["EAST"],["COLLECT_FERTILIZER"],["NORTH"],["WATER"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["FEED"],["SOUTH"],["EAST"],["CARE"],["WEST"],["WATER"],["WATER"],["WEST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["FERTILIZE"],["CARE"],["FEED"],["EAST"],["NORTH"],["NORTH"],["NORTH"],["EAST"],["WEST"],["WATER"],["WATER"]],"market":[]},{"farmer":["SOUTH"],"hands":[["EAST"],["EAST"],["CARE"],["NORTH"],["WATER"],["HARVEST"],["WATER"],["EAST"],["FERTILIZE"],["NORTH"],["EAST"]],"market":[]},{"farmer":["DROP"],"hands":[["FERTILIZE"],["NORTH"],["COLLECT_FERTILIZER"],["FEED"],["WEST"],["WATER"],["NORTH"],["WATER"],["WATER"],["WATER"],["EAST"]],"market":[["SELL","MILK",12]]},{"farmer":["NORTH"],"hands":[["SOUTH"],["NORTH"],["WEST"],["CARE"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["EAST"],["EAST"],["WATER"]],"market":[["BUY_SEED","WHEAT",5],["BUY_ANIMAL","SHEEP",2],["BUY_PRODUCT","WHEAT",5]]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["FERTILIZE"],["NORTH"],["EAST"],["WEST"],["NORTH"],["SOUTH"],["WEST"],["NORTH"],["SOUTH"],["WEST"],["WEST"]],"market":[["BUY_SEED","WHEAT",4]]},{"farmer":["WEST"],"hands":[["WEST"],["FEED"],["SOUTH"],["WEST"],["BUILD_PASTURE"],["SOUTH"],["FERTILIZE"],["WATER"],["SOUTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WEST"],["CARE"],["BUILD_PASTURE"],["WEST"],["WEST"],["SOUTH"],["WEST"],["WEST"],["SOUTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WEST"],["EAST"],["WEST"],["WEST"],["SOUTH"],["EAST"],["WEST"],["WEST"],["NORTH"],["WEST"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["NORTH"],["WATER"],["NORTH"],["WATER"],["NORTH"],["WATER"],["COLLECT_FERTILIZER"],["NORTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["WEST"],["FEED"],["NORTH"],["COLLECT_FERTILIZER"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["WEST"],["COLLECT_FERTILIZER"],["SOUTH"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["CARE"],["NORTH"],["EAST"],["WATER"],["WEST"],["SOUTH"],["FERTILIZE"],["NORTH"],["PLACE","SHEEP"],["NORTH"]],"market":[]},{"farmer":["PICKUP","SHEEP",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",3],["SELL","STRAWBERRY",2],["SELL","FERTILIZER",1]]},{"farmer":["PICKUP","WHEAT",14],"hands":[["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["FEED"],"hands":[["PICKUP","FERTILIZER",8],["WEST"],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8]],"market":[["HIRE"],["HIRE"]]},{"farmer":["CARE"],"hands":[["WATER"],["WEST"],["WEST"],["CARE"],["WATER"],["WATER"],["WEST"],["CARE"],["WATER"],["WATER"],["WATER"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["WEST"],["SOUTH"],["NORTH"],["WEST"],["EAST"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["EAST"],"hands":[["WEST"],["PLACE","SHEEP"],["NORTH"],["WEST"],["WATER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["EAST"],["SOUTH"],["WEST"],["WEST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["WEST"],["WATER"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["WEST"],["NORTH"],["WATER"],["EAST"],["NORTH"],["WEST"],["WEST"],["WEST"],["NORTH"],["WEST"],["EAST"]],"market":[["BUY_PRODUCT","WHEAT",15],["SELL","MILK",6]]},{"farmer":["NORTH"],"hands":[["EAST"],["WEST"],["WEST"],["SOUTH"],["NORTH"],["FEED"],["FEED"],["WATER"],["WATER"],["NORTH"],["NORTH"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["NORTH"],["WATER"],["NORTH"],["HARVEST"],["WATER"],["CARE"],["CARE"],["NORTH"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["NORTH"],["WEST"],["NORTH"],["EAST"],["NORTH"],["EAST"],["NORTH"],["WATER"],["WATER"],["WATER"],["WEST"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[["NORTH"],["WATER"],["NORTH"],["FEED"],["WATER"],["FEED"],["FEED"],["NORTH"],["EAST"],["WEST"],["WEST"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["FERTILIZE"],["NORTH"],["WATER"],["CARE"],["EAST"],["CARE"],["CARE"],["WATER"],["WATER"],["HARVEST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["HARVEST"],["WEST"],["NORTH"],["EAST"],["EAST"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["FERTILIZE"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["NORTH"],"hands":[["FERTILIZE"],["EAST"],["WATER"],["FEED"],["WATER"],["EAST"],["NORTH"],["HARVEST"],["WATER"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["EAST"],["WEST"],["CARE"],["EAST"],["NORTH"],["FEED"],["EAST"],["EAST"],["FEED"],["SOUTH"],["HARVEST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["FEED"],"hands":[["FERTILIZE"],["SOUTH"],["WATER"],["COLLECT_FERTILIZER"],["WATER"],["FEED"],["CARE"],["CARE"],["EAST"],["EAST"],["WATER"],["WEST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["CARE"],"hands":[["WEST"],["FEED"],["WEST"],["WEST"],["NORTH"],["CARE"],["EAST"],["COLLECT_FERTILIZER"],["HARVEST"],["EAST"],["EAST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["SOUTH"],["CARE"],["DIG"],["FERTILIZE"],["HARVEST"],["EAST"],["NORTH"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"],["EAST"]],"market":[["BUY_SEED","CARROT",1]]},{"farmer":["WEST"],"hands":[["FERTILIZE"],["EAST"],["PLANT","WHEAT"],["EAST"],["NORTH"],["FERTILIZE"],["FEED"],["SOUTH"],["HARVEST"],["SOUTH"],["WATER"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["NORTH"],["WATER"],["EAST"],["HARVEST"],["NORTH"],["CARE"],["SOUTH"],["WEST"],["FERTILIZE"],["SOUTH"],["FERTILIZE"]],"market":[["BUY_SEED","WHEAT",5]]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["FEED"],["SOUTH"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"],["WATER"],["COLLECT_FERTILIZER"],["SOUTH"],["SOUTH"],["EAST"],["WATER"],["WEST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["FERTILIZE"],"hands":[["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"],["WEST"],["WATER"],["WEST"],["WEST"],["FERTILIZE"],["HARVEST"],["EAST"],["WEST"],["HARVEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["WEST"],["WATER"],["WEST"],["WEST"],["SOUTH"],["WEST"],["SOUTH"],["PLANT","WHEAT"],["EAST"],["DIG"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["EAST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","WHEAT",21],["BUY_SEED","WHEAT",2]]},{"farmer":["NORTH"],"hands":[["HARVEST"],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["HARVEST"],["PICKUP","SHEEP",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"]]},{"farmer":["HARVEST"],"hands":[["NORTH"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["WEST"],["HARVEST"],["HARVEST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["WEST"],["HARVEST"]],"market":[["HIRE"],["HIRE"]]},{"farmer":["EAST"],"hands":[["HARVEST"],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["HARVEST"],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["NORTH"],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8],["PICKUP","FERTILIZER",8]],"market":[["BUY_PRODUCT","WHEAT",15]]},{"farmer":["EAST"],"hands":[["EAST"],["NORTH"],["NORTH"],["NORTH"],["EAST"],["WATER"],["NORTH"],["WEST"],["HARVEST"],["WATER"],["NORTH"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["FEED"],["NORTH"],["NORTH"],["HARVEST"],["EAST"],["FEED"],["NORTH"],["WEST"],["EAST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["HARVEST"],["CARE"],["FEED"],["HARVEST"],["EAST"],["NORTH"],["CARE"],["CARE"],["WATER"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["HARVEST"],"hands":[["EAST"],["NORTH"],["CARE"],["WATER"],["HARVEST"],["NORTH"],["WEST"],["WEST"],["WEST"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["FEED"],["NORTH"],["NORTH"],["WATER"],["HARVEST"],["FEED"],["NORTH"],["NORTH"],["NORTH"],["HARVEST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["CARE"],["NORTH"],["NORTH"],["EAST"],["WATER"],["CARE"],["FEED"],["WATER"],["NORTH"],["WATER"],["WATER"]],"market":[["SELL","MILK",6]]},{"farmer":["SOUTH"],"hands":[["WEST"],["NORTH"],["HARVEST"],["WATER"],["EAST"],["EAST"],["WEST"],["CARE"],["WEST"],["SOUTH"],["WEST"],["EAST"]],"market":[]},{"farmer":["DROP"],"hands":[["NORTH"],["NORTH"],["WATER"],["WEST"],["WATER"],["EAST"],["FEED"],["NORTH"],["WATER"],["COLLECT_FERTILIZER"],["WEST"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["HARVEST"],["NORTH"],["EAST"],["WEST"],["NORTH"],["WATER"],["CARE"],["NORTH"],["SOUTH"],["WEST"],["WEST"],["COLLECT_FERTILIZER"]],"market":[["SELL","STRAWBERRY",2]]},{"farmer":["SOUTH"],"hands":[["WATER"],["FEED"],["NORTH"],["WATER"],["NORTH"],["SOUTH"],["SOUTH"],["FEED"],["SOUTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["SOUTH"],["CARE"],["FEED"],["SOUTH"],["WATER"],["COLLECT_FERTILIZER"],["FEED"],["CARE"],["WATER"],["SOUTH"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["COLLECT_FERTILIZER"],["CARE"],["EAST"],["NORTH"],["NORTH"],["CARE"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["SOUTH"],["EAST"],["EAST"],["EAST"],["PLANT","WHEAT"],["NORTH"],["SOUTH"],["WEST"],["WATER"],["WEST"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["SOUTH"],["EAST"],["SOUTH"],["COLLECT_FERTILIZER"],["WATER"],["FERTILIZE"],["FEED"],["FEED"],["SOUTH"],["NORTH"],["EAST"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["DROP"],["COLLECT_FERTILIZER"],["SOUTH"],["EAST"],["WEST"],["WATER"],["CARE"],["CARE"],["SOUTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["FERTILIZE"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["EAST"],["EAST"],["FEED"],["SOUTH"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["EAST"],["WATER"],["WEST"],["SOUTH"],["WEST"]],"market":[["SELL","MILK",6],["SELL","STRAWBERRY",4]]},{"farmer":["HARVEST"],"hands":[["EAST"],["EAST"],["CARE"],["SOUTH"],["PLANT","WHEAT"],["DIG"],["WEST"],["DIG"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["SOUTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["EAST"],["WATER"],["EAST"],["SOUTH"],["WEST"],["PLANT","WHEAT"],["WEST"],["PLANT","WHEAT"],["PLANT","WHEAT"],["PLANT","WHEAT"],["WEST"],["WEST"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["DIG"],["EAST"],["SOUTH"],["DROP"],["PLANT","WHEAT"],["WATER"],["FERTILIZE"],["WATER"],["WATER"],["WATER"],["WEST"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["PLANT","WHEAT"],["FERTILIZE"],["FEED"],["SOUTH"],["WATER"],["EAST"],["WATER"],["NORTH"],["SOUTH"],["WEST"],["PLANT","WHEAT"],["SOUTH"]],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",2]]},{"farmer":["PICKUP","SHEEP",1],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",3],["SELL","STRAWBERRY",12],["SELL","EGG",1],["SELL","FERTILIZER",5],["SELL","WHEAT",1]]},{"farmer":["PICKUP","WHEAT",14],"hands":[["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_SEED","WHEAT",2]]},{"farmer":["FEED"],"hands":[["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["PICKUP","FERTILIZER",10],"hands":[["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10]],"market":[["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",15]]},{"farmer":["CARE"],"hands":[["NORTH"],["WATER"],["WEST"],["CARE"],["WATER"],["WATER"],["NORTH"],["CARE"],["WATER"],["WATER"],["NORTH"],["CARE"]],"market":[]},{"farmer":["NORTH"],"hands":[["FEED"],["WEST"],["WEST"],["NORTH"],["EAST"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["FEED"],"hands":[["EAST"],["NORTH"],["WEST"],["NORTH"],["WATER"],["COLLECT_FERTILIZER"],["EAST"],["WEST"],["NORTH"],["WEST"],["CARE"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["EAST"],["FEED"],["FEED"],["WATER"],["EAST"],["NORTH"],["WATER"],["NORTH"],["WATER"],["WEST"],["WEST"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["FERTILIZE"],["CARE"],["CARE"],["NORTH"],["EAST"],["NORTH"],["EAST"],["NORTH"],["NORTH"],["NORTH"],["CARE"],["NORTH"]],"market":[]},{"farmer":["FERTILIZE"],"hands":[["NORTH"],["NORTH"],["NORTH"],["WATER"],["EAST"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["WATER"],["COLLECT_FERTILIZER"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["FEED"],["FEED"],["FEED"],["EAST"],["HARVEST"],["WATER"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["EAST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["CARE"],["CARE"],["CARE"],["WATER"],["WATER"],["WEST"],["HARVEST"],["FERTILIZE"],["WATER"],["WATER"],["NORTH"],["EAST"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["EAST"],["SOUTH"],["NORTH"],["NORTH"],["WEST"],["WATER"],["WATER"],["EAST"],["SOUTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["NORTH"],["SOUTH"],["WATER"],["NORTH"],["WATER"],["NORTH"],["WEST"],["SOUTH"],["SOUTH"],["EAST"],["WEST"]],"market":[]},{"farmer":["FEED"],"hands":[["FEED"],["NORTH"],["FEED"],["EAST"],["WATER"],["WEST"],["HARVEST"],["WEST"],["COLLECT_FERTILIZER"],["HARVEST"],["FERTILIZE"],["WEST"]],"market":[]},{"farmer":["CARE"],"hands":[["CARE"],["NORTH"],["CARE"],["WEST"],["NORTH"],["WATER"],["WEST"],["WATER"],["WEST"],["EAST"],["WEST"],["COLLECT_FERTILIZER"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["WEST"],"hands":[["COLLECT_FERTILIZER"],["FEED"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"],["NORTH"],["WEST"],["WEST"],["SOUTH"],["WATER"],["WEST"],["SOUTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["CARE"],["NORTH"],["WEST"],["WATER"],["FERTILIZE"],["NORTH"],["WATER"],["FERTILIZE"],["EAST"],["WEST"],["FERTILIZE"]],"market":[]},{"farmer":["FEED"],"hands":[["SOUTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["SOUTH"],["WEST"],["WATER"],["FEED"],["SOUTH"],["WEST"],["EAST"],["WEST"],["WEST"]],"market":[]},{"farmer":["CARE"],"hands":[["FERTILIZE"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"],["SOUTH"],["SOUTH"],["CARE"],["EAST"],["WEST"],["NORTH"],["WATER"],["WEST"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["WEST"],["EAST"],["COLLECT_FERTILIZER"],["WEST"],["PLANT","WHEAT"],["SOUTH"],["COLLECT_FERTILIZER"],["EAST"],["WEST"],["FEED"],["SOUTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["WEST"],"hands":[["FERTILIZE"],["EAST"],["WEST"],["FERTILIZE"],["WATER"],["HARVEST"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"],["HARVEST"],["SOUTH"],["SOUTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["FERTILIZE"],["NORTH"],["WATER"],["EAST"],["PLANT","WHEAT"],["EAST"],["HARVEST"],["WEST"],["CARE"],["SOUTH"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["FERTILIZE"],"hands":[["DROP"],["WATER"],["FERTILIZE"],["WEST"],["SOUTH"],["WATER"],["FERTILIZE"],["PLANT","WHEAT"],["WEST"],["SOUTH"],["SOUTH"],["FERTILIZE"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","STRAWBERRY",4],["SELL","EGG",1],["SELL","FERTILIZER",1],["SELL","WHEAT",15],["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[["NORTH"],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["HARVEST"]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["HARVEST"],"hands":[["NORTH"],["NORTH"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["NORTH"],["NORTH"],["HARVEST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["WEST"]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["EAST"],"hands":[["HARVEST"],["HARVEST"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["NORTH"],["HARVEST"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["HARVEST"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9]],"market":[["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",15]]},{"farmer":["NORTH"],"hands":[["NORTH"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["EAST"],["NORTH"],["WEST"],["WEST"],["EAST"],["EAST"]],"market":[]},{"farmer":["HARVEST"],"hands":[["NORTH"],["HARVEST"],["FEED"],["NORTH"],["HARVEST"],["NORTH"],["HARVEST"],["FEED"],["NORTH"],["NORTH"],["WATER"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["HARVEST"],["NORTH"],["CARE"],["FEED"],["WATER"],["HARVEST"],["NORTH"],["CARE"],["CARE"],["WATER"],["EAST"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["EAST"],["HARVEST"],["NORTH"],["CARE"],["EAST"],["WATER"],["HARVEST"],["WEST"],["WEST"],["WEST"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["DROP"],"hands":[["HARVEST"],["WATER"],["FEED"],["EAST"],["EAST"],["NORTH"],["NORTH"],["FEED"],["NORTH"],["NORTH"],["HARVEST"],["WATER"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["WATER"],["SOUTH"],["CARE"],["WATER"],["WATER"],["NORTH"],["HARVEST"],["CARE"],["FEED"],["WATER"],["WATER"],["EAST"]],"market":[["SELL","MILK",9]]},{"farmer":["WEST"],"hands":[["SOUTH"],["WATER"],["NORTH"],["EAST"],["EAST"],["WATER"],["WATER"],["WEST"],["CARE"],["WEST"],["NORTH"],["EAST"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["SOUTH"],["NORTH"],["NORTH"],["WATER"],["WEST"],["EAST"],["FEED"],["NORTH"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["SOUTH"],["FERTILIZE"],["FEED"],["EAST"],["WEST"],["EAST"],["CARE"],["NORTH"],["SOUTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["FERTILIZE"],"hands":[["SOUTH"],["DROP"],["NORTH"],["CARE"],["EAST"],["WATER"],["WATER"],["SOUTH"],["FEED"],["SOUTH"],["COLLECT_FERTILIZER"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["DROP"],["WEST"],["FEED"],["EAST"],["WATER"],["SOUTH"],["EAST"],["FEED"],["CARE"],["SOUTH"],["EAST"],["WEST"]],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",4]]},{"farmer":["BUILD_PASTURE"],"hands":[["WEST"],["SOUTH"],["CARE"],["SOUTH"],["WEST"],["SOUTH"],["WATER"],["SOUTH"],["WEST"],["SOUTH"],["BUILD_PASTURE"],["WEST"]],"market":[["SELL","MILK",6],["SELL","STRAWBERRY",2],["SELL","MELON",1]]},{"farmer":["SOUTH"],"hands":[["WEST"],["WEST"],["SOUTH"],["EAST"],["NORTH"],["SOUTH"],["WEST"],["FEED"],["WEST"],["EAST"],["SOUTH"],["NORTH"]],"market":[["SELL","MELON",1]]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["COLLECT_FERTILIZER"],["WEST"],["SOUTH"],["PLACE","SHEEP"],["WATER"],["SOUTH"],["WEST"],["CARE"],["FEED"],["WATER"],["FERTILIZE"],["COLLECT_FERTILIZER"]],"market":[["SELL","MELON",1]]},{"farmer":["EAST"],"hands":[["EAST"],["CARE"],["PLACE","SHEEP"],["FEED"],["WEST"],["WATER"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["CARE"],["WEST"],["WATER"],["NORTH"]],"market":[["SELL","MELON",1],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["PICKUP","SHEEP",1],["EAST"],["FEED"],["WEST"],["WEST"],["SOUTH"],["WEST"],["SOUTH"],["WEST"],["BUILD_PASTURE"],["WEST"],["FERTILIZE"]],"market":[["SELL","MELON",1],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["NORTH"],"hands":[["WEST"],["COLLECT_FERTILIZER"],["CARE"],["FEED"],["COLLECT_FERTILIZER"],["WATER"],["WEST"],["EAST"],["EAST"],["NORTH"],["EAST"],["WEST"]],"market":[["SELL","MELON",1]]},{"farmer":["EAST"],"hands":[["WEST"],["SOUTH"],["EAST"],["CARE"],["WEST"],["NORTH"],["NORTH"],["PLANT","WHEAT"],["COLLECT_FERTILIZER"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["EAST"],"hands":[["WEST"],["WATER"],["EAST"],["SOUTH"],["FERTILIZE"],["NORTH"],["FERTILIZE"],["WATER"],["EAST"],["WATER"],["CARE"],["NORTH"]],"market":[]},{"farmer":["EAST"],"hands":[["WEST"],["WEST"],["NORTH"],["DIG"],["WEST"],["WATER"],["SOUTH"],["FERTILIZE"],["EAST"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["WEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",9],["SELL","STRAWBERRY",14],["SELL","FERTILIZER",2],["BUY_SEED","CARROT",5]]},{"farmer":["NORTH"],"hands":[["EAST"],["EAST"],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1],["PICKUP","SHEEP",1]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_SEED","CARROT",3],["SELL","MELON",6]]},{"farmer":["NORTH"],"hands":[["EAST"],["EAST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["NORTH"],"hands":[["EAST"],["NORTH"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["FEED"],["FEED"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["HARVEST"],"hands":[["EAST"],["NORTH"],["WEST"],["EAST"],["CARE"],["CARE"],["WATER"],["WATER"],["NORTH"],["CARE"],["WATER"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["HARVEST"],["WEST"],["NORTH"],["NORTH"],["WEST"],["EAST"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["NORTH"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["HARVEST"],["SOUTH"],["FEED"],["NORTH"],["FEED"],["FEED"],["WATER"],["HARVEST"],["EAST"],["EAST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["EAST"],["WATER"],["WEST"],["NORTH"],["CARE"],["CARE"],["NORTH"],["WEST"],["WATER"],["EAST"],["WATER"],["WATER"]],"market":[]},{"farmer":["WEST"],"hands":[["SOUTH"],["EAST"],["WEST"],["NORTH"],["WEST"],["WEST"],["NORTH"],["CARE"],["WEST"],["EAST"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["HARVEST"],["COLLECT_FERTILIZER"],["SOUTH"],["NORTH"],["FEED"],["FEED"],["WATER"],["NORTH"],["WEST"],["EAST"],["WATER"],["NORTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WATER"],["NORTH"],["PLACE","SHEEP"],["FEED"],["CARE"],["CARE"],["WEST"],["NORTH"],["WEST"],["EAST"],["NORTH"],["WATER"]],"market":[]},{"farmer":["SOUTH"],"hands":[["NORTH"],["FERTILIZE"],["FEED"],["HARVEST"],["NORTH"],["SOUTH"],["WEST"],["WATER"],["WEST"],["NORTH"],["WATER"],["EAST"]],"market":[["BUY_ANIMAL","GOOSE",1],["BUY_PRODUCT","WHEAT",1]]},{"farmer":["WATER"],"hands":[["NORTH"],["HARVEST"],["CARE"],["CARE"],["NORTH"],["SOUTH"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["WEST"],["NORTH"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[["WATER"],["WATER"],["EAST"],["WEST"],["FEED"],["FEED"],["WATER"],["SOUTH"],["EAST"],["NORTH"],["WEST"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["NORTH"],["WATER"],["WEST"],["CARE"],["CARE"],["WEST"],["WATER"],["PICKUP","GOOSE",1],["NORTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["WATER"],["EAST"],["FEED"],["WEST"],["COLLECT_FERTILIZER"],["SOUTH"],["WEST"],["SOUTH"],["FERTILIZE"],["SOUTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["HARVEST"],"hands":[["HARVEST"],["WEST"],["EAST"],["CARE"],["WEST"],["EAST"],["WATER"],["WATER"],["SOUTH"],["WEST"],["WATER"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["WEST"],["NORTH"],["SOUTH"],["FEED"],["EAST"],["WEST"],["NORTH"],["PLACE","GOOSE"],["FERTILIZE"],["EAST"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["CARE"],"hands":[["SOUTH"],["WEST"],["FEED"],["SOUTH"],["EAST"],["EAST"],["HARVEST"],["WATER"],["SOUTH"],["NORTH"],["COLLECT_FERTILIZER"],["FERTILIZE"]],"market":[["BUY_PRODUCT","WHEAT",1]]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["SOUTH"],["SOUTH"],["COLLECT_FERTILIZER"],["FEED"],["EAST"],["NORTH"],["EAST"],["EAST"],["PLANT","WHEAT"],["WATER"],["EAST"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[["FEED"],["SOUTH"],["EAST"],["CARE"],["EAST"],["NORTH"],["EAST"],["WATER"],["WATER"],["EAST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WEST"],["SOUTH"],["DROP"],["EAST"],["SOUTH"],["NORTH"],["COLLECT_FERTILIZER"],["EAST"],["FERTILIZE"],["SOUTH"],["COLLECT_FERTILIZER"],["WEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["FEED"],["DROP"],["WEST"],["EAST"],["SOUTH"],["FEED"],["WEST"],["EAST"],["SOUTH"],["SOUTH"],["WEST"],["WEST"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["FERTILIZE"],"hands":[["CARE"],["WEST"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["CARE"],["WEST"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"],["SOUTH"],["WEST"],["WEST"]],"market":[["SELL","MILK",6],["SELL","STRAWBERRY",1],["SELL","FERTILIZER",1]]},{"farmer":["WEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",13],["SELL","MILK",2],["SELL","EGG",2],["SELL","FERTILIZER",3],["SELL","WHEAT",1]]},{"farmer":["WEST"],"hands":[["HARVEST"],["NORTH"],["NORTH"],["WEST"],["HARVEST"]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_SEED","WHEAT",2]]},{"farmer":["SOUTH"],"hands":[["NORTH"],["HARVEST"],["WEST"],["PICKUP","WHEAT",14],["WEST"],["HARVEST"],["PICKUP","WHEAT",14],["HARVEST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["HARVEST"],"hands":[["HARVEST"],["NORTH"],["HARVEST"],["PICKUP","FERTILIZER",9],["NORTH"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["WEST"],["PICKUP","FERTILIZER",9]],"market":[["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",18]]},{"farmer":["NORTH"],"hands":[["NORTH"],["HARVEST"],["WEST"],["NORTH"],["HARVEST"],["EAST"],["EAST"],["WATER"],["NORTH"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["HARVEST"],"hands":[["NORTH"],["NORTH"],["WEST"],["FEED"],["NORTH"],["HARVEST"],["EAST"],["EAST"],["FEED"],["NORTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["HARVEST"],["NORTH"],["CARE"],["HARVEST"],["NORTH"],["NORTH"],["EAST"],["CARE"],["FEED"],["WATER"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["HARVEST"],["NORTH"],["NORTH"],["NORTH"],["WATER"],["HARVEST"],["NORTH"],["HARVEST"],["WEST"],["CARE"],["WEST"],["HARVEST"]],"market":[]},{"farmer":["WEST"],"hands":[["WEST"],["HARVEST"],["NORTH"],["FEED"],["WEST"],["NORTH"],["NORTH"],["WATER"],["FEED"],["WEST"],["HARVEST"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["WATER"],["WATER"],["HARVEST"],["CARE"],["WEST"],["HARVEST"],["FEED"],["NORTH"],["CARE"],["WEST"],["EAST"],["EAST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["NORTH"],"hands":[["WEST"],["EAST"],["NORTH"],["NORTH"],["WATER"],["WATER"],["CARE"],["WATER"],["WEST"],["FEED"],["SOUTH"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["SOUTH"],["WATER"],["FEED"],["SOUTH"],["WEST"],["EAST"],["WEST"],["FEED"],["CARE"],["SOUTH"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["HARVEST"],["WATER"],["SOUTH"],["CARE"],["SOUTH"],["WATER"],["SOUTH"],["WATER"],["CARE"],["NORTH"],["FEED"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["EAST"],["EAST"],["NORTH"],["SOUTH"],["SOUTH"],["FEED"],["SOUTH"],["SOUTH"],["WEST"],["CARE"],["HARVEST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["FEED"],["EAST"],["PLANT","WHEAT"],["NORTH"],["SOUTH"],["SOUTH"],["CARE"],["WATER"],["SOUTH"],["WEST"],["WEST"],["WEST"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["EAST"],"hands":[["CARE"],["NORTH"],["WATER"],["FEED"],["WATER"],["DROP"],["EAST"],["EAST"],["FEED"],["NORTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["EAST"],"hands":[["COLLECT_FERTILIZER"],["HARVEST"],["EAST"],["CARE"],["EAST"],["WEST"],["FEED"],["EAST"],["CARE"],["FEED"],["SOUTH"],["FEED"]],"market":[["SELL","STRAWBERRY",8]]},{"farmer":["SOUTH"],"hands":[["EAST"],["SOUTH"],["EAST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["SOUTH"],["CARE"],["EAST"],["EAST"],["CARE"],["FEED"],["CARE"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["SOUTH"],["SOUTH"],["SOUTH"],["EAST"],["EAST"],["WATER"],["NORTH"],["WATER"],["HARVEST"],["COLLECT_FERTILIZER"],["CARE"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["DROP"],"hands":[["SOUTH"],["WATER"],["COLLECT_FERTILIZER"],["EAST"],["EAST"],["NORTH"],["WATER"],["NORTH"],["EAST"],["EAST"],["COLLECT_FERTILIZER"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["SOUTH"],["WEST"],["SOUTH"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["COLLECT_FERTILIZER"],["FEED"],["FERTILIZE"],["EAST"],["SOUTH"]],"market":[["SELL","WHEAT",9]]},{"farmer":["EAST"],"hands":[["DROP"],["COLLECT_FERTILIZER"],["SOUTH"],["HARVEST"],["FERTILIZE"],["NORTH"],["HARVEST"],["WEST"],["COLLECT_FERTILIZER"],["SOUTH"],["NORTH"],["FERTILIZE"]],"market":[]},{"farmer":["FERTILIZE"],"hands":[["WEST"],["WEST"],["DROP"],["PLANT","WHEAT"],["WEST"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"],["COLLECT_FERTILIZER"],["EAST"],["PLANT","WHEAT"],["HARVEST"],["WEST"]],"market":[["SELL","MILK",9],["BUY_SEED","WHEAT",3]]},{"farmer":["NORTH"],"hands":[["COLLECT_FERTILIZER"],["SOUTH"],["WEST"],["WATER"],["COLLECT_FERTILIZER"],["EAST"],["WATER"],["WEST"],["WEST"],["WATER"],["PLANT","WHEAT"],["WEST"]],"market":[["SELL","MILK",3],["SELL","WOOL",5],["BUY_SEED","WHEAT",1]]},{"farmer":["WEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",6],["SELL","STRAWBERRY",10],["SELL","FERTILIZER",5],["SELL","WHEAT",8],["BUY_SEED","WHEAT",2]]},{"farmer":["WEST"],"hands":[["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["SOUTH"],"hands":[["FEED"],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["FEED"],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",19]]},{"farmer":["SOUTH"],"hands":[["CARE"],["NORTH"],["WATER"],["WEST"],["CARE"],["EAST"],["WATER"],["EAST"],["CARE"],["WATER"],["WATER"],["NORTH"]],"market":[]},{"farmer":["HARVEST"],"hands":[["NORTH"],["FEED"],["WEST"],["WEST"],["WEST"],["EAST"],["EAST"],["EAST"],["WEST"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["WEST"],"hands":[["FEED"],["CARE"],["WEST"],["FEED"],["FEED"],["EAST"],["EAST"],["NORTH"],["NORTH"],["WATER"],["EAST"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["CARE"],["WEST"],["FEED"],["WEST"],["CARE"],["NORTH"],["EAST"],["NORTH"],["NORTH"],["NORTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["NORTH"],["CARE"],["NORTH"],["WEST"],["HARVEST"],["EAST"],["NORTH"],["NORTH"],["NORTH"],["NORTH"],["EAST"]],"market":[]},{"farmer":["NORTH"],"hands":[["FEED"],["FEED"],["SOUTH"],["FEED"],["NORTH"],["NORTH"],["EAST"],["HARVEST"],["HARVEST"],["NORTH"],["WATER"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["CARE"],["CARE"],["FEED"],["CARE"],["WATER"],["HARVEST"],["NORTH"],["WEST"],["NORTH"],["NORTH"],["WEST"],["WEST"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["NORTH"],["CARE"],["WEST"],["WEST"],["WATER"],["HARVEST"],["WATER"],["WATER"],["HARVEST"],["WEST"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["NORTH"],["NORTH"],["WEST"],["NORTH"],["NORTH"],["EAST"],["WATER"],["NORTH"],["WEST"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["FEED"],["FEED"],["WEST"],["NORTH"],["WATER"],["WATER"],["NORTH"],["WATER"],["WEST"],["WATER"],["COLLECT_FERTILIZER"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["CARE"],["CARE"],["FEED"],["NORTH"],["EAST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["WATER"],["WEST"],["NORTH"],["WEST"]],"market":[]},{"farmer":["WATER"],"hands":[["EAST"],["EAST"],["CARE"],["FEED"],["EAST"],["NORTH"],["NORTH"],["WATER"],["WEST"],["COLLECT_FERTILIZER"],["FERTILIZE"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["SOUTH"],"hands":[["WATER"],["EAST"],["EAST"],["CARE"],["WATER"],["HARVEST"],["FERTILIZE"],["WEST"],["DIG"],["EAST"],["EAST"],["EAST"]],"market":[]},{"farmer":["HARVEST"],"hands":[["EAST"],["FEED"],["FERTILIZE"],["COLLECT_FERTILIZER"],["NORTH"],["WEST"],["NORTH"],["SOUTH"],["PLANT","WHEAT"],["FERTILIZE"],["WEST"],["NORTH"]],"market":[["BUY_SEED","WHEAT",3]]},{"farmer":["PLANT","WHEAT"],"hands":[["EAST"],["CARE"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["SOUTH"],["NORTH"],["SOUTH"],["WATER"],["WEST"],["WEST"],["FERTILIZE"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["WATER"],"hands":[["EAST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"],["FERTILIZE"],["EAST"],["FEED"],["PLANT","WHEAT"],["COLLECT_FERTILIZER"],["SOUTH"],["SOUTH"],["WEST"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["EAST"],["NORTH"],["EAST"],["FERTILIZE"],["CARE"],["WATER"],["WEST"],["PLANT","WHEAT"],["SOUTH"],["SOUTH"],["EAST"]],"market":[]},{"farmer":["PLANT","WHEAT"],"hands":[["SOUTH"],["FERTILIZE"],["NORTH"],["DIG"],["SOUTH"],["EAST"],["WEST"],["NORTH"],["WATER"],["SOUTH"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"]],"market":[]},{"farmer":["WATER"],"hands":[["SOUTH"],["SOUTH"],["NORTH"],["PLANT","WHEAT"],["SOUTH"],["EAST"],["PLANT","WHEAT"],["FERTILIZE"],["EAST"],["SOUTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["FEED"],["SOUTH"],["NORTH"],["WATER"],["SOUTH"],["SOUTH"],["WATER"],["SOUTH"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["CARE"],["COLLECT_FERTILIZER"],["FERTILIZE"],["SOUTH"],["WEST"],["FEED"],["SOUTH"],["SOUTH"],["SOUTH"],["EAST"],["EAST"],["WEST"]],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",10],["SELL","STRAWBERRY",4],["SELL","WOOL",6],["SELL","FERTILIZER",3],["SELL","WHEAT",5]]},{"farmer":["NORTH"],"hands":[["NORTH"],["HARVEST"],["PICKUP","WHEAT",14],["WEST"],["EAST"]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_SEED","WHEAT",1]]},{"farmer":["HARVEST"],"hands":[["HARVEST"],["WEST"],["PICKUP","WHEAT",14],["HARVEST"],["EAST"],["WEST"],["HARVEST"],["WEST"],["HARVEST"],["PICKUP","WHEAT",14]],"market":[["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["WEST"],"hands":[["EAST"],["WEST"],["PICKUP","FERTILIZER",10],["WEST"],["NORTH"],["NORTH"],["PICKUP","FERTILIZER",10],["WEST"],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["WEST"],["WEST"]],"market":[["SELL","EGG",1],["BUY_PRODUCT","WHEAT",4]]},{"farmer":["WEST"],"hands":[["NORTH"],["HARVEST"],["DROP"],["WEST"],["HARVEST"],["HARVEST"],["WEST"],["HARVEST"],["WATER"],["NORTH"],["WEST"],["NORTH"]],"market":[["HIRE"]]},{"farmer":["HARVEST"],"hands":[["NORTH"],["EAST"],["PICKUP","WHEAT",14],["HARVEST"],["WEST"],["NORTH"],["PICKUP","WHEAT",14],["WEST"],["PICKUP","WHEAT",14],["FEED"],["WEST"],["NORTH"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["WEST"],"hands":[["HARVEST"],["EAST"],["PICKUP","FERTILIZER",9],["WEST"],["NORTH"],["NORTH"],["PICKUP","FERTILIZER",9],["EAST"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["EAST"],["HARVEST"]],"market":[["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",14]]},{"farmer":["NORTH"],"hands":[["WEST"],["WATER"],["DROP"],["HARVEST"],["HARVEST"],["NORTH"],["CARE"],["EAST"],["NORTH"],["CARE"],["EAST"],["WATER"]],"market":[]},{"farmer":["HARVEST"],"hands":[["HARVEST"],["PICKUP","FERTILIZER",9],["PICKUP","FERTILIZER",9],["EAST"],["WEST"],["HARVEST"],["PICKUP","FERTILIZER",9],["EAST"],["FEED"],["PICKUP","FERTILIZER",9],["EAST"],["EAST"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["WEST"],"hands":[["NORTH"],["DROP"],["EAST"],["EAST"],["HARVEST"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"],["CARE"],["WEST"],["COLLECT_FERTILIZER"],["NORTH"]],"market":[]},{"farmer":["WATER"],"hands":[["HARVEST"],["PICKUP","FERTILIZER",9],["WEST"],["EAST"],["WATER"],["HARVEST"],["FEED"],["EAST"],["SOUTH"],["FEED"],["PICKUP","FERTILIZER",9],["HARVEST"]],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",2],["SELL","FERTILIZER",1]]},{"farmer":["EAST"],"hands":[["WATER"],["WEST"],["EAST"],["CARE"],["NORTH"],["COLLECT_FERTILIZER"],["CARE"],["EAST"],["WEST"],["NORTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["HARVEST"],["NORTH"],["EAST"],["WATER"],["SOUTH"],["NORTH"],["HARVEST"],["WEST"],["FEED"],["EAST"],["EAST"]],"market":[]},{"farmer":["WATER"],"hands":[["SOUTH"],["COLLECT_FERTILIZER"],["WATER"],["DROP"],["EAST"],["SOUTH"],["FEED"],["NORTH"],["WEST"],["CARE"],["EAST"],["EAST"]],"market":[["SELL","MELON",2]]},{"farmer":["EAST"],"hands":[["SOUTH"],["WEST"],["EAST"],["NORTH"],["SOUTH"],["CARE"],["NORTH"],["HARVEST"],["FEED"],["NORTH"],["HARVEST"],["EAST"]],"market":[["SELL","MILK",1]]},{"farmer":["EAST"],"hands":[["SOUTH"],["COLLECT_FERTILIZER"],["WATER"],["COLLECT_FERTILIZER"],["WATER"],["SOUTH"],["NORTH"],["WATER"],["CARE"],["NORTH"],["EAST"],["NORTH"]],"market":[["SELL","MELON",2]]},{"farmer":["SOUTH"],"hands":[["DROP"],["WEST"],["EAST"],["NORTH"],["EAST"],["SOUTH"],["FEED"],["WEST"],["SOUTH"],["FEED"],["EAST"],["WATER"]],"market":[["SELL","MELON",1]]},{"farmer":["SOUTH"],"hands":[["WEST"],["SOUTH"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["DROP"],["CARE"],["COLLECT_FERTILIZER"],["FEED"],["CARE"],["WATER"],["WEST"]],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",4],["SELL","MELON",2]]},{"farmer":["DROP"],"hands":[["WEST"],["WATER"],["NORTH"],["WEST"],["WATER"],["WEST"],["EAST"],["SOUTH"],["CARE"],["WEST"],["NORTH"],["WEST"]],"market":[["SELL","MILK",6],["SELL","MELON",3]]},{"farmer":["WEST"],"hands":[["COLLECT_FERTILIZER"],["WEST"],["NORTH"],["WEST"],["EAST"],["NORTH"],["EAST"],["DROP"],["SOUTH"],["WEST"],["COLLECT_FERTILIZER"],["WEST"]],"market":[["SELL","MILK",6],["SELL","MELON",6]]},{"farmer":["WEST"],"hands":[["WEST"],["COLLECT_FERTILIZER"],["WATER"],["FERTILIZE"],["NORTH"],["COLLECT_FERTILIZER"],["FEED"],["WEST"],["FEED"],["FEED"],["SOUTH"],["WEST"]],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",2],["SELL","MELON",5]]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["WEST"],["SOUTH"],["WEST"],["NORTH"],["WATER"],["EAST"],["CARE"],["SOUTH"],["CARE"],["CARE"],["FERTILIZE"],["WEST"]],"market":[["SELL","WHEAT",5]]},{"farmer":["WEST"],"hands":[["PLANT","WHEAT"],["PLANT","WHEAT"],["SOUTH"],["WATER"],["WEST"],["EAST"],["EAST"],["SOUTH"],["WEST"],["NORTH"],["WEST"],["WEST"]],"market":[["SELL","MELON",5]]},{"farmer":["WATER"],"hands":[["NORTH"],["WATER"],["COLLECT_FERTILIZER"],["NORTH"],["PLANT","WHEAT"],["EAST"],["SOUTH"],["COLLECT_FERTILIZER"],["WEST"],["WATER"],["PLANT","WHEAT"],["SOUTH"]],"market":[["SELL","MELON",5]]},{"farmer":["PICKUP","WHEAT",14],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","STRAWBERRY",8],["SELL","MELON",3],["SELL","EGG",3],["SELL","FERTILIZER",6],["SELL","WHEAT",8]]},{"farmer":["FEED"],"hands":[["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MELON",1],["SELL","FERTILIZER",1],["BUY_SEED","WHEAT",1]]},{"farmer":["PASS"],"hands":[["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10]],"market":[["HIRE"],["HIRE"],["SELL","MELON",1],["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",18]]},{"farmer":["PICKUP","FERTILIZER",10],"hands":[["EAST"],["WEST"],["EAST"],["CARE"],["EAST"],["WATER"],["WEST"],["CARE"],["WEST"],["WATER"],["WEST"],["CARE"]],"market":[["SELL","MELON",1]]},{"farmer":["CARE"],"hands":[["EAST"],["WEST"],["EAST"],["WEST"],["NORTH"],["WEST"],["WEST"],["WEST"],["EAST"],["WEST"],["EAST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["FEED"],["EAST"],["NORTH"],["NORTH"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["EAST"],["WEST"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["FEED"],"hands":[["NORTH"],["WEST"],["NORTH"],["NORTH"],["NORTH"],["CARE"],["EAST"],["WEST"],["EAST"],["WEST"],["EAST"],["FERTILIZE"]],"market":[]},{"farmer":["CARE"],"hands":[["FEED"],["WEST"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["WEST"],["NORTH"],["EAST"],["SOUTH"],["EAST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["EAST"],["SOUTH"],["FEED"],["HARVEST"],["HARVEST"],["WEST"],["WEST"],["NORTH"],["EAST"],["HARVEST"],["EAST"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["EAST"],["FEED"],["EAST"],["WEST"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["HARVEST"],["WATER"],["WEST"],["DIG"]],"market":[]},{"farmer":["CARE"],"hands":[["HARVEST"],["CARE"],["FEED"],["WEST"],["SOUTH"],["NORTH"],["NORTH"],["HARVEST"],["WATER"],["EAST"],["NORTH"],["PLANT","WHEAT"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["SOUTH"],"hands":[["WATER"],["EAST"],["CARE"],["NORTH"],["SOUTH"],["NORTH"],["COLLECT_FERTILIZER"],["WEST"],["WEST"],["WEST"],["NORTH"],["WATER"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["EAST"],["WEST"],["HARVEST"],["HARVEST"],["HARVEST"],["WEST"],["WATER"],["DIG"],["WEST"],["HARVEST"],["WEST"]],"market":[]},{"farmer":["CARE"],"hands":[["HARVEST"],["FEED"],["CARE"],["WATER"],["CARE"],["WATER"],["FERTILIZE"],["NORTH"],["PLANT","WHEAT"],["COLLECT_FERTILIZER"],["EAST"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["WATER"],["CARE"],["WEST"],["EAST"],["NORTH"],["SOUTH"],["NORTH"],["WATER"],["WATER"],["SOUTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["EAST"],["WEST"],["WATER"],["WATER"],["WATER"],["FERTILIZE"],["EAST"],["NORTH"],["FERTILIZE"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["WEST"],["NORTH"],["WEST"],["EAST"],["EAST"],["SOUTH"],["WATER"],["SOUTH"],["COLLECT_FERTILIZER"],["EAST"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["PASS"],"hands":[["NORTH"],["FEED"],["FEED"],["PLANT","WHEAT"],["NORTH"],["PLANT","WHEAT"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["PLANT","WHEAT"],["FERTILIZE"],["WEST"]],"market":[]},{"farmer":["EAST"],"hands":[["NORTH"],["EAST"],["CARE"],["WATER"],["WATER"],["WATER"],["FERTILIZE"],["WEST"],["NORTH"],["WATER"],["NORTH"],["DIG"]],"market":[]},{"farmer":["NORTH"],"hands":[["FEED"],["NORTH"],["WEST"],["SOUTH"],["SOUTH"],["SOUTH"],["EAST"],["FERTILIZE"],["NORTH"],["EAST"],["WATER"],["PLANT","WHEAT"]],"market":[]},{"farmer":["NORTH"],"hands":[["CARE"],["NORTH"],["NORTH"],["SOUTH"],["DIG"],["DIG"],["NORTH"],["EAST"],["FERTILIZE"],["PLANT","WHEAT"],["SOUTH"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["WEST"],["WEST"],["FEED"],["SOUTH"],["PLANT","WHEAT"],["PLANT","WHEAT"],["PLANT","WHEAT"],["EAST"],["SOUTH"],["WATER"],["WATER"],["FERTILIZE"]],"market":[]},{"farmer":["FEED"],"hands":[["WEST"],["COLLECT_FERTILIZER"],["CARE"],["SOUTH"],["WATER"],["WATER"],["WATER"],["WATER"],["COLLECT_FERTILIZER"],["NORTH"],["WEST"],["NORTH"]],"market":[]},{"farmer":["CARE"],"hands":[["FEED"],["WEST"],["EAST"],["COLLECT_FERTILIZER"],["WEST"],["EAST"],["EAST"],["SOUTH"],["EAST"],["COLLECT_FERTILIZER"],["WEST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["NORTH"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",8],["SELL","STRAWBERRY",11],["SELL","WOOL",4],["SELL","FERTILIZER",6],["SELL","WHEAT",1]]},{"farmer":["NORTH"],"hands":[["HARVEST"],["NORTH"],["NORTH"],["WEST"],["HARVEST"]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["HARVEST"],"hands":[["NORTH"],["HARVEST"],["WEST"],["HARVEST"],["WEST"],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["HARVEST"],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["PASS"],"hands":[["HARVEST"],["EAST"],["HARVEST"],["WEST"],["NORTH"],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["PICKUP","FERTILIZER",10],["WEST"],["PICKUP","FERTILIZER",10],["WEST"],["PICKUP","FERTILIZER",10]],"market":[["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",18]]},{"farmer":["NORTH"],"hands":[["SOUTH"],["EAST"],["WEST"],["WEST"],["HARVEST"],["NORTH"],["WEST"],["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["SOUTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["DROP"],["EAST"],["HARVEST"],["HARVEST"],["COLLECT_FERTILIZER"],["FEED"],["NORTH"],["FEED"],["SOUTH"],["NORTH"],["EAST"],["WATER"]],"market":[]},{"farmer":["HARVEST"],"hands":[["CARE"],["EAST"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["CARE"],["CARE"],["CARE"],["HARVEST"],["CARE"],["CARE"],["NORTH"]],"market":[["SELL","MILK",6]]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["COLLECT_FERTILIZER"],["HARVEST"],["EAST"],["WEST"],["NORTH"],["WEST"],["WEST"],["WEST"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["WEST"],"hands":[["EAST"],["NORTH"],["DROP"],["NORTH"],["COLLECT_FERTILIZER"],["FEED"],["FEED"],["NORTH"],["WATER"],["WEST"],["EAST"],["WEST"]],"market":[]},{"farmer":["FERTILIZE"],"hands":[["FERTILIZE"],["WATER"],["EAST"],["WATER"],["WEST"],["CARE"],["CARE"],["FEED"],["WEST"],["FEED"],["FERTILIZE"],["SOUTH"]],"market":[["SELL","MILK",6]]},{"farmer":["WEST"],"hands":[["NORTH"],["NORTH"],["WEST"],["SOUTH"],["SOUTH"],["NORTH"],["WEST"],["WEST"],["SOUTH"],["CARE"],["WEST"],["HARVEST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["COLLECT_FERTILIZER"],["HARVEST"],["NORTH"],["SOUTH"],["CARE"],["FEED"],["SOUTH"],["WEST"],["WATER"],["SOUTH"],["WEST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["HARVEST"],"hands":[["EAST"],["SOUTH"],["COLLECT_FERTILIZER"],["FERTILIZE"],["WEST"],["CARE"],["FEED"],["EAST"],["NORTH"],["SOUTH"],["WEST"],["SOUTH"]],"market":[["BUY_SEED","WHEAT",2]]},{"farmer":["WEST"],"hands":[["DIG"],["SOUTH"],["WEST"],["NORTH"],["WEST"],["NORTH"],["CARE"],["EAST"],["COLLECT_FERTILIZER"],["FEED"],["COLLECT_FERTILIZER"],["PLANT","WHEAT"]],"market":[]},{"farmer":["FEED"],"hands":[["PLANT","WHEAT"],["FEED"],["WEST"],["DIG"],["WEST"],["NORTH"],["WEST"],["NORTH"],["NORTH"],["CARE"],["NORTH"],["WATER"]],"market":[]},{"farmer":["CARE"],"hands":[["WATER"],["WEST"],["PLANT","WHEAT"],["PLANT","WHEAT"],["NORTH"],["FEED"],["WEST"],["NORTH"],["FERTILIZE"],["COLLECT_FERTILIZER"],["NORTH"],["FERTILIZE"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["FEED"],["WATER"],["WATER"],["FERTILIZE"],["CARE"],["SOUTH"],["FEED"],["NORTH"],["SOUTH"],["HARVEST"],["SOUTH"]],"market":[]},{"farmer":["WATER"],"hands":[["EAST"],["CARE"],["WEST"],["NORTH"],["WATER"],["EAST"],["FEED"],["CARE"],["NORTH"],["FERTILIZE"],["EAST"],["DIG"]],"market":[["BUY_SEED","TOMATO",1]]},{"farmer":["SOUTH"],"hands":[["EAST"],["WEST"],["NORTH"],["NORTH"],["EAST"],["EAST"],["CARE"],["COLLECT_FERTILIZER"],["WATER"],["SOUTH"],["EAST"],["PLANT","WHEAT"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["CARE"],["NORTH"],["FERTILIZE"],["WATER"],["EAST"],["FEED"],["NORTH"],["WEST"],["NORTH"],["PLANT","WHEAT"],["EAST"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["NORTH"],["FEED"],["EAST"],["EAST"],["PLANT","WHEAT"],["CARE"],["NORTH"],["PLANT","WHEAT"],["NORTH"],["WATER"],["DIG"],["SOUTH"]],"market":[]},{"farmer":["FERTILIZE"],"hands":[["FERTILIZE"],["CARE"],["WATER"],["EAST"],["NORTH"],["EAST"],["FERTILIZE"],["WATER"],["WATER"],["WEST"],["PLANT","WHEAT"],["SOUTH"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["COLLECT_FERTILIZER"],["EAST"],["EAST"],["NORTH"],["HARVEST"],["EAST"],["FERTILIZE"],["NORTH"],["PLANT","CARROT"],["WATER"],["PLANT","CARROT"]],"market":[]},{"farmer":["EAST"],"hands":[["COLLECT_FERTILIZER"],["WEST"],["EAST"],["EAST"],["FERTILIZE"],["EAST"],["SOUTH"],["EAST"],["WATER"],["WATER"],["FERTILIZE"],["WATER"]],"market":[["BUY_SEED","WHEAT",1]]},{"farmer":["PICKUP","WHEAT",14],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",12],["SELL","STRAWBERRY",2],["SELL","WOOL",13],["SELL","FERTILIZER",6],["SELL","WHEAT",24]]},{"farmer":["FEED"],"hands":[["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1]]},{"farmer":["CARE"],"hands":[["NORTH"],["WEST"],["EAST"],["CARE"],["EAST"],["WATER"],["EAST"],["CARE"],["WEST"],["WATER"]],"market":[["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",18]]},{"farmer":["NORTH"],"hands":[["FEED"],["WEST"],["EAST"],["WEST"],["NORTH"],["WEST"],["EAST"],["EAST"],["EAST"],["WEST"],["WATER"],["NORTH"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["FEED"],"hands":[["CARE"],["FEED"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["EAST"],["EAST"],["WEST"],["WEST"],["WEST"],["WEST"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["CARE"],"hands":[["WEST"],["WEST"],["NORTH"],["NORTH"],["NORTH"],["CARE"],["NORTH"],["EAST"],["COLLECT_FERTILIZER"],["WEST"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["WEST"],"hands":[["NORTH"],["WEST"],["NORTH"],["NORTH"],["NORTH"],["WEST"],["NORTH"],["EAST"],["WEST"],["SOUTH"],["WEST"],["WEST"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["FEED"],"hands":[["FEED"],["SOUTH"],["HARVEST"],["HARVEST"],["HARVEST"],["NORTH"],["HARVEST"],["WATER"],["WEST"],["HARVEST"],["COLLECT_FERTILIZER"],["WEST"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["CARE"],"hands":[["CARE"],["HARVEST"],["EAST"],["WEST"],["COLLECT_FERTILIZER"],["NORTH"],["COLLECT_FERTILIZER"],["NORTH"],["WEST"],["WATER"],["NORTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["SOUTH"],"hands":[["NORTH"],["FEED"],["EAST"],["WEST"],["WEST"],["HARVEST"],["SOUTH"],["NORTH"],["WEST"],["SOUTH"],["NORTH"],["WEST"]],"market":[]},{"farmer":["FEED"],"hands":[["NORTH"],["CARE"],["HARVEST"],["NORTH"],["WEST"],["EAST"],["FERTILIZE"],["NORTH"],["NORTH"],["WATER"],["WATER"],["WEST"]],"market":[]},{"farmer":["CARE"],"hands":[["FEED"],["EAST"],["WATER"],["HARVEST"],["WEST"],["SOUTH"],["EAST"],["WATER"],["NORTH"],["EAST"],["WEST"],["WATER"]],"market":[]},{"farmer":["NORTH"],"hands":[["CARE"],["EAST"],["SOUTH"],["WATER"],["WATER"],["FEED"],["DIG"],["NORTH"],["HARVEST"],["WATER"],["FERTILIZE"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["FEED"],["COLLECT_FERTILIZER"],["WEST"],["WEST"],["CARE"],["PLANT","CARROT"],["HARVEST"],["WATER"],["WEST"],["WEST"],["HARVEST"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["CARE"],["WEST"],["HARVEST"],["HARVEST"],["EAST"],["WATER"],["WEST"],["NORTH"],["WEST"],["WATER"],["EAST"]],"market":[]},{"farmer":["FEED"],"hands":[["FEED"],["COLLECT_FERTILIZER"],["NORTH"],["EAST"],["PLANT","CARROT"],["SOUTH"],["NORTH"],["SOUTH"],["HARVEST"],["NORTH"],["SOUTH"],["EAST"]],"market":[]},{"farmer":["CARE"],"hands":[["CARE"],["WEST"],["NORTH"],["SOUTH"],["WATER"],["FEED"],["NORTH"],["SOUTH"],["EAST"],["NORTH"],["PLANT","CARROT"],["FERTILIZE"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["EAST"],["SOUTH"],["FERTILIZE"],["FEED"],["FERTILIZE"],["COLLECT_FERTILIZER"],["NORTH"],["FEED"],["WEST"],["WATER"],["WATER"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["EAST"],["FERTILIZE"],["WEST"],["CARE"],["EAST"],["HARVEST"],["NORTH"],["CARE"],["PLANT","CARROT"],["SOUTH"],["SOUTH"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["NORTH"],"hands":[["EAST"],["EAST"],["HARVEST"],["COLLECT_FERTILIZER"],["EAST"],["CARE"],["HARVEST"],["EAST"],["WATER"],["COLLECT_FERTILIZER"],["WATER"],["EAST"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["SOUTH"],["EAST"],["PLANT","CARROT"],["EAST"],["SOUTH"],["WEST"],["PLANT","CARROT"],["SOUTH"],["FERTILIZE"],["EAST"],["WEST"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["EAST"],["WATER"],["EAST"],["SOUTH"],["COLLECT_FERTILIZER"],["WATER"],["FEED"],["EAST"],["NORTH"],["SOUTH"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["NORTH"],["SOUTH"],["EAST"],["COLLECT_FERTILIZER"],["EAST"],["WEST"],["CARE"],["EAST"],["FERTILIZE"],["SOUTH"],["COLLECT_FERTILIZER"]],"market":[["SELL","MELON",1],["SELL","MELON",1]]},{"farmer":["EAST"],"hands":[["FEED"],["NORTH"],["COLLECT_FERTILIZER"],["SOUTH"],["SOUTH"],["SOUTH"],["WEST"],["NORTH"],["WATER"],["WATER"],["SOUTH"],["EAST"]],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","MILK",10],["SELL","STRAWBERRY",8],["SELL","WOOL",5],["SELL","EGG",1],["SELL","FERTILIZER",4]]},{"farmer":["NORTH"],"hands":[["NORTH"],["HARVEST"],["PICKUP","WHEAT",14],["WEST"],["PICKUP","WHEAT",14]],"market":[["HIRE"],["HIRE"],["HIRE"],["HIRE"],["HIRE"],["SELL","FERTILIZER",1],["SELL","WHEAT",42],["HIRE"]]},{"farmer":["HARVEST"],"hands":[["HARVEST"],["WEST"],["PICKUP","FERTILIZER",5],["HARVEST"],["PICKUP","FERTILIZER",5],["WEST"],["WEST"],["PICKUP","FERTILIZER",5],["PICKUP","FERTILIZER",5],["PICKUP","FERTILIZER",5]],"market":[["HIRE"],["SELL","FERTILIZER",1],["BUY_PRODUCT","WHEAT",18]]},{"farmer":["WEST"],"hands":[["WEST"],["WEST"],["WEST"],["WEST"],["NORTH"],["WEST"],["WEST"],["WEST"],["WATER"],["WEST"],["SOUTH"],["SOUTH"]],"market":[]},{"farmer":["HARVEST"],"hands":[["NORTH"],["HARVEST"],["NORTH"],["HARVEST"],["FEED"],["WEST"],["WEST"],["COLLECT_FERTILIZER"],["WEST"],["WEST"],["WEST"],["NORTH"]],"market":[]},{"farmer":["EAST"],"hands":[["NORTH"],["WEST"],["FEED"],["NORTH"],["CARE"],["HARVEST"],["NORTH"],["SOUTH"],["SOUTH"],["COLLECT_FERTILIZER"],["SOUTH"],["EAST"]],"market":[]},{"farmer":["SOUTH"],"hands":[["NORTH"],["SOUTH"],["CARE"],["WATER"],["WEST"],["EAST"],["NORTH"],["FERTILIZE"],["WATER"],["EAST"],["SOUTH"],["WEST"]],"market":[]},{"farmer":["DROP"],"hands":[["HARVEST"],["WATER"],["WEST"],["NORTH"],["FEED"],["FEED"],["NORTH"],["EAST"],["WEST"],["WEST"],["WATER"],["EAST"]],"market":[]},{"farmer":["EAST"],"hands":[["COLLECT_FERTILIZER"],["WEST"],["FEED"],["WATER"],["CARE"],["CARE"],["HARVEST"],["EAST"],["COLLECT_FERTILIZER"],["SOUTH"],["WEST"],["NORTH"]],"market":[["SELL","MILK",9]]},{"farmer":["EAST"],"hands":[["SOUTH"],["SOUTH"],["CARE"],["WEST"],["WEST"],["SOUTH"],["NORTH"],["EAST"],["WEST"],["SOUTH"],["SOUTH"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["HARVEST"],["NORTH"],["WEST"],["FEED"],["FEED"],["WATER"],["EAST"],["SOUTH"],["FERTILIZE"],["WATER"],["WATER"]],"market":[]},{"farmer":["WATER"],"hands":[["COLLECT_FERTILIZER"],["NORTH"],["CARE"],["WATER"],["NORTH"],["CARE"],["SOUTH"],["EAST"],["SOUTH"],["SOUTH"],["NORTH"],["SOUTH"]],"market":[]},{"farmer":["EAST"],"hands":[["SOUTH"],["FEED"],["EAST"],["EAST"],["NORTH"],["SOUTH"],["COLLECT_FERTILIZER"],["NORTH"],["WATER"],["SOUTH"],["WATER"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["NORTH"],"hands":[["SOUTH"],["CARE"],["NORTH"],["WATER"],["FEED"],["FEED"],["SOUTH"],["NORTH"],["EAST"],["WATER"],["NORTH"],["EAST"]],"market":[]},{"farmer":["COLLECT_FERTILIZER"],"hands":[["DROP"],["COLLECT_FERTILIZER"],["FEED"],["EAST"],["CARE"],["CARE"],["SOUTH"],["NORTH"],["FERTILIZE"],["NORTH"],["NORTH"],["FERTILIZE"]],"market":[["SELL","MELON",1]]},{"farmer":["NORTH"],"hands":[["NORTH"],["EAST"],["CARE"],["EAST"],["EAST"],["EAST"],["FERTILIZE"],["WATER"],["WEST"],["NORTH"],["COLLECT_FERTILIZER"],["EAST"]],"market":[["SELL","MILK",5],["SELL","FERTILIZER",1]]},{"farmer":["FERTILIZE"],"hands":[["COLLECT_FERTILIZER"],["NORTH"],["EAST"],["EAST"],["NORTH"],["EAST"],["EAST"],["SOUTH"],["NORTH"],["NORTH"],["NORTH"],["EAST"]],"market":[["SELL","FERTILIZER",1]]},{"farmer":["EAST"],"hands":[["WEST"],["NORTH"],["EAST"],["SOUTH"],["FEED"],["EAST"],["EAST"],["COLLECT_FERTILIZER"],["WATER"],["NORTH"],["COLLECT_FERTILIZER"],["COLLECT_FERTILIZER"]],"market":[]},{"farmer":["WATER"],"hands":[["COLLECT_FERTILIZER"],["NORTH"],["EAST"],["SOUTH"],["CARE"],["EAST"],["SOUTH"],["SOUTH"],["NORTH"],["HARVEST"],["EAST"],["SOUTH"]],"market":[]},{"farmer":["SOUTH"],"hands":[["NORTH"],["NORTH"],["FEED"],["DROP"],["EAST"],["EAST"],["COLLECT_FERTILIZER"],["FERTILIZE"],["NORTH"],["EAST"],["NORTH"],["WATER"]],"market":[]},{"farmer":["DIG"],"hands":[["FERTILIZE"],["NORTH"],["CARE"],["EAST"],["EAST"],["EAST"],["EAST"],["WEST"],["WATER"],["EAST"],["NORTH"],["EAST"]],"market":[["SELL","MILK",6]]},{"farmer":["WEST"],"hands":[["WATER"],["FEED"],["EAST"],["WATER"],["FEED"],["EAST"],["EAST"],["WEST"],["WEST"],["NORTH"],["NORTH"],["NORTH"]],"market":[]},{"farmer":["WEST"],"hands":[["NORTH"],["CARE"],["SOUTH"],["EAST"],["CARE"],["NORTH"],["WEST"],["DIG"],["WATER"],["WEST"],["COLLECT_FERTILIZER"],["NORTH"]],"market":[]},{"farmer":["DIG"],"hands":[["NORTH"],["EAST"],["FEED"],["DIG"],["COLLECT_FERTILIZER"],["NORTH"],["WEST"],["EAST"],["NORTH"],["SOUTH"],["EAST"],["NORTH"]],"market":[]},{"farmer":["NORTH"],"hands":[],"market":[["SELL","MILK",3],["SELL","STRAWBERRY",2],["SELL","WOOL",3],["SELL","EGG",1],["SELL","FERTILIZER",13],["SELL","WHEAT",38]]},{"farmer":["NORTH"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["WEST"],"hands":[],"market":[]},{"farmer":["NORTH"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["EAST"],"hands":[],"market":[]},{"farmer":["EAST"],"hands":[],"market":[]},{"farmer":["EAST"],"hands":[],"market":[]},{"farmer":["NORTH"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["EAST"],"hands":[],"market":[]},{"farmer":["SOUTH"],"hands":[],"market":[]},{"farmer":["SOUTH"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["EAST"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["EAST"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["NORTH"],"hands":[],"market":[]},{"farmer":["HARVEST"],"hands":[],"market":[]},{"farmer":["WATER"],"hands":[],"market":[]},{"farmer":["PASS"],"hands":[],"market":[]},{"farmer":["PASS"],"hands":[],"market":[]}]''')
# control: the reverse split

def _mix_step(obs, turns_per_day=24):
    """Which step this is.

    The environment writes `day` and `hour` onto every observation, and `step`
    is only a fallback. Getting it wrong is silent and total -- a replay stuck
    on step 0 re-issues the opening turn 720 times and hires on every one.
    """
    day = g(obs, "day", None)
    hour = g(obs, "hour", None)
    if day is not None and hour is not None:
        return int(day) * turns_per_day + int(hour)
    return int(g(obs, "step", 0) or 0)


def agent(obs, config=None):
    # The policy runs on every turn whichever half is used, because it carries
    # the per-turn memory (_MEM) that its own assignment rule reads next turn.
    # Skipping it on the turns the plan owns would quietly change the policy
    # into a different agent, and then the control would not be a control.
    policy = _POLICY_AGENT(obs, config)
    try:
        step = min(max(0, _mix_step(obs)), len(_PLAN) - 1)
        planned = _PLAN[step]
        me = int(g(obs, "player", 0) or 0)
        farms = g(obs, "farms", []) or []
        farm = farms[me] if me < len(farms) else {}
        n_hands = len(g(farm, "hands", []) or [])
        if HALF == "labour":
            hands = [list(h) for h in (planned.get("hands") or [])][:n_hands]
            while len(hands) < n_hands:
                hands.append(["PASS"])
            return {"farmer": list(planned.get("farmer", ["PASS"])),
                    "hands": hands,
                    "market": policy.get("market", [])}
        return {"farmer": policy.get("farmer", ["PASS"]),
                "hands": policy.get("hands", []),
                "market": [list(o) for o in (planned.get("market") or [])][:10]}
    except Exception:
        # A broken plan must not be scored as a broken policy: fall all the way
        # back rather than returning half an action.
        return policy

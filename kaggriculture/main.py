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
LAND_PRICES = [1000, 2000, 4000]

# Units per tile per day. Crops: total harvest divided by days occupied under
# daily watering. Animals: steady state *with* daily CARE, which is how they
# are actually run here (cow = 3 milk / 2 days, sheep = 4 wool / 3 days).
RATE = {"WHEAT": 0.80, "CARROT": 0.75, "TOMATO": 0.33, "STRAWBERRY": 0.24,
        "MELON": 0.55, "EGG": 2.00, "MILK": 1.50, "WOOL": 1.33}
PRODUCER = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}

# How far past the town's drain rate it still pays to supply a product, given
# how hard its glut curve bites. Wool (sq 3.20) and melon (sq 3.60) punish
# oversupply immediately; wheat and egg (log 0.20) barely notice it.
GLUT_TOL = {"WHEAT": 2.0, "EGG": 2.0, "CARROT": 1.6, "TOMATO": 1.6,
            "STRAWBERRY": 1.15, "MILK": 1.15, "WOOL": 0.9, "MELON": 0.9}

P = {
    "max_hands": 12,
    "hands_early": 5,
    "animal_buy_last_day": 22,
    "plant_last_day": {"WHEAT": 26, "CARROT": 27, "MELON": 17,
                       "TOMATO": 21, "STRAWBERRY": 19},
    "cash_buffer": 120,
    "reserve_frac": 0.80,      # never sell under this fraction of base price
    "slice_frac": 0.92,        # ...nor push the live price below this of itself
    "dump_day": 29,
    # (earliest day, cash floor) per quadrant. Land is what caps the whole farm,
    # so the gates are early and the agent saves toward the next one instead of
    # spending its last coin on livestock.
    "land_gate": [(2, 1200), (6, 2600), (10, 5200)],
    "tile_margin": 1.15,       # plan slightly past the tiles we own
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
    plan = {}
    for item, rate in RATE.items():
        gap = demand.get(item, 1.0) * GLUT_TOL[item] - theirs.get(item, 0) * rate
        want = max(0.0, gap) / rate
        plan[item] = want
    # Wheat is feed before it is produce: the herd eats one per head per day.
    plan["WHEAT"] = max(plan.get("WHEAT", 0.0), (herd * 1.25) / RATE["WHEAT"])

    ranked = sorted(RATE, key=lambda i: -price(i) * RATE[i])
    budget_tiles = int(tiles_owned * P["tile_margin"])
    target = {}
    # Feed comes off the top: wheat pays the least per tile but a starved herd
    # loses every animal, so its ration is reserved before the ranking runs.
    feed_tiles = int(min(budget_tiles, math.ceil(herd * 1.25 / RATE["WHEAT"])))
    budget_tiles -= feed_tiles
    for item in ranked:
        if budget_tiles <= 0:
            target[item] = 0
            continue
        take = int(min(plan[item], budget_tiles))
        if item == "WHEAT":
            take += feed_tiles
        target[item] = take
        budget_tiles -= take - (feed_tiles if item == "WHEAT" else 0)

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
    fert_keep = 0 if liquidate else min(40, int(fert_targets * max(1, days_left) / 3) + fert_targets)
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
            # Never sit on stock the shed will discard tonight.
            if sum(shed.values()) > 85:
                qty = max(qty, have // 2)
        if qty > 0:
            sell_orders.append(["SELL", item, qty])

    # 2. Hire. A dozen hands cost ~$376 for the day and add 288 actions.
    if hour <= 1 and not liquidate:
        want = P["max_hands"] if day >= 3 else P["hands_early"]
        room = max(0, want - hires_today)
        for _ in range(min(room, 6)):
            hire_orders.append(["HIRE"])

    # 3. Land, gated on both a day and a cash floor.
    extra = len(unlocked) - 1
    saving_for_land = 0.0
    if extra < len(LAND_PRICES) and not liquidate:
        min_day, min_money = P["land_gate"][extra]
        if day >= min_day and money >= min_money and tiles_used > 0.55 * tiles_owned:
            buy_orders.append(["BUY_LAND"])
            money -= LAND_PRICES[extra]
        elif day >= min_day - 2 and money >= 0.5 * LAND_PRICES[extra]:
            # Hold back the price of the next quadrant instead of sinking the
            # last coin into livestock -- but only once the quadrant is within
            # reach, or the farm saves itself out of seed money on day 0.
            saving_for_land = min(LAND_PRICES[extra], min_money)
    # Seeds are exempt from the land fund: wheat costs $10 and the herd starves
    # without it, which is how v4 lost every cow by day 9.
    spendable = money - P["cash_buffer"] - saving_for_land
    seed_budget = money - P["cash_buffer"]

    # 4. Seeds first, animals second. Livestock outranks every crop per tile,
    #    so if the purchases run the other way the herd eats the whole budget
    #    and the farm ends up buying its own feed at $47 a bushel all season.
    if not liquidate:
        for crop in ("WHEAT", "STRAWBERRY", "TOMATO", "CARROT", "MELON"):
            if day > P["plant_last_day"][crop]:
                continue
            short = deficit(crop) - seeds.get(crop, 0)
            short = min(short, 5)
            cost = CROPS[crop]["seed"]
            if short > 0 and seed_budget >= short * cost:
                buy_orders.append(["BUY_SEED", crop, short])
                money -= short * cost
                seed_budget -= short * cost
                spendable = min(spendable, seed_budget)

    # 5. Animals, best payer first. An animal bought on day 22 still has time
    #    to return its price once; later than that it is a donation. The herd
    #    may not outgrow the feed the farm can actually grow.
    feed_capacity = mine.get("WHEAT", 0) * RATE["WHEAT"]
    if day <= P["animal_buy_last_day"] and not liquidate:
        pending = shed_animals + carried_animals
        room = len(empty_tiles) + len(empty_struct) - pending
        for item in ("MILK", "EGG", "WOOL"):
            a = PRODUCER[item]
            need = deficit(item) - pending
            if need <= 0 or room <= 0:
                continue
            cost = ANIMALS[a]["cost"]
            payback = price(item) * RATE[item] * max(0, days_left - ANIMALS[a]["first_yield_day"])
            if payback < cost * 1.2:
                continue
            # Each extra head eats one wheat a day; buying that on the market
            # costs more than the animal earns once the price climbs.
            wheat_px = prices.get("WHEAT", 25)
            headroom = int(feed_capacity) - (herd + pending)
            # Buy anyway when the feed is genuinely affordable: a month of
            # rations for one head is 30 wheat.
            can_buy_feed = money > cost + 30 * wheat_px
            if headroom <= 0 and wheat_px > 32 and not can_buy_feed:
                continue
            k = 0
            while k < need and k < room and spendable >= cost and k < 4:
                if headroom <= k and wheat_px > 32 and not can_buy_feed:
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
        need = herd - int(shed.get("WHEAT", 0))
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

    def pick_crop():
        """Sow whatever is furthest under plan, weighted by what it earns."""
        # Feed first, always: wheat is the worst tile on the farm by revenue and
        # the only one whose absence kills the herd.
        need_feed = math.ceil((herd + shed_animals + carried_animals) * 1.3 / RATE["WHEAT"])
        if (mine.get("WHEAT", 0) < need_feed and seeds.get("WHEAT", 0) > 0
                and day <= P["plant_last_day"]["WHEAT"]):
            return "WHEAT"
        best, best_val = None, 0.0
        for crop in ("STRAWBERRY", "TOMATO", "MELON", "CARROT", "WHEAT"):
            if seeds.get(crop, 0) <= 0 or day > P["plant_last_day"][crop]:
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
                out.append((unit_price * urgency / (1 + d), (x, y), "FEED"))
            if t.get("yield_units", 0) > 0:
                val = t["yield_units"] * unit_price
                if t["yield_units"] >= a["max_held"]:
                    val *= 2  # production is being thrown away while it sits full
                out.append((val / (1 + d), (x, y), "HARVEST"))
            if t.get("fed_today") and not t.get("cared_today"):
                # One care day = one extra unit on the next production.
                out.append((unit_price * 0.9 / (1 + d), (x, y), "CARE"))
            if t.get("fertilizer_available"):
                out.append((fert_price / (1 + d), (x, y), "COLLECT_FERTILIZER"))

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
                    out.append((unit_price * extra / (1 + d), (x, y), "FERTILIZE"))
            if plant_ready(t):
                out.append((t["yield_units"] * unit_price / (1 + d), (x, y), "HARVEST"))
            elif not t.get("watered_today"):
                age = day - t["planted_day"]
                window = (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]
                if t.get("consecutive_unwatered", 0) >= 1:
                    val = unit_price * 3.0        # dies tonight otherwise
                elif window or cd["ongoing"]:
                    val = unit_price * 1.0        # this watering is a unit of yield
                else:
                    val = unit_price * 0.3
                out.append((val / (1 + d), (x, y), "WATER"))

        if held_animal:
            struct = ANIMALS[held_animal]["structure"]
            for (x, y, t) in empty_struct:
                if (x, y) in claimed or not can_act((x, y)) or t["kind"] != struct:
                    continue
                val = price(ANIMALS[held_animal]["product"]) * RATE[ANIMALS[held_animal]["product"]] * 4
                out.append((val / (1 + dist(pos, (x, y))), (x, y), ("PLACE", held_animal)))

        crop = pick_crop()
        pending_animals = shed_animals + carried_animals
        need_coop = sum(1 for _, _, t in empty_struct if t["kind"] == "COOP")
        need_past = sum(1 for _, _, t in empty_struct if t["kind"] == "PASTURE")
        for (x, y) in empty_tiles:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            if pending_animals > 0:
                if shed.get("GOOSE", 0) + inv.get("GOOSE", 0) > need_coop:
                    out.append((price("EGG") * RATE["EGG"] * 2 / (1 + d), (x, y), "BUILD_COOP"))
                if shed.get("COW", 0) + shed.get("SHEEP", 0) > need_past:
                    out.append((price("MILK") * RATE["MILK"] * 2 / (1 + d), (x, y), "BUILD_PASTURE"))
            if crop:
                out.append((price(crop) * RATE[crop] * 1.5 / (1 + d), (x, y), ("PLANT", crop)))

        # A weed is worth clearing exactly as much as whatever would be planted
        # on the tile it is squatting on; a flat low score leaves half the farm
        # idle by mid-season.
        weed_val = 25.0
        if crop and len(empty_tiles) < 4:
            weed_val = price(crop) * RATE[crop] * 1.4
        for (x, y, t) in weeds:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            out.append((weed_val / (1 + dist(pos, (x, y))), (x, y), "DIG"))

        # Shed trips: fetch feed, fetch an animal, or unload a full pack.
        unfed = sum(1 for _, _, t in animals if not t.get("fed_today"))
        carried = sum(v for v in inv.values() if isinstance(v, int))
        carried_wheat = sum(iv.get("WHEAT", 0) for iv in invs if isinstance(iv, dict))
        carried_fert = sum(iv.get("FERTILIZER", 0) for iv in invs if isinstance(iv, dict))
        for st in sheds:
            d = dist(pos, st)
            if unfed > carried_wheat and unfed > wheat_held and shed.get("WHEAT", 0) > 0:
                out.append((price("MILK") * 1.2 / (1 + d), st,
                            ("PICKUP", "WHEAT", min(8, shed.get("WHEAT", 0)))))
            if fert_targets > carried_fert and fert_held == 0 and shed.get("FERTILIZER", 0) > 0:
                out.append((price("STRAWBERRY") * 1.5 / (1 + d), st,
                            ("PICKUP", "FERTILIZER", min(6, shed.get("FERTILIZER", 0)))))
            if shed_animals > carried_animals and not held_animal:
                a = next(a for a in ANIMALS if shed.get(a, 0) > 0)
                out.append((price(ANIMALS[a]["product"]) * 3 / (1 + d), st, ("PICKUP", a, 1)))
            if carried >= 6 and d <= 2:
                # Produce only earns once it is in the shed and sellable.
                out.append((60.0 / (1 + d), st, "DROP"))
        return out

    def resolve(pos, target_tile, op):
        if pos != target_tile:
            mv = step_toward(pos, target_tile)
            return [mv] if mv else ["PASS"]
        return list(op) if isinstance(op, tuple) else [op]

    actions = []
    for i, pos in enumerate(units):
        cand = jobs_for(pos, unit_inv(i))
        if not cand:
            actions.append(["PASS"])
            continue
        cand.sort(key=lambda c: -c[0])
        _, tile, op = cand[0]
        if tile not in shed_here:
            claimed.add(tile)
        actions.append(resolve(pos, tile, op))

    return {"farmer": actions[0] if actions else ["PASS"],
            "hands": actions[1:],
            "market": orders}

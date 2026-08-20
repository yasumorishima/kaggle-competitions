"""Kaggriculture agent.

Design notes (why this shape, from the published rules + kaggriculture.py):

* Labour is nearly free. A hand costs fib(n) coins for the day (1,1,2,3,5,8,...),
  so a dozen hands cost ~$376/day while each hand adds 24 actions. The binding
  constraints are tiles, capital and *market depth* -- never the wage bill.
* Geese are the engine. EGG's glut curve is `log` at target 0.20, so the price
  only sags from $50 to ~$39 after a thousand units sold; every other premium
  good (melon, strawberry, milk, wool) hits the $1 floor within ~100-150 units.
  A fed+cared goose lays 2 eggs/day and hands over 1 fertilizer/day for free.
* Fertilizer is a second income stream nobody has to grow: every surviving
  animal makes one per day whether fed or not, and it sells at $100 base.
* Wheat is feed, not produce. Buying it drives its own price up (`sqrt` on the
  scarcity side), so we grow our own and only buy to cover a shortfall.
* Selling is a scheduling problem: price is a known function of market
  inventory, so we can compute exactly how many units can be sold before the
  price drops under a reserve, and sell only that many.
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
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

# Units harvested per tile per day under daily watering, straight from the
# rules table; used to rank crops by revenue rather than by sticker price.
YIELD_RATE = {"WHEAT": 0.80, "CARROT": 0.75, "TOMATO": 0.33,
              "STRAWBERRY": 0.24, "MELON": 0.55}

# Tuning knobs. Kept together so the eval harness can sweep them.
P = {
    "max_hands": 12,          # hands hired per day
    "goose_target": 22,       # coops we want running
    "cow_target": 3,
    "sheep_target": 3,
    "wheat_min": 8,           # feed floor before the herd exists
    "wheat_cap": 30,
    # Per-crop tile ceilings. These are market-depth limits, not space limits:
    # melon and strawberry reach the $1 floor after ~100-150 units sold.
    "crop_cap": {"WHEAT": 30, "MELON": 20, "STRAWBERRY": 8, "CARROT": 14, "TOMATO": 8},
    "animal_buy_last_day": 20,   # after this, an animal cannot pay itself back
    "plant_last_day": {"WHEAT": 26, "CARROT": 27, "MELON": 19, "TOMATO": 20, "STRAWBERRY": 18},
    "cash_buffer": 250,
    "reserve_frac_early": 0.85,   # sell only above this fraction of base price
    "reserve_frac_late": 0.45,
    "dump_day": 29,               # final day: liquidate everything
    "land_gate": [(1, 2200), (4, 4500), (7, 8500)],  # (min day, min money) per quadrant
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
    """How many units can be sold before the marginal price falls under reserve.

    The market takes one unit at a time and re-prices after each, so this walks
    the curve instead of trusting the currently quoted price.
    """
    n = 0
    cur = inv
    while n < have:
        if price_at(item, cur) < reserve:
            break
        n += 1
        cur += 1
    return n


def crop_value(crop, inventory):
    """Revenue per tile-day at the price the market is quoting right now.

    Uses the live market inventory rather than the base price, so a hinge good
    the town has drained (carrot, tomato) outranks a premium good the players
    have already glutted.
    """
    inv = int(inventory.get(crop, MARKET_I0)) if inventory else MARKET_I0
    return price_at(crop, inv) * YIELD_RATE[crop]


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(fr, to):
    """One move that reduces Manhattan distance, or None when already there."""
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


# --------------------------------------------------------------- the agent
def agent(obs, config=None):
    me = g(obs, "player", 0)
    day = int(g(obs, "day", 0))
    hour = int(g(obs, "hour", 0))
    farms = g(obs, "farms", []) or []
    if not farms or me >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[me]
    private = g(obs, "private", {}) or {}
    market = g(obs, "market", {}) or {}
    prices = g(market, "prices", {}) or {}
    inventory = g(market, "inventory", {}) or {}
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

    # ---------------------------------------------------------- farm census
    animals, plants, weeds, empty_struct, empty_tiles = [], [], [], [], []
    counts = {"COOP": 0, "PASTURE": 0}
    crop_counts = {c: 0 for c in CROPS}
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t == "LOCKED" or t is None:
                if t is None:
                    empty_tiles.append((x, y))
                continue
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "PLANT":
                plants.append((x, y, t))
                crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
            elif kind == "WEED":
                weeds.append((x, y, t))
            elif kind in ("COOP", "PASTURE"):
                counts[kind] += 1
                if "animal" in t:
                    animals.append((x, y, t))
                else:
                    empty_struct.append((x, y, t))

    n_geese = sum(1 for _, _, t in animals if t["animal"] == "GOOSE")
    n_cows = sum(1 for _, _, t in animals if t["animal"] == "COW")
    n_sheep = sum(1 for _, _, t in animals if t["animal"] == "SHEEP")
    shed_animals = sum(shed.get(a, 0) for a in ANIMALS)
    carried_animals = sum(sum(iv.get(a, 0) for a in ANIMALS) for iv in invs if isinstance(iv, dict))

    # ------------------------------------------------------- market actions
    # Only 10 orders per turn are processed, so they are built in priority
    # groups: hiring must land at dawn or the day's labour is lost, sales fund
    # everything else and are processed before the purchases behind them.
    hire_orders, sell_orders, buy_orders = [], [], []
    orders = []
    liquidate = day >= P["dump_day"]
    frac = P["reserve_frac_late"] if day >= 22 else P["reserve_frac_early"]

    # 1. Sell produce. Wheat is feed first: only the surplus over the herd's
    #    remaining needs is ever sold.
    herd = n_geese + n_cows + n_sheep
    wheat_reserve_units = 0 if liquidate else herd * max(0, min(3, last_day - day))
    for item in ("EGG", "MILK", "WOOL", "MELON", "STRAWBERRY", "TOMATO",
                 "CARROT", "FERTILIZER", "WHEAT"):
        have = int(shed.get(item, 0))
        if item == "WHEAT":
            have = max(0, have - wheat_reserve_units)
        if have <= 0:
            continue
        if liquidate:
            qty = have
        else:
            reserve = MARKET_PARAMS[item]["base"] * frac
            qty = sellable_qty(item, int(inventory.get(item, MARKET_I0)), have, reserve)
        if qty > 0:
            sell_orders.append(["SELL", item, qty])

    # 2. Hire. Hands are the cheapest resource in the game; hire at dawn so
    #    they get a full day of actions.
    if hour <= 1 and not liquidate:
        want = P["max_hands"] if day >= 1 else 4
        room = max(0, want - hires_today)
        for _ in range(min(room, 6 if hour == 0 else 6)):
            hire_orders.append(["HIRE"])

    # 3. Land. Buying early only helps if we can afford to stock the tiles, so
    #    each quadrant is gated on both a day and a cash floor.
    extra = len(unlocked) - 1
    if extra < len(LAND_PRICES) and not liquidate:
        min_day, min_money = P["land_gate"][extra]
        if day >= min_day and money >= min_money:
            buy_orders.append(["BUY_LAND"])
            money -= LAND_PRICES[extra]

    # 4. Animals. Each needs a structure eventually; buy while there is still
    #    enough season left for the bird to pay for itself.
    if day <= P["animal_buy_last_day"] and not liquidate:
        pending = shed_animals + carried_animals
        free_tiles = len(empty_tiles) + len(empty_struct)
        budget = money - P["cash_buffer"]
        wish = []
        if n_geese + pending < P["goose_target"]:
            wish.append("GOOSE")
        if n_cows < P["cow_target"] and day >= 2:
            wish.append("COW")
        if n_sheep < P["sheep_target"] and day >= 2:
            wish.append("SHEEP")
        for a in wish:
            cost = ANIMALS[a]["cost"]
            k = 0
            while budget >= cost and pending + k < free_tiles and k < 3:
                k += 1
                budget -= cost
            if k:
                buy_orders.append(["BUY_ANIMAL", a, k])
                money = budget + P["cash_buffer"]
                pending += k

    # 5. Seeds. Wheat is sized off the herd's appetite; the cash crops are
    #    ranked by revenue per tile-day at the *current* price, which is what
    #    makes the hinge goods (carrot, tomato) worth planting exactly when the
    #    town has drained them and nobody else is supplying.
    wheat_need = 0 if not herd else max(6, min(P["wheat_cap"],
                                               int(herd * 1.3 / YIELD_RATE["WHEAT"]) + 1))
    crop_rank = sorted(
        (c for c in ("MELON", "STRAWBERRY", "CARROT", "TOMATO")
         if day <= P["plant_last_day"].get(c, 25)),
        key=lambda c: -crop_value(c, inventory))
    if not liquidate:
        want_seeds = {"WHEAT": max(P["wheat_min"], wheat_need)}
        for rank, c in enumerate(crop_rank):
            want_seeds[c] = P["crop_cap"][c] if rank < 2 else max(2, P["crop_cap"][c] // 3)
        for crop, target in want_seeds.items():
            if day > P["plant_last_day"].get(crop, 25):
                continue
            deficit = target - crop_counts.get(crop, 0) - seeds.get(crop, 0)
            deficit = min(deficit, 6)
            cost = CROPS[crop]["seed"]
            if deficit > 0 and money - P["cash_buffer"] >= deficit * cost:
                buy_orders.append(["BUY_SEED", crop, deficit])
                money -= deficit * cost

    # 6. Emergency feed. Losing an animal costs far more than an expensive
    #    wheat unit, so top up from the market when the harvest lags.
    if herd and not liquidate:
        need = herd - int(shed.get("WHEAT", 0))
        if need > 0 and prices.get("WHEAT", 25) <= 60 and money - P["cash_buffer"] > 0:
            k = int(min(need, (money - P["cash_buffer"]) // max(1, prices.get("WHEAT", 25))))
            if k > 0:
                buy_orders.append(["BUY_PRODUCT", "WHEAT", k])

    # Sales run before purchases so the coins they raise are available to the
    # buy orders queued behind them in the same turn.
    orders = (hire_orders + sell_orders + buy_orders)[:10]

    # ---------------------------------------------------------- unit actions
    claimed = set()
    sheds = shed_tiles(n)
    shed_here = set(sheds)

    def can_act(pos):
        return quadrant_of(pos[0], pos[1], n) in unlocked

    def unit_inv(i):
        iv = invs[i] if i < len(invs) and isinstance(invs[i], dict) else {}
        return iv

    def plant_ready(t, x, y):
        cd = CROPS[t["crop"]]
        age = day - t["planted_day"]
        if cd["ongoing"]:
            return t.get("yield_units", 0) > 0 and age >= cd["first_yield_day"]
        return t.get("yield_units", 0) > 0 and age >= cd["max_yield_day"]

    def pick_crop():
        """Next crop to sow: feed first, then the best revenue per tile-day."""
        if (crop_counts.get("WHEAT", 0) < wheat_need and seeds.get("WHEAT", 0) > 0
                and day <= P["plant_last_day"]["WHEAT"]):
            return "WHEAT"
        best, best_val = None, 0.0
        for crop in ("MELON", "STRAWBERRY", "CARROT", "TOMATO", "WHEAT"):
            if seeds.get(crop, 0) <= 0 or day > P["plant_last_day"].get(crop, 25):
                continue
            if crop_counts.get(crop, 0) >= P["crop_cap"][crop]:
                continue
            val = crop_value(crop, inventory)
            if val > best_val:
                best, best_val = crop, val
        return best

    def jobs_for(pos, inv):
        """Candidate (score, target, op) for one unit, scored by value/distance."""
        out = []
        wheat_held = inv.get("WHEAT", 0)
        held_animal = next((a for a in ANIMALS if inv.get(a, 0) > 0), None)

        for (x, y, t) in animals:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            if not t.get("fed_today") and wheat_held > 0:
                # An unfed animal on its second day is gone for good.
                urgency = 400 if t.get("consecutive_unfed", 0) >= 1 else 150
                out.append((urgency / (1 + d), (x, y), "FEED"))
            if t.get("yield_units", 0) > 0:
                a = ANIMALS[t["animal"]]
                val = t["yield_units"] * MARKET_PARAMS[a["product"]]["base"] * 0.5
                if t["yield_units"] >= a["max_held"]:
                    val *= 2  # production is being wasted while the tile is full
                out.append((val / (1 + d), (x, y), "HARVEST"))
            if t.get("fertilizer_available"):
                out.append((70.0 / (1 + d), (x, y), "COLLECT_FERTILIZER"))
            if t.get("fed_today") and not t.get("cared_today"):
                out.append((45.0 / (1 + d), (x, y), "CARE"))

        for (x, y, t) in plants:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            if plant_ready(t, x, y):
                val = t["yield_units"] * MARKET_PARAMS[t["crop"]]["base"] * 0.6
                out.append((val / (1 + d), (x, y), "HARVEST"))
            elif not t.get("watered_today"):
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                window = (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]
                urgency = 120 if t.get("consecutive_unwatered", 0) >= 1 else (
                    80 if (window or cd["ongoing"]) else 35)
                out.append((urgency / (1 + d), (x, y), "WATER"))

        if held_animal:
            for (x, y, t) in empty_struct:
                if (x, y) in claimed or not can_act((x, y)):
                    continue
                if t["kind"] != ANIMALS[held_animal]["structure"]:
                    continue
                out.append((300.0 / (1 + dist(pos, (x, y))), (x, y), ("PLACE", held_animal)))

        crop = pick_crop()
        pending_animals = shed_animals + carried_animals
        for (x, y) in empty_tiles:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            d = dist(pos, (x, y))
            if pending_animals > 0:
                need_coop = sum(1 for _, _, t in empty_struct if t["kind"] == "COOP")
                if shed.get("GOOSE", 0) + inv.get("GOOSE", 0) > need_coop:
                    out.append((90.0 / (1 + d), (x, y), "BUILD_COOP"))
                need_past = sum(1 for _, _, t in empty_struct if t["kind"] == "PASTURE")
                if shed.get("COW", 0) + shed.get("SHEEP", 0) > need_past:
                    out.append((85.0 / (1 + d), (x, y), "BUILD_PASTURE"))
            if crop:
                gain = MARKET_PARAMS[crop]["base"] * 0.25
                out.append((gain / (1 + d), (x, y), ("PLANT", crop)))

        for (x, y, t) in weeds:
            if (x, y) in claimed or not can_act((x, y)):
                continue
            out.append((12.0 / (1 + dist(pos, (x, y))), (x, y), "DIG"))

        # Trips to the shed: fetch feed, fetch an animal, or drop a full load.
        unfed = sum(1 for _, _, t in animals if not t.get("fed_today"))
        carried = sum(v for k, v in inv.items() if isinstance(v, int))
        # Fetch trips are counted farm-wide: without this every idle hand walks
        # to the shed for the same sack of wheat or the same goose.
        carried_wheat = sum(iv.get("WHEAT", 0) for iv in invs if isinstance(iv, dict))
        for st in sheds:
            d = dist(pos, st)
            if unfed > carried_wheat and unfed > wheat_held and shed.get("WHEAT", 0) > 0:
                out.append((130.0 / (1 + d), st, ("PICKUP", "WHEAT",
                                                  min(6, shed.get("WHEAT", 0)))))
            if shed_animals > carried_animals and not held_animal:
                a = next(a for a in ANIMALS if shed.get(a, 0) > 0)
                out.append((160.0 / (1 + d), st, ("PICKUP", a, 1)))
            if carried >= 8 and d <= 2:
                out.append((60.0 / (1 + d), st, "DROP"))
        return out

    def resolve(pos, target, op):
        """Either walk one tile toward the job or perform it."""
        if pos != target:
            mv = step_toward(pos, target)
            return [mv] if mv else ["PASS"]
        if isinstance(op, tuple):
            return list(op)
        return [op]

    actions = []
    for i, pos in enumerate(units):
        inv = unit_inv(i)
        cand = jobs_for(pos, inv)
        if not cand:
            actions.append(["PASS"])
            continue
        cand.sort(key=lambda c: -c[0])
        score, target, op = cand[0]
        # Shed tiles hold no crop, so several units may share them.
        if target not in shed_here:
            claimed.add(target)
        actions.append(resolve(pos, target, op))

    return {"farmer": actions[0] if actions else ["PASS"],
            "hands": actions[1:],
            "market": orders}

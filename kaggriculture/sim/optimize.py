#!/usr/bin/env python
"""Search over the action list itself, because that is what this competition is.

Every published agent above the 100k mark is a replay: a fixed 720-step action
list with a thin repair layer. Decoding one made the reason concrete -- it
spends 43% of its actions walking against this farm's 63%, and does 2.07 jobs
each time it stops against 1.33 -- and eleven attempts at fixing that from
inside a per-turn policy all came back a tie or worse. A per-turn policy cannot
see it: which tile a hand should be standing on at hour 9 is a question about
the whole day, and the policy only ever sees the hour.

So the plan becomes the thing that gets optimised, and this is the optimiser.
It is a (1+lambda) hill climb: hold an incumbent plan, propose lambda mutated
copies, keep the best if it beats the incumbent on the same seeds.

Two properties of this environment make that work, and both were checked rather
than assumed (--selftest checks the first one on every run):

* Scoring is deterministic. Given a seed, the board, the shop draw and the weed
  spawns are fixed, and a replay takes no decisions, so the same plan scores
  exactly the same money twice. There is no evaluation noise to average away,
  which is the usual thing that makes plan search hopeless.
* Comparisons use common random numbers. Parent and child are scored on the
  identical seed list, so the season cancels out of the difference the same way
  the paired-seed design cancels it in evaluate.py.

Mutations are structural, never random bytes. A unit's action stream is
(walk, walk, work, work, ...), and its position at any step is decided purely
by the order of its moves. So inserting or deleting a step inside one day
shifts *when* the rest of that day happens while leaving *where* it happens
untouched -- the plan stays valid by construction, and the operator attacks the
walking gap directly. See MUTATIONS for what each one does.

Held-out seeds are scored every --confirm-every generations. A plan search can
memorise its training seasons -- learning where one seed's weeds are is worth
real money and transfers to nothing -- and the held-out line is what makes that
visible rather than a surprise on the leaderboard.

Usage:
    python sim/optimize.py --plan plan.json --minutes 60 --out best.json
    python sim/optimize.py --resume ckpt.json --minutes 300 --out best.json
    python sim/optimize.py --plan plan.json --selftest
"""
import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

TURNS = 24
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
IDLE = "PASS"
MARKET_CAP = 10


# --------------------------------------------------------------------------
# the replay that gets scored
# --------------------------------------------------------------------------

def _get(o, k, d=None):
    if o is None:
        return d
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)


def _which_step(obs):
    """Recover the step from day/hour, which the environment sets every turn.

    `step` is only a fallback. Reading it wrong is silent and total: a replay
    stuck on step 0 re-issues the opening turn 720 times and ends with nothing.
    """
    day, hour = _get(obs, "day"), _get(obs, "hour")
    if day is not None and hour is not None:
        return int(day) * TURNS + int(hour)
    return int(_get(obs, "step", 0) or 0)


def make_replay(plan, repair="dig"):
    """A callable agent that plays `plan`, optionally digging weeds it meets.

    The repair exists because weeds are the one thing the season randomises
    that a fixed list cannot absorb: PLANT and BUILD_PASTURE both fail outright
    on a weed tile, so the action is spent for nothing and everything the plan
    expected to grow there never does. Digging costs the unit one step and
    re-issues the intended action next turn, which is the whole of it.
    """
    state = {"pending": {}, "last": -1}

    def agent(obs, config=None):
        try:
            step = min(max(0, _which_step(obs)), len(plan) - 1)
            planned = plan[step]
            me = int(_get(obs, "player", 0) or 0)
            farms = _get(obs, "farms", []) or []
            farm = farms[me] if me < len(farms) else {}
            positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
            n_hands = max(0, len(positions) - 1)

            units = [list(planned.get("farmer") or [IDLE])]
            hands = [list(h) for h in (planned.get("hands") or [])][:n_hands]
            while len(hands) < n_hands:
                hands.append([IDLE])
            units += hands

            if repair == "dig":
                if step <= state["last"]:      # a fresh episode in this process
                    state["pending"], state["last"] = {}, -1
                state["last"] = step
                for slot, intended in list(state["pending"].items()):
                    if slot < len(units):
                        units[slot] = list(intended)
                    state["pending"].pop(slot, None)
                for slot, unit in enumerate(units):
                    if not unit or unit[0] not in ("PLANT", "BUILD_PASTURE"):
                        continue
                    pos = positions[slot] if slot < len(positions) else None
                    try:
                        tile = (_get(farm, "tiles", []) or [])[int(pos[1])][int(pos[0])]
                    except (IndexError, TypeError, ValueError):
                        tile = None
                    if isinstance(tile, dict) and tile.get("kind") == "WEED":
                        state["pending"][slot] = list(unit)
                        units[slot] = ["DIG"]

            return {"farmer": units[0],
                    "hands": units[1:],
                    "market": [list(o) for o in (planned.get("market") or [])][:MARKET_CAP]}
        except Exception:
            farms = _get(obs, "farms", []) or []
            me = int(_get(obs, "player", 0) or 0)
            farm = farms[me] if me < len(farms) else {}
            return {"farmer": [IDLE],
                    "hands": [[IDLE] for _ in (_get(farm, "hands", []) or [])],
                    "market": []}

    return agent


def play(job):
    """One episode. job = (plan, opponent, seed, steps, side, repair) -> money."""
    from kaggle_environments import make

    plan, opponent, seed, steps, side, repair = job
    me = make_replay(plan, repair)
    order = [me, opponent] if side == 0 else [opponent, me]
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.run(order)
    return float(env.steps[-1][side].reward or 0)


def score(pool, plan, opponent, seeds, sides, steps, repair):
    jobs = [(plan, opponent, s, steps, side, repair)
            for s in seeds for side in sides]
    mapper = pool.map if pool is not None else map
    vals = list(mapper(play, jobs))
    return sum(vals) / len(vals), vals


# --------------------------------------------------------------------------
# plan surgery -- every operator leaves a structurally valid plan
# --------------------------------------------------------------------------

def n_units(action):
    return 1 + len(action.get("hands") or [])


def get_unit(action, slot):
    if slot == 0:
        return list(action.get("farmer") or [IDLE])
    hands = action.get("hands") or []
    i = slot - 1
    return list(hands[i]) if i < len(hands) else None


def set_unit(action, slot, op):
    if slot == 0:
        action["farmer"] = list(op)
        return
    hands = action.setdefault("hands", [])
    i = slot - 1
    while len(hands) <= i:
        hands.append([IDLE])
    hands[i] = list(op)


def live_window(plan, slot, day):
    """The steps of `day` where `slot` actually exists, as a list of indices.

    A hand hired on day 9 has no stream before it, and writing into those steps
    would conjure a worker the farm has not paid for -- the replay would pad it
    away, so the mutation would read as a silent no-op rather than an error.
    """
    lo, hi = day * TURNS, min((day + 1) * TURNS, len(plan))
    return [t for t in range(lo, hi) if get_unit(plan[t], slot) is not None]


def pick_stream(plan, rng):
    """A (slot, [steps]) pair to operate on."""
    for _ in range(20):
        day = rng.randrange(0, max(1, len(plan) // TURNS))
        base = day * TURNS
        if base >= len(plan):
            continue
        slot = rng.randrange(0, n_units(plan[base]))
        win = live_window(plan, slot, day)
        if len(win) >= 4:
            return slot, win
    return 0, list(range(0, min(TURNS, len(plan))))


def op_delay(plan, rng):
    """Insert a wait inside one unit-day; the rest of that day slides later.

    The order of the unit's moves is untouched, so every later action in the
    day still happens on the tile it was recorded on -- only its hour changes.
    That is the point: it lets the search retime work against the market and
    the animals' production days without breaking the route.
    """
    slot, win = pick_stream(plan, rng)
    p = rng.randrange(0, len(win) - 1)
    ops = [get_unit(plan[t], slot) for t in win]
    ops = ops[:p] + [[IDLE]] + ops[p:-1]
    for t, op in zip(win, ops):
        set_unit(plan[t], slot, op)
    return "delay/slot%d" % min(slot, 3)


def op_advance(plan, rng):
    """Delete one action from a unit-day; the rest slides earlier.

    The mirror of op_delay, and the operator that attacks the walking gap head
    on -- a wasted step removed here compounds for the rest of the day.
    """
    slot, win = pick_stream(plan, rng)
    ops = [get_unit(plan[t], slot) for t in win]
    idle = [i for i, o in enumerate(ops) if o and o[0] == IDLE]
    p = rng.choice(idle) if idle and rng.random() < 0.7 else rng.randrange(0, len(ops))
    ops = ops[:p] + ops[p + 1:] + [[IDLE]]
    for t, op in zip(win, ops):
        set_unit(plan[t], slot, op)
    return "advance/slot%d" % min(slot, 3)


def op_repeat(plan, rng):
    """Turn an idle step into a repeat of what the unit last did in place.

    Raises jobs-per-arrival without moving anything: a hand already standing at
    an animal can care for it again instead of waiting. The published plan does
    967 CARE actions to this farm's 169, and the difference is mostly this.
    """
    slot, win = pick_stream(plan, rng)
    ops = [get_unit(plan[t], slot) for t in win]
    spots = [i for i, o in enumerate(ops) if o and o[0] == IDLE and i > 0]
    if not spots:
        return None
    i = rng.choice(spots)
    prev = None
    for j in range(i - 1, -1, -1):
        cand = ops[j]
        if cand and cand[0] not in MOVES and cand[0] != IDLE:
            prev = cand
            break
    if prev is None:
        return None
    set_unit(plan[win[i]], slot, prev)
    return "repeat"


def op_retarget(plan, rng):
    """Swap the argument of one action for another argument the plan uses.

    Only arguments seen elsewhere under the same verb are offered, so PLANT
    never gets handed an animal and the action stays one the farm can take.
    """
    vocab = {}
    for action in plan:
        for slot in range(n_units(action)):
            op = get_unit(action, slot)
            if op and len(op) >= 2 and isinstance(op[1], str):
                vocab.setdefault(op[0], set()).add(op[1])
    slot, win = pick_stream(plan, rng)
    cands = []
    for t in win:
        op = get_unit(plan[t], slot) or [IDLE]
        if op[0] in vocab and len(vocab[op[0]]) > 1:
            cands.append(t)
    if not cands:
        return None
    t = rng.choice(cands)
    op = list(get_unit(plan[t], slot))
    choices = sorted(vocab[op[0]] - {op[1]})
    op[1] = rng.choice(choices)
    set_unit(plan[t], slot, op)
    return "retarget/%s" % op[0]


def _market_steps(plan):
    return [t for t, a in enumerate(plan) if a.get("market")]


def op_market_shift(plan, rng):
    """Move one market order a few steps earlier or later.

    Timing is most of a sale's value here: the town takes a fixed number of
    units every four steps, so an order that lands on a step where demand has
    already been met sells into a floor price.
    """
    steps = _market_steps(plan)
    if not steps:
        return None
    t = rng.choice(steps)
    orders = plan[t]["market"]
    i = rng.randrange(0, len(orders))
    d = rng.choice((-3, -2, -1, 1, 2, 3))
    u = t + d
    if not 0 <= u < len(plan):
        return None
    dest = plan[u].setdefault("market", [])
    if len(dest) >= MARKET_CAP:
        return None
    dest.append(list(orders.pop(i)))
    return "market_shift/%+d" % d


def op_market_qty(plan, rng):
    """Resize one order. Selling less now keeps stock for a better hour."""
    steps = _market_steps(plan)
    if not steps:
        return None
    t = rng.choice(steps)
    order = rng.choice(plan[t]["market"])
    if len(order) < 3:
        return None
    try:
        q = int(order[2])
    except (TypeError, ValueError):
        return None
    new = max(1, q + rng.choice((-4, -2, -1, 1, 2, 4)))
    if new == q:
        return None
    order[2] = new
    return "market_qty"


def op_market_drop(plan, rng):
    steps = _market_steps(plan)
    if not steps:
        return None
    t = rng.choice(steps)
    plan[t]["market"].pop(rng.randrange(0, len(plan[t]["market"])))
    return "market_drop"


def op_market_add(plan, rng):
    """Copy an order the plan already places onto a step that has room.

    Without this the order list can only ever shrink -- shift, resize and drop
    all leave the count the same or smaller -- so the search is a ratchet that
    can spend a herd but never buy one. The copied order is one the plan
    already issues, so it stays an order the farm knows how to place.
    """
    steps = _market_steps(plan)
    if not steps:
        return None
    src = rng.choice(steps)
    order = list(rng.choice(plan[src]["market"]))
    t = rng.randrange(0, len(plan))
    dest = plan[t].setdefault("market", [])
    if len(dest) >= MARKET_CAP:
        return None
    dest.append(order)
    return "market_add/%s" % order[0]


def op_market_retarget(plan, rng):
    """Point an order at a different item of the same kind.

    This is the one operator that can change the shape of the economy rather
    than its timing: BUY_ANIMAL GOOSE becomes BUY_ANIMAL COW, and the season
    that follows is a different farm. Everything else here retimes what the
    plan already decided, which is why the herd could never move.

    Items are only ever swapped for items the plan uses under the same verb, so
    an order stays one the environment will accept.
    """
    vocab = {}
    for action in plan:
        for order in action.get("market") or []:
            if len(order) >= 2 and isinstance(order[1], str):
                vocab.setdefault(order[0], set()).add(order[1])
    steps = [t for t in _market_steps(plan)
             if any(len(o) >= 2 and len(vocab.get(o[0], ())) > 1
                    for o in plan[t]["market"])]
    if not steps:
        return None
    t = rng.choice(steps)
    cands = [o for o in plan[t]["market"]
             if len(o) >= 2 and len(vocab.get(o[0], ())) > 1]
    order = rng.choice(cands)
    order[1] = rng.choice(sorted(vocab[order[0]] - {order[1]}))
    return "market_retarget/%s" % order[0]


MUTATIONS = [
    (op_delay, 3),
    (op_advance, 4),
    (op_repeat, 4),
    (op_retarget, 2),
    (op_market_shift, 3),
    (op_market_qty, 2),
    (op_market_drop, 1),
    (op_market_add, 3),
    (op_market_retarget, 3),
]


def mutate(plan, rng, n_ops):
    """A mutated copy plus the names of the operators that produced it.

    Applying several operators at once is deliberate. A single edit is usually
    worth less than the smallest amount of money the environment can resolve,
    so a one-edit-per-generation climb spends its whole budget on exact ties.
    """
    child = copy.deepcopy(plan)
    names = []
    fns = [f for f, _w in MUTATIONS]
    weights = [w for _f, w in MUTATIONS]
    for _ in range(n_ops):
        fn = rng.choices(fns, weights=weights)[0]
        got = fn(child, rng)
        if got:
            names.append(got)
    return child, names


# --------------------------------------------------------------------------
# the climb
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", help="starting plan(s), comma separated. Each may "
                                   "be a plan JSON or an agent .py that embeds "
                                   "one; the best on the training seeds is used")
    ap.add_argument("--resume", help="checkpoint JSON to continue from")
    ap.add_argument("--out", default="best_plan.json")
    ap.add_argument("--checkpoint", default="", help="written every generation")
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--seeds", default="3000,3001", help="training seeds")
    ap.add_argument("--holdout", default="3100,3101,3102,3103")
    ap.add_argument("--sides", default="0", help="seats to score, e.g. 0,1")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--repair", default="dig", choices=("dig", "none"))
    ap.add_argument("--lam", type=int, default=4, help="candidates per generation")
    ap.add_argument("--ops", type=int, default=6, help="edits per candidate")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--confirm-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true",
                    help="only check that scoring is deterministic, then exit")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    holdout = [int(s) for s in args.holdout.split(",") if s.strip()]
    sides = [int(s) for s in args.sides.split(",") if s.strip()]

    stats = {}
    gen = 0
    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            ck = json.load(f)
        plan, gen, stats = ck["plan"], int(ck.get("gen", 0)), ck.get("stats", {})
        print("RESUMED gen=%d score=%s" % (gen, ck.get("score")), flush=True)
    elif args.plan:
        plan = None
    else:
        ap.error("one of --plan or --resume is required")

    rng = random.Random(args.seed + gen)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        t0 = time.time()
        if plan is None:
            # A plan recorded on one season is that season's plan: it knows
            # where those weeds were and which shops opened, and the first
            # smoke run scored 78,083 on the seed it was recorded from against
            # 28,908 on seeds it had never seen. Which recording a climb starts
            # from therefore decides most of where it can get to, so several
            # are scored on the training seeds and the most transferable one
            # wins the start.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import route_shape
            best = None
            for path in [p.strip() for p in args.plan.split(",") if p.strip()]:
                cand = route_shape.load_plan(path)
                s, vs = score(pool, cand, args.opponent, seeds, sides,
                              args.steps, args.repair)
                print("CANDIDATE %-28s score=%.0f values=%s"
                      % (path, s, [round(v) for v in vs]), flush=True)
                if best is None or s > best[0]:
                    best = (s, cand, path)
            plan, chosen = best[1], best[2]
            print("CHOSE " + chosen, flush=True)
        base, vals = score(pool, plan, args.opponent, seeds, sides, args.steps, args.repair)
        per = (time.time() - t0) / max(1, len(vals))
        print("START score=%.0f per-episode=%.1fs values=%s"
              % (base, per, [round(v) for v in vals]), flush=True)

        if args.selftest:
            # If the same plan does not score the same money twice, every
            # comparison below is measuring the season rather than the edit,
            # and the whole design has to change. Cheap to check, fatal to
            # assume.
            again, vals2 = score(pool, plan, args.opponent, seeds, sides,
                                 args.steps, args.repair)
            ok = all(abs(a - b) < 1e-6 for a, b in zip(vals, vals2))
            print("SELFTEST deterministic=%s first=%s second=%s"
                  % (ok, [round(v) for v in vals], [round(v) for v in vals2]),
                  flush=True)
            print("VERDICT=" + ("DETERMINISTIC" if ok else "STOCHASTIC"))
            return 0

        hold, hvals = score(pool, plan, args.opponent, holdout, sides, args.steps, args.repair)
        # The per-seed values, not just their mean: a plan that scores 78,000
        # on one season and 29,000 on the next is not a 53,000 plan, and the
        # mean is the one number that hides which of the two it is.
        print("HOLDOUT gen=%d score=%.0f values=%s"
              % (gen, hold, [round(v) for v in hvals]), flush=True)

        deadline = time.time() + args.minutes * 60.0
        best_hold = hold
        while time.time() < deadline:
            gen += 1
            kids = [mutate(plan, rng, args.ops) for _ in range(args.lam)]
            jobs = [(child, args.opponent, s, args.steps, side, args.repair)
                    for child, _n in kids for s in seeds for side in sides]
            mapper = pool.map if pool is not None else map
            flat = list(mapper(play, jobs))
            per_kid = len(seeds) * len(sides)
            scores = [sum(flat[i * per_kid:(i + 1) * per_kid]) / per_kid
                      for i in range(len(kids))]

            top = max(range(len(kids)), key=lambda i: scores[i])
            for i, (_child, names) in enumerate(kids):
                for nm in set(names):
                    s = stats.setdefault(nm, {"tried": 0, "kept": 0, "gain": 0.0})
                    s["tried"] += 1
                    if i == top and scores[i] > base:
                        s["kept"] += 1
                        s["gain"] += scores[i] - base

            if scores[top] > base:
                plan, base = kids[top][0], scores[top]
                print("gen %d ACCEPT score=%.0f ops=%s"
                      % (gen, base, "+".join(sorted(set(kids[top][1])))), flush=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(plan, f)
            elif gen % 10 == 0:
                print("gen %d score=%.0f best-child=%.0f (%.0f min left)"
                      % (gen, base, scores[top], (deadline - time.time()) / 60),
                      flush=True)

            if gen % args.confirm_every == 0:
                hold, hvals = score(pool, plan, args.opponent, holdout, sides,
                                    args.steps, args.repair)
                best_hold = max(best_hold, hold)
                print("HOLDOUT gen=%d score=%.0f train=%.0f values=%s"
                      % (gen, hold, base, [round(v) for v in hvals]), flush=True)

            if args.checkpoint:
                with open(args.checkpoint, "w", encoding="utf-8") as f:
                    json.dump({"plan": plan, "score": base, "gen": gen,
                               "stats": stats}, f)

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(plan, f)
        hold, hvals = score(pool, plan, args.opponent, holdout, sides, args.steps, args.repair)
        print("HOLDOUT gen=%d score=%.0f values=%s"
              % (gen, hold, [round(v) for v in hvals]), flush=True)
        print("\nOPERATORS (tried/kept/mean gain when kept)", flush=True)
        for nm in sorted(stats, key=lambda k: -stats[k]["gain"]):
            s = stats[nm]
            avg = s["gain"] / s["kept"] if s["kept"] else 0.0
            print("  %-22s %5d %4d %+9.0f" % (nm, s["tried"], s["kept"], avg))
        print("\nSUMMARY=" + json.dumps({
            "gen": gen, "train": round(base, 1), "holdout": round(hold, 1),
            "best_holdout": round(best_hold, 1), "out": args.out,
            "seeds": seeds, "holdout_seeds": holdout, "repair": args.repair}),
            flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

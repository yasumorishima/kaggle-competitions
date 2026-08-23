#!/usr/bin/env python
"""Hill-climb the capital calendar instead of the action list.

Same (1+lambda) machinery as optimize.py, same paired seeds, same held-out
check -- what changes is the genome. optimize.py searches 720 steps of who
stands where; this searches roughly 150 integers of what the farm owns. The
reason for the move is measured and is written up in sim/schedule.py: a
recorded action list is welded to the season it was recorded in, and 137
generations of climbing moved the held-out win count not at all.

Every point in this search space is a season-independent statement, so there
is much less for the climb to memorise. It also cannot break the game: a
calendar the farm cannot afford is simply a calendar it does not reach, and
the policy goes on playing.

Usage:
    python sim/optimize_schedule.py --sched theirs.json --minutes 60 --out best.json
    python sim/optimize_schedule.py --sched a.json,b.json --selftest
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schedule as sched_mod                    # noqa: E402
import sched_agent                              # noqa: E402
from optimize import objective, summarise       # noqa: E402

SPECIES = sched_mod.SPECIES
DAYS = sched_mod.DAYS
MAX_HANDS = 20
# Three, not four: main.py refuses the fourth quadrant (P["max_quadrants"]),
# so a calendar asking for it would quietly behave as three and the search
# would spend its moves on a state it can never reach. Four was swept under
# the old land gate and came back 6,233 worse.
MAX_LAND = 3


# --------------------------------------------------------------------------
# scoring


def play(job):
    """One episode under a calendar. Returns (mine, theirs)."""
    from kaggle_environments import make as make_env

    sched, opponent, seed, steps, side = job
    me = sched_agent.make(sched)
    order = [me, opponent] if side == 0 else [opponent, me]
    env = make_env("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.run(order)
    final = env.steps[-1]
    return (float(final[side].reward or 0), float(final[1 - side].reward or 0))


def score(pool, sched, opponent, seeds, sides, steps, kind="wins"):
    jobs = [(sched, opponent, s, steps, side) for s in seeds for side in sides]
    mapper = pool.map if pool is not None else map
    vals = list(mapper(play, jobs))
    return summary_objective(vals, kind), vals


def summary_objective(vals, kind):
    if kind == "margin":
        return statistics.fmean([m - t for m, t in vals])
    return objective(vals, kind)


# --------------------------------------------------------------------------
# the ruler
#
# What the climb used to do: score a child on six fixed seeds from both sides,
# compare that number against the incumbent's number on the same twelve
# episodes, keep the child if it leads. Two things are wrong with it and they
# compound.
#
# The first is width. The paired spread of an edit over those twelve episodes
# is wide enough that an edit worth nothing wins a good share of the time,
# and the edits worth having are worth less than the spread. See
# sim/noise_band.py, which measures both rather than assuming them.
#
# The second is the ratchet. The incumbent's number is whatever it scored on
# those same twelve episodes on the day it was accepted -- which is to say, a
# number chosen for being high. Every later comparison is against that lucky
# reading on those same seeds, on the same seeds forever. That is how a climb
# memorises a season without ever being told to, and it is exactly the shape
# of the failure: training rising, held-out flat.
#
# So: draw fresh seeds every generation, screen candidates cheaply on one
# draw, and re-measure the survivor against the incumbent on a second draw it
# was not chosen on, with a paired t-test deciding. A candidate picked for
# being lucky on the screening episodes is lucky on those episodes only, so
# the confirmation is unbiased for the candidate that reaches it. Nothing is
# ever compared against a remembered number.


def key_of(sched):
    return hashlib.sha1(
        json.dumps(sched, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Arena:
    """Plays episodes, remembers them, never plays one twice.

    Exact, not an approximation: the environment is deterministic given a seed
    and --selftest asserts that on every run, so an episode is a pure function
    of (calendar, seed, side). It earns its keep because rotating seeds means
    re-measuring the incumbent constantly, and the cache makes every seed it
    has already met free.
    """

    def __init__(self, pool, opponent, steps):
        self.pool, self.opponent, self.steps = pool, opponent, steps
        self.seen = {}
        self.played = 0

    def values(self, sched, episodes):
        key = key_of(sched)
        want = [e for e in episodes if (key,) + e not in self.seen]
        if want:
            jobs = [(sched, self.opponent, seed, self.steps, side)
                    for seed, side in want]
            mapper = self.pool.map if self.pool is not None else map
            for episode, val in zip(want, mapper(play, jobs)):
                self.seen[(key,) + episode] = val
            self.played += len(want)
        return [self.seen[(key,) + e] for e in episodes]

    def keep_only(self, scheds):
        """Forget everything but these calendars. Children are one-generation."""
        alive = {key_of(c) for c in scheds}
        self.seen = {k: v for k, v in self.seen.items() if k[0] in alive}


def per_episode(vals, kind):
    """The objective, one episode at a time -- what a paired test needs.

    `min` has no per-episode form, so it is not offered here; --accept plain
    still has it.
    """
    if kind == "wins":
        return [1.0 if m > t else 0.0 for m, t in vals]
    if kind == "margin":
        return [m - t for m, t in vals]
    return [m for m, _t in vals]


def paired_t(child, base):
    """Mean paired difference and its t. Returns (mean, t, n)."""
    d = [c - b for c, b in zip(child, base)]
    if len(d) < 2:
        return (d[0] if d else 0.0), 0.0, len(d)
    mean = statistics.fmean(d)
    sd = statistics.stdev(d)
    if sd <= 0:
        return mean, (float("inf") if mean > 0 else 0.0), len(d)
    return mean, mean / (sd / len(d) ** 0.5), len(d)


def draw(rng, seed_pool, n_screen, n_confirm, sides):
    """Two disjoint seed sets, fresh this generation."""
    picked = rng.sample(seed_pool, n_screen + n_confirm)
    screen = [(s, side) for s in picked[:n_screen] for side in sides]
    confirm = [(s, side) for s in picked[n_screen:] for side in sides]
    return screen, confirm


def race(arena, sched, children, screen, confirm, kind, z):
    """Screen, then confirm the survivor on episodes it was not chosen on."""
    base = per_episode(arena.values(sched, screen), kind)
    best = None
    for child, applied in children:
        got = per_episode(arena.values(child, screen), kind)
        mean, _t, _n = paired_t(got, base)
        if best is None or mean > best[0]:
            best = (mean, child, applied)
    seen, child, applied = best
    got = per_episode(arena.values(child, confirm), kind)
    mean, t, _n = paired_t(got, per_episode(arena.values(sched, confirm), kind))
    return dict(child=child, applied=applied, seen=seen, mean=mean, t=t,
                accepted=(mean > 0 and t >= z))


# --------------------------------------------------------------------------
# calendar surgery -- every operator leaves a calendar the farm could follow


def _tidy(rows):
    """Clamp, then make the cumulative columns non-decreasing again.

    Mutations are allowed to be careless: repairing a calendar is cheaper than
    writing seven operators that each maintain the invariant themselves, and a
    repaired calendar is still a calendar. Only the direction is forced -- a
    farm cannot un-buy a quadrant or return a cow.
    """
    for row in rows:
        row["hands"] = max(0, min(MAX_HANDS, int(row.get("hands", 0))))
        row["land"] = max(1, min(MAX_LAND, int(row.get("land", 1))))
        for s in SPECIES:
            row[s] = max(0, int(row.get(s, 0)))
        for c in sched_mod.CROP_KEYS + sched_mod.TASK_KEYS:
            if c in row:
                row[c] = max(0, min(sched_mod.MAX_PCT, int(row[c])))
        # Unconditional, like the species columns: the non-decreasing pass
        # below indexes every row, so a row that never grew the key would take
        # the repair down with a KeyError rather than a wrong calendar.
        for s in sched_mod.STRUCTS:
            row[s] = max(0, min(sched_mod.MAX_STRUCT, int(row.get(s, 0))))
    for key in SPECIES + ("land",) + sched_mod.STRUCTS:
        run = rows[0][key]
        for row in rows:
            run = max(run, row[key])
            row[key] = run
    return rows


def _col(rows, key):
    return [row[key] for row in rows]


def _incs(rows, key):
    """A cumulative column as the purchases that built it.

    Mutating the level directly does not work: the column may not fall, so a
    repair pass has to put back any decrease, and four of the seven operators
    turned out to be partial no-ops because of it -- op_herd_when changed the
    calendar in a third of its firings and the rest vanished into the repair.
    Increments have the invariant built in. Nothing can fall, because nothing
    is ever asked to; a smaller herd is one fewer purchase, which is a thing a
    farm can actually do.
    """
    col = _col(rows, key)
    return [col[0]] + [col[d] - col[d - 1] for d in range(1, len(col))]


def _put(rows, key, incs):
    run = 0
    for row, inc in zip(rows, incs):
        run += max(0, inc)
        row[key] = run


def _nearby(day, floor, rng, span=4):
    """A different day within `span`, or None if the calendar has no room.

    Drawing a shift and clamping it looks equivalent and is not: at the edges
    the clamp lands back on the day it started from, and the mutation silently
    does nothing.
    """
    options = [d for d in range(max(floor, day - span), min(DAYS - 1, day + span) + 1)
               if d != day]
    return rng.choice(options) if options else None


def _buy_days(incs, skip_first=False):
    return [d for d, v in enumerate(incs) if v > 0 and not (skip_first and d == 0)]


def op_hands_shift(rows, rng):
    start = rng.randrange(0, DAYS)
    span = rng.randint(1, 8)
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    for row in rows[start:start + span]:
        row["hands"] += delta
    return "hands_shift"


def op_hands_scale(rows, rng):
    factor = rng.choice([0.7, 0.85, 1.2, 1.4])
    for row in rows:
        row["hands"] = int(round(row["hands"] * factor))
    return "hands_scale"


def op_herd_size(rows, rng):
    """Buy more, or fewer, head on one day."""
    species = rng.choice(SPECIES)
    incs = _incs(rows, species)
    delta = rng.choice([-2, -1, 1, 2])
    if delta < 0:
        days = _buy_days(incs)
        if not days:
            return None
        day = rng.choice(days)
    else:
        day = rng.randrange(0, DAYS)
    incs[day] = max(0, incs[day] + delta)
    _put(rows, species, incs)
    return "herd_size:" + species


def op_herd_when(rows, rng):
    """Move one purchase earlier or later.

    When the herd arrives is a different question from how big it gets, and
    the measured difference between the two farms is mostly when: theirs has
    four sheep on day 0 and eight cows by day 8, and every day an animal is
    bought earlier is a day it produces.
    """
    # Only a species the calendar actually buys can have a purchase moved --
    # introducing one is op_herd_size's job, and picking a species with an
    # empty column was two thirds of why this operator fired 121 times in 300.
    stocked = [s for s in SPECIES if _buy_days(_incs(rows, s))]
    if not stocked:
        return None
    species = rng.choice(stocked)
    incs = _incs(rows, species)
    day = rng.choice(_buy_days(incs))
    target = _nearby(day, 0, rng)
    if target is None:
        return None
    incs[target] += incs[day]
    incs[day] = 0
    _put(rows, species, incs)
    return "herd_when:" + species


def op_convert(rows, rng):
    """Spend one day's purchase on a different species."""
    a, b = rng.sample(list(SPECIES), 2)
    ia, ib = _incs(rows, a), _incs(rows, b)
    days = _buy_days(ia)
    if not days:
        return None
    day = rng.choice(days)
    moved = min(ia[day], rng.randint(1, 3))
    ia[day] -= moved
    ib[day] += moved
    _put(rows, a, ia)
    _put(rows, b, ib)
    return "convert:%s->%s" % (a, b)


def op_land_when(rows, rng):
    """Move the day a quadrant is bought. The home quadrant is not for sale."""
    incs = _incs(rows, "land")
    days = _buy_days(incs, skip_first=True)
    if not days:
        return None
    day = rng.choice(days)
    target = _nearby(day, 1, rng)
    if target is None:
        return None
    incs[target] += 1
    incs[day] -= 1
    _put(rows, "land", incs)
    return "land_when"


def op_land_count(rows, rng):
    incs = _incs(rows, "land")
    owned = sum(max(0, v) for v in incs)
    add = rng.random() < 0.5
    if add and owned >= MAX_LAND:
        add = False
    if add:
        incs[rng.randrange(1, DAYS)] += 1
    else:
        days = _buy_days(incs, skip_first=True)
        if not days:
            return None
        incs[rng.choice(days)] -= 1
    _put(rows, "land", incs)
    return "land_count"


def op_crop_dial(rows, rng):
    """Grow more, or less, of one crop over a stretch of days.

    The remaining measured gap is crops: 1,679 units sold against 913, with
    strawberry at 300 against 58 and melon 126 against 50. The dial is a
    percentage of what the policy would have planted anyway, so 100 is the
    behaviour that is already on the leaderboard and the search moves out from
    there rather than jumping to somebody else's tile counts.
    """
    crop = rng.choice(sched_mod.CROP_KEYS)
    start = rng.randrange(0, DAYS)
    span = rng.randint(3, 14)
    delta = rng.choice([-60, -40, -25, 25, 40, 60])
    for row in rows[start:start + span]:
        row[crop] = max(0, min(sched_mod.MAX_PCT, row.get(crop, 100) + delta))
    return "crop_dial:" + crop


def op_task_dial(rows, rng):
    """Push one kind of work up or down over a stretch of days.

    The capital family is swept out -- herd, land, hands and the crop dial all
    came back tie or worse against the plan near the top of the ladder -- and
    the farm still turns comparable capital into 58k where that plan turns it
    into 129k. So the difference left is not the balance sheet, it is the day:
    the policy scores every job by price and distance under one rule for all
    thirty days, and a season whose first week is clearing and sowing and whose
    last is harvesting and selling cannot be well served by one number. A
    global knob cannot say that; a per-day multiplier can.

    Spans are drawn the way the crop dial draws them, because the unit of a
    useful answer here is a phase of the season, not a single day.
    """
    task = rng.choice(sched_mod.TASK_KEYS)
    start = rng.randrange(0, DAYS)
    span = rng.randint(3, 14)
    delta = rng.choice([-60, -40, -25, 25, 40, 60])
    for row in rows[start:start + span]:
        row[task] = max(0, min(sched_mod.MAX_PCT, row.get(task, 100) + delta))
    return "task_dial:" + task


def op_struct_when(rows, rng):
    """Have the pens standing earlier, or more of them.

    The policy only offers to build a pasture once animals are already waiting
    in the shed, so the season serialises -- buy, wait, walk, build, walk back,
    carry, place -- and the measured cost is a third of the herd's working
    life: 229 animal-days against the published plan's 312. A structure target
    lets the calendar have the pen ready before the cow arrives.

    Mutated as a step change from a day onward, like the land column, because
    "nine pens standing by day five" is the shape of a useful answer. The
    repair pass then makes it non-decreasing: a farm cannot un-build a pen.
    """
    struct = rng.choice(sched_mod.STRUCTS)
    start = rng.randrange(0, DAYS)
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    for row in rows[start:]:
        row[struct] = max(0, min(sched_mod.MAX_STRUCT, row.get(struct, 0) + delta))
    return "struct_when:" + struct


OPERATORS = (op_hands_shift, op_hands_scale, op_herd_size, op_herd_when,
             op_convert, op_land_when, op_land_count, op_crop_dial,
             op_task_dial, op_struct_when)


def mutate(sched, rng, ops=2, operators=OPERATORS):
    rows = sched_mod.expand(sched)
    for row in rows:                      # every operator wants complete rows
        row.setdefault("hands", 0)
        row.setdefault("land", 1)
        for s in SPECIES:
            row.setdefault(s, 0)
        for c in sched_mod.CROP_KEYS + sched_mod.TASK_KEYS:
            row.setdefault(c, 100)
        # Zero means the calendar says nothing about structures, which leaves
        # the policy's own build gate in charge -- so an untouched calendar is
        # the behaviour that is already measured.
        for s in sched_mod.STRUCTS:
            row.setdefault(s, 0)
    applied = []
    for _ in range(ops):
        name = rng.choice(operators)(rows, rng)
        if name:
            applied.append(name)
    _tidy(rows)
    out = sched_mod.compress({str(d): rows[d] for d in range(DAYS)})
    sched_mod.validate(out)
    return out, applied


# --------------------------------------------------------------------------


def parse_pool(spec):
    """`3000-3095` or a comma list, either way a list of seeds."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    assert out, "empty seed pool"
    assert len(set(out)) == len(out), "a seed appears twice in the pool"
    return out


def load(paths):
    out = []
    for path in [p for p in paths.split(",") if p.strip()]:
        path = path.strip()
        with open(path, encoding="utf-8") as f:
            sched = json.load(f)
        sched_mod.validate(sched)
        out.append((path, sched))
    return out


def selftest(args):
    """The one property the whole climb rests on: same calendar, same money."""
    _, sched = load(args.sched)[0]
    a = play((sched, args.opponent, args.seeds_list[0], args.steps, 0))
    b = play((sched, args.opponent, args.seeds_list[0], args.steps, 0))
    print("run1=%s run2=%s" % (a, b))
    print("VERDICT=%s" % ("DETERMINISTIC" if a == b else "NOISY"))
    return 0 if a == b else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", required=True, help="one or more calendar JSONs")
    ap.add_argument("--opponent", default="main.py")
    ap.add_argument("--seeds", default="3000,3001,3002,3003",
                    help="race: only picks which starting calendar to climb")
    ap.add_argument("--holdout", default="3100,3101,3102,3103")
    ap.add_argument("--accept", default="race", choices=("race", "plain"))
    ap.add_argument("--pool", default="3000-3095",
                    help="seeds the race draws from, `a-b` or a comma list")
    ap.add_argument("--screen", type=int, default=3,
                    help="seeds per screening draw (both sides each)")
    ap.add_argument("--confirm", type=int, default=8,
                    help="seeds per confirmation draw, disjoint from the screen")
    ap.add_argument("--z", type=float, default=1.0,
                    help="t the confirmation must reach before an accept")
    ap.add_argument("--sides", default="0,1")
    ap.add_argument("--steps", type=int, default=721)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--lam", type=int, default=4)
    ap.add_argument("--ops", type=int, default=2)
    ap.add_argument("--objective", default="wins",
                    choices=("wins", "mean", "min", "margin"))
    ap.add_argument("--confirm-every", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="best_sched.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    args.seeds_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.pool_list = parse_pool(args.pool)
    holdout = [int(s) for s in args.holdout.split(",") if s.strip()]
    sides = [int(s) for s in args.sides.split(",") if s.strip()]
    if args.selftest:
        return selftest(args)

    if args.accept == "race":
        if args.objective == "min":
            ap.error("--objective min has no per-episode form; use --accept plain")
        need = args.screen + args.confirm
        if len(args.pool_list) < need:
            ap.error("--pool has %d seeds, a generation draws %d"
                     % (len(args.pool_list), need))
        clash = sorted(set(args.pool_list) & set(holdout))
        if clash:
            ap.error("held-out seeds are inside the training pool: %s" % clash)

    rng = random.Random(args.seed)
    workers = args.workers or min(8, (os.cpu_count() or 2))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    best = None
    for path, sched in load(args.sched):
        s, vals = score(pool, sched, args.opponent, args.seeds_list, sides,
                        args.steps, args.objective)
        print("CANDIDATE %-30s score=%.4f %s" % (path, s, summarise(vals)), flush=True)
        if best is None or s > best[0]:
            best = (s, sched, path)
    base, sched, path = best
    print("CHOSE " + path, flush=True)
    hold, hvals = score(pool, sched, args.opponent, holdout, sides, args.steps,
                        args.objective)
    print("HOLDOUT gen=0 score=%.4f %s" % (hold, summarise(hvals)), flush=True)

    # A `wins` objective against an opponent this farm never beats is flat: an
    # edit has to flip a whole episode to register at all, and measured against
    # the plan near the top of the ladder every one of 96 episodes was a loss
    # for the incumbent and for all eight of its children alike -- paired
    # spread exactly zero. Four hours of that is four hours of nothing, so it
    # is an error rather than a warning.
    if args.objective == "wins":
        won = sum(1 for m, t in hvals if m > t)
        if won in (0, len(hvals)):
            print("OBJECTIVE=FLAT won %d of %d -- `wins` has no gradient here;"
                  " use --objective mean (or margin)" % (won, len(hvals)),
                  flush=True)
            if pool is not None:
                pool.shutdown()
            return 2

    deadline = time.time() + args.minutes * 60
    arena = Arena(pool, args.opponent, args.steps)
    start = sched
    accepts = 0
    gen = 0
    while time.time() < deadline:
        gen += 1
        if args.accept == "plain":
            best_child = None
            for _ in range(args.lam):
                child, applied = mutate(sched, rng, args.ops)
                s, vals = score(pool, child, args.opponent, args.seeds_list,
                                sides, args.steps, args.objective)
                if best_child is None or s > best_child[0]:
                    best_child = (s, child, applied, vals)
            if best_child[0] > base:
                base, sched = best_child[0], best_child[1]
                accepts += 1
                print("GEN %d accepted %.4f via %s  %s"
                      % (gen, base, ",".join(best_child[2]),
                         summarise(best_child[3])), flush=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(sched, f, indent=1, sort_keys=True)
        else:
            screen, confirm = draw(rng, args.pool_list, args.screen,
                                   args.confirm, sides)
            children = [mutate(sched, rng, args.ops) for _ in range(args.lam)]
            got = race(arena, sched, children, screen, confirm,
                       args.objective, args.z)
            if got["accepted"]:
                sched = got["child"]
                accepts += 1
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(sched, f, indent=1, sort_keys=True)
            # Every generation prints, accepted or not. A rejection is the
            # thing the old loop could not say, and the rate of them is how
            # you tell a working test from a broken one.
            print("GEN %d %s screen=%+.1f confirm=%+.1f t=%+.2f ep=%d via %s"
                  % (gen, "ACCEPT" if got["accepted"] else "reject",
                     got["seen"], got["mean"], got["t"], arena.played,
                     ",".join(got["applied"]) or "-"), flush=True)
            arena.keep_only([sched, start])
        if gen % args.confirm_every == 0:
            hold, hvals = score(pool, sched, args.opponent, holdout, sides,
                                args.steps, args.objective)
            # Absolute, and paired against the calendar this run started from.
            # The pairing is what makes the line readable: the held-out spread
            # across seeds is far wider than the distance a climb travels, so
            # an absolute number moves mostly with which seeds are in the set.
            moved = ""
            if args.accept == "race" and sched is not start:
                held = [(s, side) for s in holdout for side in sides]
                mean, t, _n = paired_t(
                    per_episode(arena.values(sched, held), args.objective),
                    per_episode(arena.values(start, held), args.objective))
                moved = " vs-start=%+.1f t=%+.2f" % (mean, t)
            print("HOLDOUT gen=%d score=%.4f train=%.4f accepts=%d%s %s"
                  % (gen, hold, base, accepts, moved, summarise(hvals)),
                  flush=True)

    hold, hvals = score(pool, sched, args.opponent, holdout, sides, args.steps,
                        args.objective)
    print("HOLDOUT gen=%d score=%.4f %s" % (gen, hold, summarise(hvals)), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sched, f, indent=1, sort_keys=True)
    sched_mod.dump(sched, args.out)
    print("SUMMARY=" + json.dumps({
        "gen": gen, "accepts": accepts, "train": base, "holdout": hold,
        "out": args.out, "accept": args.accept, "objective": args.objective,
        "episodes": arena.played, "z": args.z,
        "screen": args.screen, "confirm": args.confirm,
        "pool": [min(args.pool_list), max(args.pool_list)],
        "seeds": args.seeds_list, "holdout_seeds": holdout}))
    if pool is not None:
        pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

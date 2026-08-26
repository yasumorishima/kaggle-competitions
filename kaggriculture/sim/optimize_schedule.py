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
    opps = ([o.strip() for o in opponent.split(",") if o.strip()]
            if isinstance(opponent, str) else list(opponent))
    jobs = [(sched, opps[s % len(opps)], s, steps, side)
            for s in seeds for side in sides]
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
        # One opponent or several. With several, the opponent is a function of
        # the seed, so a candidate and the incumbent meet exactly the same
        # (seed, side, opponent) triples and the paired test still holds --
        # robustness across opponents for free, at no extra episodes.
        #
        # It is worth having because this game is not transitive. dist_weight
        # 0.7 out-earned 1.0 against a common third party by +3,443 over 512
        # games and lost the direct contest with it by -3,037 and -3,906. A
        # calendar climbed against one opponent is a best response to that
        # opponent, and the ladder does not send that opponent.
        self.opponents = ([o.strip() for o in opponent.split(",") if o.strip()]
                          if isinstance(opponent, str) else list(opponent))
        assert self.opponents, "no opponent"
        self.opponent = self.opponents[0]
        self.pool, self.steps = pool, steps
        self.seen = {}
        self.played = 0

    def opponent_for(self, seed):
        return self.opponents[seed % len(self.opponents)]

    def values(self, sched, episodes):
        key = key_of(sched)
        want = [e for e in episodes if (key,) + e not in self.seen]
        if want:
            jobs = [(sched, self.opponent_for(seed), seed, self.steps, side)
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


def draw(rng, seed_pool, n_screen, n_confirm, sides, n_rep=0):
    """Three disjoint seed sets, fresh this generation.

    The third one only gets played when a child has already passed the other
    two, so it costs nothing on the generations that reject.
    """
    picked = rng.sample(seed_pool, n_screen + n_confirm + n_rep)
    screen = [(s, side) for s in picked[:n_screen] for side in sides]
    confirm = [(s, side) for s in picked[n_screen:n_screen + n_confirm]
               for side in sides]
    rep = [(s, side) for s in picked[n_screen + n_confirm:] for side in sides]
    return screen, confirm, rep


def lower_bound(d):
    """The mean, docked one standard error -- a pessimistic reading.

    A plain argmax over a noisy screen does not pick the child with the best
    mean, it picks the child with the widest spread: over twelve episodes a
    volatile child has the better chance of topping the list by luck alone.
    That would be a nuisance if spread and quality were unrelated here. They
    are not. Measured one operator at a time (run 32668088485), the widest
    edits are the destructive ones -- hands_scale swings 16,670 an episode to
    average -4,761, land_count swings 7,418 to average -4,615 across three of
    three children -- while what is worth having is tight: task_dial:PLANT_w
    moved +1,841 with a spread of 2,512, which is t=+3.28 on twenty episodes.
    So the plain screen was systematically handing the confirmation the worst
    candidate of the four. Docking a standard error puts them back in order,
    and costs nothing: the confirmation still decides.
    """
    mean = statistics.fmean(d)
    if len(d) < 2:
        return mean
    return mean - statistics.stdev(d) / len(d) ** 0.5


def live_children(arena, rng, sched, lam, ops, probe, tries=4):
    """`lam` mutations that actually change how the season plays.

    A third of mutations leave a calendar that reads differently and behaves
    identically. The timing operators are the worst of it: measured one
    operator at a time across nineteen children (runs 32668088485 and
    32671320007), herd_when changed the outcome once, struct_when twice,
    land_when three times -- the rest scored the incumbent's money to the
    cent on every episode. The likely reason is that the farm is money-bound
    rather than schedule-bound, so moving a purchase target a few days does
    not move the purchase.

    Left alone, such a child costs a whole generation. It cannot be spotted
    from the calendar, because the calendar really did change; and the
    pessimistic screen prefers it to anything negative, since a spread of
    zero is docked nothing. One generation of the first climb under the new
    rule went exactly that way: screen +0.0, confirm +0.0, 86 episodes spent
    proving that nothing is worth nothing.

    One episode settles it. If a child scores the incumbent's number on the
    probe it is dropped and another is drawn; the probe is the first episode
    of the screening draw, so a child that survives has already paid for it.
    """
    base = arena.values(sched, probe)
    kept, skipped = [], 0
    for _ in range(lam * tries):
        if len(kept) >= lam:
            break
        child, applied = mutate(sched, rng, ops)
        if arena.values(child, probe) == base:
            skipped += 1
            continue
        kept.append((child, applied))
    return kept, skipped


def race(arena, sched, children, screen, confirm, kind, z, cut, rep=()):
    """Screen, refuse the unmeasurable, confirm the survivor.

    The refusal is the part that was missing, and without it the whole rule
    loses money. Measured: fifty one generations under screen-then-confirm
    accepted eleven edits whose confirmations summed to +56,290, and the
    same two calendars played on ninety six fresh episodes came out
    -3,419 +/- 3,691 apart -- the climb had spent four hours going backwards.

    Replaying the stored matrix explains it without spending an episode. A
    child's spread is not a property of the search, it is a property of the
    child: `task_dial` edits came back at +263 with a standard deviation of
    1,119, while a rescheduled quadrant swung 15,000 an episode. Twelve
    screening episodes cannot tell a +250 edit from noise at that width, so
    a wide child reaching the confirmation is a coin, and a coin that lands
    well is accepted. Conditioning on `t >= z` does not fix it: the
    confirmation is unbiased for a candidate named in advance, and accepting
    on the strength of that same confirmation names it afterwards.

    Raising the bar does not fix it either -- it converges on accepting
    nothing, which is where the measurement put it: no threshold made the
    accepted set worth having, while dropping the children the screen cannot
    measure turned the true value of an accept from -129 to +209 and the
    value of a generation from -28 to +105. So the bar changed from
    significance to measurability: judge only what can be judged, and having
    dropped what cannot, the t-bar is spending accepts to buy nothing.
    """
    """Screen, then confirm the survivor on episodes it was not chosen on."""
    base = per_episode(arena.values(sched, screen), kind)
    best, wild = None, 0
    for child, applied in children:
        got = per_episode(arena.values(child, screen), kind)
        d = [c - b for c, b in zip(got, base)]
        if len(d) > 1 and statistics.stdev(d) > cut:
            wild += 1
            continue
        rank = lower_bound(d)
        if best is None or rank > best[0]:
            best = (rank, child, applied, statistics.fmean(d))
    if best is None:
        return dict(child=None, applied=[], seen=0.0, floor=0.0, mean=0.0,
                    t=0.0, accepted=False, wild=wild, rep=float("nan"))
    seen, child, applied = best[3], best[1], best[2]
    floor = best[0]
    got = per_episode(arena.values(child, confirm), kind)
    mean, t, _n = paired_t(got, per_episode(arena.values(sched, confirm), kind))
    ok = mean > 0 and t >= z
    # The confirmation is unbiased for a child named in advance, and the
    # screen does name it in advance -- but accepting only when the
    # confirmation comes out positive names it again, afterwards, and the
    # accepted set is the upper half of whatever the confirmation happened to
    # say. For a child that is truly worth zero that is a coin, so a run of
    # them is a random walk with the calendar as the walker: run 22 spent
    # fifty five generations and twenty seven accepts to arrive at +529 +/-
    # 2,696 over ninety six fresh games, which is what a random walk looks
    # like. The third draw is the same medicine `sweep.py --replicate` got on
    # 2026-08-25, and it costs episodes only on the generations that would
    # otherwise have accepted.
    rep_mean = float("nan")
    if ok and rep:
        rgot = per_episode(arena.values(child, rep), kind)
        rep_mean, _rt, _rn = paired_t(
            rgot, per_episode(arena.values(sched, rep), kind))
        ok = rep_mean > 0
    return dict(child=child, applied=applied, seen=seen, floor=floor,
                mean=mean, t=t, accepted=ok, wild=wild, rep=rep_mean)


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


def _nearby(day, floor, rng, span=DAYS):
    """A different day within `span`, or None if the calendar has no room.

    Drawing a shift and clamping it looks equivalent and is not: at the edges
    the clamp lands back on the day it started from, and the mutation silently
    does nothing.

    The span used to be four days, and four days is inside the slack.
    Measured one operator at a time over nineteen children (runs 32668088485
    and 32671320007), thirteen of them scored the incumbent's money to the
    cent on every episode: the farm is money-bound, not schedule-bound -- it
    buys the cow on the day it can afford the cow -- so a target that moves
    within the wait is not a target that moved at all. A move bites when it
    clears the wait, and the wait is the size of the season.

    So the new day is drawn from the whole season. Short moves are not
    forbidden and do not need to be: about a quarter of the draws still land
    within four days, purely because thirty days is not many, which is the
    right amount of attention to pay a distance that measured out at nothing.
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


def _task_key(rows, rng):
    """Which kind of work to push, half the time one the climb already moved.

    Thirteen kinds of work share one operator among ten, so a named dial is
    drawn under one draw in a hundred and a generation of four children with
    two operators each reaches it about once in sixteen generations. That is
    the lever the measurement liked -- PLANT_w came back +1,841 with a paired
    standard deviation of 2,512 over twenty episodes, t = +3.28 -- and the
    search was buying a ticket for it.

    Refining a dial the calendar has already been moved off 100 costs no
    prior about which dial that is: the climb's own accepted history says it,
    and a dial that was never any good never gets into the history. The other
    half of the draws stay uniform so the season can still find its first
    good dial and change its mind about an old one.
    """
    touched = sorted({k for row in rows for k in sched_mod.TASK_KEYS
                      if row.get(k, 100) != 100})
    if touched and rng.random() < 0.5:
        return rng.choice(touched)
    return rng.choice(sched_mod.TASK_KEYS)


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
    task = _task_key(rows, rng)
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

    The day is drawn early and the step reaches further up than it used to,
    because of how the target is spent. main.py builds against
    `max(0, target - already standing)`, and its reactive path already puts
    up a pen the moment an animal is waiting -- so a step placed late in the
    season asks for something that has happened anyway. Measured one
    operator at a time over forty episodes, both PASTURE children came back
    identical to the incumbent to the cent, while the COOP children moved:
    two geese need one coop, so a target of three was still above the farm,
    and fourteen cows and sheep had long since outrun a target of four.

    So `start` is the smaller of two draws -- early, without forbidding a
    late one -- and the step goes to eight, which is above what the herd
    pulls up on its own.
    """
    struct = rng.choice(sched_mod.STRUCTS)
    start = min(rng.randrange(0, DAYS), rng.randrange(0, DAYS))
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 4, 6, 8])
    for row in rows[start:]:
        row[struct] = max(0, min(sched_mod.MAX_STRUCT, row.get(struct, 0) + delta))
    return "struct_when:" + struct


OPERATORS = (op_hands_shift, op_hands_scale, op_herd_size, op_herd_when,
             op_convert, op_land_when, op_land_count, op_crop_dial,
             op_task_dial, op_struct_when)

# An even draw over ten operators is an even draw over effects that are not
# even. Measured one operator at a time, twenty episodes a child (run
# 32668088485): task_dial +560 a child and herd_size +916, against land_count
# -4,615 with all three children individually significant and hands_scale
# -4,761 on a swing of 16,670. A generation is four children; spending three
# of them on the columns that are known to cost money is why the climb reads
# so little for eighty-six episodes.
#
# The weights are the measurement, not a hunch, and they are re-measured the
# same way after any change to an operator:
#     mode=calibrate per_op=true only=<names>
# Nothing is dropped to zero. The losing columns are losing from this
# calendar, at this point on the ladder, and the race rule is what keeps a
# bad child out -- the weight only decides how much of the budget finds out.
OPERATOR_WEIGHTS = {
    "op_task_dial": 5.0,
    "op_herd_size": 3.0,
    "op_crop_dial": 2.0,
    "op_convert": 2.0,
    "op_struct_when": 1.5,
    "op_hands_shift": 1.0,
    "op_land_when": 1.0,
    "op_herd_when": 1.0,
    "op_land_count": 0.5,
    "op_hands_scale": 0.5,
}


def draw_operator(rng, operators):
    """One operator, drawn in proportion to what it has been worth."""
    weights = [OPERATOR_WEIGHTS.get(o.__name__, 1.0) for o in operators]
    total = sum(weights)
    mark = rng.random() * total
    for op, w in zip(operators, weights):
        mark -= w
        if mark < 0:
            return op
    return operators[-1]


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
        name = draw_operator(rng, operators)(rows, rng)
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
    first = args.opponent.split(",")[0].strip()
    a = play((sched, first, args.seeds_list[0], args.steps, 0))
    b = play((sched, first, args.seeds_list[0], args.steps, 0))
    print("run1=%s run2=%s" % (a, b))
    print("VERDICT=%s" % ("DETERMINISTIC" if a == b else "NOISY"))
    return 0 if a == b else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", required=True, help="one or more calendar JSONs")
    # Comma-separated for a pool. The opponent is picked by seed, so the
    # incumbent and every candidate meet the same mix.
    ap.add_argument("--opponent", default="main.py")
    ap.add_argument("--seeds", default="3000,3001,3002,3003",
                    help="race: only picks which starting calendar to climb")
    ap.add_argument("--holdout", default="3100,3101,3102,3103")
    ap.add_argument("--accept", default="race", choices=("race", "plain"))
    # Above this the screen is not measuring the child, it is measuring the
    # season. Chosen on the replayed matrix, where every cut from 2,000 to
    # 12,000 came back positive and 4,000 / 6,000 / 8,000 returned +92 /
    # +99 / +105 a generation -- a plateau, not a peak, so the number is not
    # load-bearing. Re-read it whenever the calendar changes: the incumbent's
    # own paired spread went from 9,130 to 12,457 over one climb.
    ap.add_argument("--spread-cut", type=float, default=8000.0,
                    dest="spread_cut",
                    help="race: drop a child whose screening spread exceeds"
                         " this, rather than judging what cannot be judged")
    ap.add_argument("--pool", default="3000-3095",
                    help="seeds the race draws from, `a-b` or a comma list")
    ap.add_argument("--screen", type=int, default=3,
                    help="seeds per screening draw (both sides each)")
    ap.add_argument("--confirm", type=int, default=8,
                    help="seeds per confirmation draw, disjoint from the screen")
    # Zero, because the spread cut already removed what a threshold was
    # there to catch. Measured on the stored matrix, gated: z=0.0 returns
    # +105 a generation, z=0.5 returns +68, z=1.0 returns +27 -- the bar
    # only rejects the small real gains, which is the shape the edits have.
    ap.add_argument("--replicate", type=int, default=6,
                    help="seeds for a third, disjoint check before an accept "
                         "is committed (0 = off). Only played when screen and "
                         "confirm have both already passed")
    ap.add_argument("--z", type=float, default=0.0,
                    help="t the confirmation must reach before an accept")
    ap.add_argument("--sides", default="0,1")
    ap.add_argument("--steps", type=int, default=721)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--lam", type=int, default=4)
    ap.add_argument("--ops", type=int, default=2)
    # `margin` -- our money minus theirs in the same episode -- is what wins a
    # game, and it is what the ladder scores. `mean` is our money alone, and
    # the market is shared: a calendar can lift its own money while handing the
    # opponent more. dist_weight 0.7 was adopted on exactly that column,
    # +3,443 +/- 1,136 over 512 games against a third party, and the agent
    # carrying it lost the direct contest with its predecessor by -3,037 and
    # -3,906 and rated 634.1 against 669.5.
    ap.add_argument("--objective", default="margin",
                    choices=("margin", "wins", "mean", "min"))
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
            screen, confirm, rep = draw(rng, args.pool_list, args.screen,
                                        args.confirm, sides, args.replicate)
            children, skipped = live_children(arena, rng, sched, args.lam,
                                              args.ops, screen[:1])
            if not children:
                print("GEN %d no live mutation in %d tries"
                      % (gen, args.lam * 4), flush=True)
                continue
            got = race(arena, sched, children, screen, confirm,
                       args.objective, args.z, args.spread_cut, rep)
            if got["child"] is None:
                print("GEN %d all %d children too wide to judge (cut %.0f)"
                      % (gen, got["wild"], args.spread_cut), flush=True)
                continue
            if got["accepted"]:
                sched = got["child"]
                accepts += 1
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(sched, f, indent=1, sort_keys=True)
            # Every generation prints, accepted or not. A rejection is the
            # thing the old loop could not say, and the rate of them is how
            # you tell a working test from a broken one.
            rp = got.get("rep", float("nan"))
            print("GEN %d %s screen=%+.1f(floor%+.1f) confirm=%+.1f t=%+.2f"
                  " rep=%s ep=%d dead=%d wild=%d via %s"
                  % (gen, "ACCEPT" if got["accepted"] else "reject",
                     got["seen"], got["floor"], got["mean"], got["t"],
                     "-" if rp != rp else "%+.1f" % rp,
                     arena.played, skipped, got["wild"],
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

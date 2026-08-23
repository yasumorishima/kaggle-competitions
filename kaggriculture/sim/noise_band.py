#!/usr/bin/env python
"""Measure the ruler before believing anything it measures.

The calendar climb accepts a child when its mean beats the incumbent's on six
training seeds played from both sides -- twelve episodes. Yesterday's sweeps
put the paired-comparison noise band at +/-4,000 to 5,000 money over forty
eight episodes, which is +/-10,000 over twelve, while the edits being accepted
are worth +2,000 to +5,000. If those numbers are right the climb is a random
walk that writes down its luck, and no genome will fix that: the held-out line
not following the training line is the symptom of the ruler, not of the search
space. That is a claim about the instrument, so it gets measured directly.

What this does: takes one incumbent calendar, mutates it --children ways, and
plays every one of them plus the incumbent on the same --nseeds seeds from both
sides. Same seeds for everybody, so the season cancels out of every difference
the way it does inside the climb.

Three things come out of that matrix.

* The paired standard deviation, for three different statistics -- the farm's
  own money, its margin over the opponent, and whether it won. The margin is
  in there as a control variate: a seed that is rich is rich for both farms, so
  subtracting the opponent may remove a chunk of variance that pairing on the
  seed alone does not. Whether it does is an empirical question about this
  environment and nobody has asked it.

* How many episodes it takes to resolve a real +4,000 edit, per statistic.
  That is the number that decides what an acceptance test has to cost.

* The winner's curse, measured rather than argued. Pick the best of four
  children on a random twelve episodes the way the climb does, then score that
  same child on the episodes that were not used to pick it. The gap between
  those two numbers is what the climb has been banking every generation.

The matrix is written out whole, so acceptance rules can be replayed against
real episodes offline (--replay) instead of costing a fresh run each time.

What it measured, first run, 2026-08-23 (climbed_g17 against the published
plan, eight children, 48 seeds from both sides = 96 episodes each; the matrix
is on the kaggriculture-schedules branch as schedules/band.json):

* Paired spread of an edit, per episode: 9,130 on the farm's own money, 8,050
  on the margin. Pairing on the seed removes 44% of the raw spread and the
  control variate removes another 12%. Twelve episodes therefore resolve
  nothing smaller than about +/-5,200, and the edits on offer are worth a few
  hundred: measured over 96 episodes the eight children came in between -5,774
  and +1,066.

* `wins` is not merely wide here, it is flat. Every one of the 96 episodes was
  a loss for the incumbent and for all eight children alike, so the paired
  spread is exactly zero and the objective has no gradient at all. main() now
  refuses it rather than climbing it.

* The winner's curse, on the money objective: best of four on twelve episodes
  reads +1,940 and is worth +66. A factor of thirty. Best of *eight* is worth
  -238 -- widening the search made it worse, which is the signature of picking
  on noise rather than a shortage of candidates.

* Screening and then confirming on a disjoint draw is positive at every
  setting tried, and the confirmation threshold does what it is meant to: at
  lambda=4, 12 screening episodes and 16 confirming ones, z=0 accepts 65% of
  generations worth +288 each and z=1 accepts 34% worth +646 each. That
  setting is also the most gain per episode of the grid, so it is the default.

Usage:
    python sim/noise_band.py --sched ../schedules/best.json --opponent main.py \\
        --children 8 --nseeds 48 --out band.json
    python sim/noise_band.py --replay band.json
"""
import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimize_schedule as opt                   # noqa: E402

Z95 = 1.959964


# --------------------------------------------------------------------------
# the matrix


def collect(args):
    """Play every calendar on every episode. Returns the raw matrix."""
    _, parent = opt.load(args.sched)[0]
    rng = random.Random(args.seed)
    cals = [("parent", parent, [])]
    for i in range(args.children):
        child, applied = opt.mutate(parent, rng, args.ops)
        cals.append(("child%02d" % i, child, applied))

    seeds = [args.seed0 + i for i in range(args.nseeds)]
    sides = [int(s) for s in args.sides.split(",") if s.strip()]
    episodes = [(s, side) for s in seeds for side in sides]

    jobs, index = [], []
    for name, cal, _applied in cals:
        for seed, side in episodes:
            jobs.append((cal, args.opponent, seed, args.steps, side))
            index.append(name)

    workers = args.workers or min(8, (os.cpu_count() or 2))
    print("playing %d episodes (%d calendars x %d episodes) on %d workers"
          % (len(jobs), len(cals), len(episodes), workers), flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            vals = list(pool.map(opt.play, jobs, chunksize=1))
    else:
        vals = list(map(opt.play, jobs))

    rows = {name: [] for name, _c, _a in cals}
    for name, val in zip(index, vals):
        rows[name].append([float(val[0]), float(val[1])])
    return {
        "opponent": args.opponent,
        "steps": args.steps,
        "ops": args.ops,
        "episodes": [[s, side] for s, side in episodes],
        "applied": {name: applied for name, _c, applied in cals},
        "rows": rows,
    }


# --------------------------------------------------------------------------
# statistics -- three ways to say "this child did better on this episode"


def stat_mine(val):
    return val[0]


def stat_margin(val):
    return val[0] - val[1]


def stat_win(val):
    return 1.0 if val[0] > val[1] else 0.0


STATS = (("mine", stat_mine), ("margin", stat_margin), ("win", stat_win))


def diffs(band, name, fn):
    """Per-episode paired difference between one child and the incumbent."""
    parent = band["rows"]["parent"]
    child = band["rows"][name]
    return [fn(c) - fn(p) for c, p in zip(child, parent)]


def sem(xs):
    if len(xs) < 2:
        return float("inf")
    return statistics.stdev(xs) / (len(xs) ** 0.5)


def children_of(band):
    return sorted(k for k in band["rows"] if k != "parent")


# --------------------------------------------------------------------------
# report


def report_band(band):
    kids = children_of(band)
    n = len(band["episodes"])
    print("\n== per-child effect, all %d episodes ==" % n)
    for label, fn in STATS:
        print("-- %s --" % label)
        for name in kids:
            d = diffs(band, name, fn)
            m, e = statistics.fmean(d), sem(d)
            print("   %-9s mean=%+10.1f sd=%9.1f sem=%8.1f t=%+5.2f  %s"
                  % (name, m, statistics.stdev(d), e, m / e if e else 0.0,
                     ",".join(band["applied"].get(name) or []) or "-"))

    print("\n== how wide is the ruler ==")
    parent = band["rows"]["parent"]
    for label, fn in STATS:
        raw = [fn(v) for v in parent]
        pooled = pooled_sd([diffs(band, name, fn) for name in kids])
        raw_sd = statistics.stdev(raw) if len(raw) > 1 else 0.0
        print("   %-7s incumbent spread across seeds sd=%9.1f | paired sd=%9.1f"
              " (pairing removes %4.1f%%)"
              % (label, raw_sd, pooled,
                 100.0 * (1.0 - pooled / raw_sd) if raw_sd else 0.0))

    print("\n== episodes needed to resolve an edge, 95% two-sided ==")
    for label, fn in STATS:
        pooled = pooled_sd([diffs(band, name, fn) for name in kids])
        edges = (0.02, 0.05, 0.10) if label == "win" else (2000, 4000, 8000)
        parts = []
        for edge in edges:
            need = (Z95 * pooled / edge) ** 2
            shown = ("%.2f" % edge) if label == "win" else ("%d" % edge)
            parts.append("%s -> %d ep" % (shown, int(need + 0.999)))
        print("   %-7s %s" % (label, "   ".join(parts)))


def pooled_sd(groups):
    """One spread for the whole family of edits, not one per child."""
    num = sum((len(g) - 1) * statistics.variance(g) for g in groups if len(g) > 1)
    den = sum(len(g) - 1 for g in groups if len(g) > 1)
    return (num / den) ** 0.5 if den else 0.0


# --------------------------------------------------------------------------
# acceptance rules, replayed on the measured episodes
#
# Every rule gets the same children and the same episodes; what differs is how
# many episodes it spends and what it demands before saying yes. `realised` is
# the honest number: the picked child's mean on episodes no part of the rule
# was allowed to look at.


def rule_plain(rng, band, kids, fn, lam, screen, _confirm, _z):
    """Today's rule: best of lambda on the training seeds, accept if it leads."""
    picks = rng.sample(kids, lam)
    ds = {k: diffs(band, k, fn) for k in picks}
    order = list(range(len(band["episodes"])))
    rng.shuffle(order)
    scr, rest = order[:screen], order[screen:]
    best = max(picks, key=lambda k: statistics.fmean([ds[k][i] for i in scr]))
    seen = statistics.fmean([ds[best][i] for i in scr])
    truth = statistics.fmean([ds[best][i] for i in rest])
    return dict(cost=lam * screen, accepted=seen > 0, seen=seen, truth=truth)


def rule_race(rng, band, kids, fn, lam, screen, confirm, z):
    """Screen cheap, then re-measure the survivor on episodes it never saw.

    The confirmation set is disjoint from the screening set on purpose. A
    candidate chosen for being lucky on some episodes is still lucky on those
    episodes when it is measured again there, so a rule that reuses them
    inherits exactly the bias it was written to remove.
    """
    picks = rng.sample(kids, lam)
    ds = {k: diffs(band, k, fn) for k in picks}
    order = list(range(len(band["episodes"])))
    rng.shuffle(order)
    scr = order[:screen]
    con = order[screen:screen + confirm]
    rest = order[screen + confirm:]
    best = max(picks, key=lambda k: statistics.fmean([ds[k][i] for i in scr]))
    got = [ds[best][i] for i in con]
    m, e = statistics.fmean(got), sem(got)
    return dict(cost=lam * screen + 2 * confirm, accepted=m > z * e,
                seen=m, truth=statistics.fmean([ds[best][i] for i in rest]))


def replay(band, args):
    kids = children_of(band)
    n = len(band["episodes"])
    fn = dict(STATS)[args.stat]
    lam = min(args.lam, len(kids))
    trials = args.trials
    print("\n== acceptance rules replayed on the measured episodes"
          " (stat=%s, lambda=%d, %d trials) ==" % (args.stat, lam, trials))
    print("   %-26s %5s %7s %10s %10s %10s"
          % ("rule", "cost", "accept", "seen", "true|acc", "true/gen"))
    specs = [("plain screen=%d" % args.screen, rule_plain, args.screen, 0, 0.0)]
    for confirm in args.confirms:
        for z in args.zs:
            specs.append(("race %d+%d z=%.1f" % (args.screen, confirm, z),
                          rule_race, args.screen, confirm, z))
    for label, rule, screen, confirm, z in specs:
        if screen + confirm >= n:
            print("   %-26s (needs %d episodes, have %d)"
                  % (label, screen + confirm + 1, n))
            continue
        rng = random.Random(args.seed)
        got = [rule(rng, band, kids, fn, lam, screen, confirm, z)
               for _ in range(trials)]
        acc = [g for g in got if g["accepted"]]
        rate = len(acc) / len(got)
        print("   %-26s %5d %6.1f%% %+10.1f %+10.1f %+10.1f"
              % (label, got[0]["cost"], 100 * rate,
                 statistics.fmean([g["seen"] for g in acc]) if acc else 0.0,
                 statistics.fmean([g["truth"] for g in acc]) if acc else 0.0,
                 statistics.fmean([g["truth"] if g["accepted"] else 0.0
                                   for g in got])))
    print("\n   cost = episodes per generation. true/gen = expected real gain"
          "\n   per generation, counting rejections as zero. Divide one by the"
          "\n   other to compare rules at equal compute.")


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", default="", help="incumbent calendar JSON")
    ap.add_argument("--opponent", default="main.py")
    ap.add_argument("--children", type=int, default=8)
    ap.add_argument("--ops", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--nseeds", type=int, default=48)
    ap.add_argument("--sides", default="0,1")
    ap.add_argument("--steps", type=int, default=721)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="band.json")
    ap.add_argument("--replay", default="", help="skip playing, read this matrix")
    ap.add_argument("--stat", default="mine", choices=[s for s, _ in STATS])
    ap.add_argument("--lam", type=int, default=4)
    ap.add_argument("--screen", type=int, default=12)
    ap.add_argument("--confirms", default="16,24,32")
    ap.add_argument("--zs", default="0,1,1.5")
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()
    args.confirms = [int(c) for c in args.confirms.split(",") if c.strip()]
    args.zs = [float(z) for z in args.zs.split(",") if z.strip()]

    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            band = json.load(f)
    else:
        if not args.sched:
            ap.error("--sched is required unless --replay is given")
        band = collect(args)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(band, f, separators=(",", ":"), sort_keys=True)
        print("wrote " + args.out)

    report_band(band)
    for stat in ([args.stat] if args.replay else [s for s, _ in STATS]):
        args.stat = stat
        replay(band, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

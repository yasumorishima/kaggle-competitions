#!/usr/bin/env python
"""Invariants for the capital calendar, with no environment in the loop.

The same trap as test_plan_ops.py guards: an operator that never fires, or
that produces a calendar the farm cannot follow, would burn a whole climb
without ever saying so. The list-side version of this test earned its keep --
two operators in optimize.py were silently no-ops until it was written -- and
the calendar has the extra hazard that its columns are cumulative, so a
careless edit can ask the farm to return a cow it already owns.

Run: python sim/test_sched_ops.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schedule as sched_mod            # noqa: E402
import optimize_schedule as opt         # noqa: E402

BASE = {
    "0": {"hands": 5, "COW": 1, "SHEEP": 4, "GOOSE": 0, "land": 1},
    "6": {"hands": 4, "COW": 4, "SHEEP": 4, "GOOSE": 0, "land": 2},
    "10": {"hands": 14, "COW": 8, "SHEEP": 4, "GOOSE": 0, "land": 3},
    "20": {"hands": 12, "COW": 8, "SHEEP": 4, "GOOSE": 0, "land": 3},
}

FAILED = []


def check(label, ok, detail=""):
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", label,
                           "" if ok else "  " + detail))
    if not ok:
        FAILED.append(label)


def rows_of(sched):
    rows = sched_mod.expand(sched)
    for row in rows:
        row.setdefault("hands", 0)
        row.setdefault("land", 1)
        for s in opt.SPECIES:
            row.setdefault(s, 0)
    return rows


def fake_play(seed_effect, cal_effect):
    """Stand in for the environment: no game, just a number we control.

    seed_effect(seed, side) is the season -- the part both farms share and the
    part that swamps everything real. cal_effect(sched, seed) is what the
    calendar is actually worth. Splitting them is the only way to ask whether
    an acceptance rule can tell one from the other.
    """
    def play(job):
        sched, _opponent, seed, _steps, side = job
        mine = seed_effect(seed, side) + cal_effect(sched, seed)
        return (float(mine), float(seed_effect(seed, side)))
    return play


def ruler():
    """The acceptance rule, with the environment replaced by arithmetic."""
    print("the ruler: seeds are drawn fresh and the two draws never overlap")
    rng = random.Random(4)
    pool = opt.parse_pool("3000-3095")
    overlap = wrong_size = 0
    for _ in range(500):
        screen, confirm = opt.draw(rng, pool, 3, 8, [0, 1])
        if set(s for s, _ in screen) & set(s for s, _ in confirm):
            overlap += 1
        if len(screen) != 6 or len(confirm) != 16:
            wrong_size += 1
    check("screen and confirm are disjoint", overlap == 0, "%d overlaps" % overlap)
    check("both sides of every drawn seed are played", wrong_size == 0)
    check("a pool refuses a repeated seed",
          _raises(lambda: opt.parse_pool("3000,3000")))

    print("episodes are never played twice")
    saved = opt.play
    try:
        opt.play = fake_play(lambda seed, side: seed * 10 + side,
                             lambda sched, seed: 0)
        arena = opt.Arena(None, "x", 721)
        eps = [(3000, 0), (3000, 1), (3001, 0)]
        first = arena.values(BASE, eps)
        again = arena.values(BASE, eps + [(3001, 1)])
        check("the cache returns the same episodes", first == again[:3])
        check("only the new episode was played", arena.played == 4,
              "played %d" % arena.played)
        other, _ = opt.mutate(BASE, random.Random(2), 2)
        arena.values(other, eps)
        arena.keep_only([BASE])
        check("forgetting a calendar drops its episodes",
              all(k[0] == opt.key_of(BASE) for k in arena.seen))

        print("a child that is only lucky on the screening seeds is rejected")
        # Worth +80,000 on exactly the six episodes the screen will draw, and
        # worth nothing anywhere else. The old rule would take it every time;
        # this is the whole reason the confirmation set is disjoint.
        rng = random.Random(8)
        screen, confirm = opt.draw(rng, pool, 3, 8, [0, 1])
        lucky_seeds = set(seed for seed, _ in screen)
        kids = [opt.mutate(BASE, rng, 2) for _ in range(4)]
        chosen = opt.key_of(kids[2][0])

        def only_on_screen(sched, seed):
            return 80000 if (opt.key_of(sched) == chosen
                             and seed in lucky_seeds) else 0

        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             only_on_screen)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0)
        check("the screen does pick the lucky child", got["seen"] > 0,
              "screen saw %+.1f" % got["seen"])
        check("the confirmation throws it out", not got["accepted"],
              "confirm=%+.1f t=%+.2f" % (got["mean"], got["t"]))

        print("a child that is better everywhere is accepted")
        real = opt.key_of(kids[1][0])
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 5000 * (opt.key_of(sched) == real))
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0)
        check("a real edit survives both stages", got["accepted"],
              "confirm=%+.1f t=%+.2f" % (got["mean"], got["t"]))
        check("and it is the right child", opt.key_of(got["child"]) == real)

        print("nothing at all is not an improvement")
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 0)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0)
        check("an edit worth zero is rejected", not got["accepted"],
              "confirm=%+.1f t=%+.2f" % (got["mean"], got["t"]))
    finally:
        opt.play = saved


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def main():
    sched_mod.validate(BASE)
    print("operators (each must fire, and each must change the calendar)")
    for op in opt.OPERATORS:
        rng = random.Random(11)
        fired = changed = 0
        trials = 300
        for _ in range(trials):
            rows = rows_of(BASE)
            before = [dict(r) for r in rows]
            if op(rows, rng) is None:
                continue
            fired += 1
            opt._tidy(rows)
            if rows != before:
                changed += 1
        check("%-22s fires" % op.__name__, fired >= trials * 0.5,
              "fired %d/%d" % (fired, trials))
        check("%-22s changes it" % op.__name__, changed >= fired * 0.9,
              "changed %d of %d firings" % (changed, fired))

    print("mutation keeps the calendar followable")
    rng = random.Random(3)
    bad = 0
    sched = dict(BASE)
    for _ in range(2000):
        child, _applied = opt.mutate(sched, rng, ops=3)
        try:
            sched_mod.validate(child)
        except AssertionError as exc:
            bad += 1
            print("     ", exc)
            break
        sched = child                       # walk, so late calendars get tested
    check("2000 chained mutations validate", bad == 0)

    print("cumulative columns never fall")
    rng = random.Random(5)
    falls = 0
    for _ in range(500):
        child, _ = opt.mutate(BASE, rng, ops=4)
        rows = rows_of(child)
        for key in opt.SPECIES + ("land",):
            column = [r[key] for r in rows]
            if any(b < a for a, b in zip(column, column[1:])):
                falls += 1
                break
    check("no species or quadrant is given back", falls == 0,
          "%d of 500 calendars fell" % falls)

    print("bounds hold")
    rng = random.Random(9)
    out_of_range = 0
    for _ in range(500):
        child, _ = opt.mutate(BASE, rng, ops=4)
        for row in rows_of(child):
            if not (0 <= row["hands"] <= opt.MAX_HANDS):
                out_of_range += 1
            if not (1 <= row["land"] <= opt.MAX_LAND):
                out_of_range += 1
    check("hands and land stay in range", out_of_range == 0,
          "%d violations" % out_of_range)

    print("round trip")
    rebuilt = sched_mod.compress({str(d): r for d, r in enumerate(rows_of(BASE))})
    check("expand then compress is identity",
          sched_mod.expand(rebuilt) == sched_mod.expand(BASE))

    ruler()

    print("")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import statistics
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

# A cut no fake child can exceed, so the older invariants still ask what
# they were written to ask.
WIDE = 1e12

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
        screen, confirm, _rep = opt.draw(rng, pool, 3, 8, [0, 1])
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
        screen, confirm, _rep = opt.draw(rng, pool, 3, 8, [0, 1])
        lucky_seeds = set(seed for seed, _ in screen)
        kids = [opt.mutate(BASE, rng, 2) for _ in range(4)]
        chosen = opt.key_of(kids[2][0])

        def only_on_screen(sched, seed):
            return 80000 if (opt.key_of(sched) == chosen
                             and seed in lucky_seeds) else 0

        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             only_on_screen)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE)
        check("the screen does pick the lucky child", got["seen"] > 0,
              "screen saw %+.1f" % got["seen"])
        check("the confirmation throws it out", not got["accepted"],
              "confirm=%+.1f t=%+.2f" % (got["mean"], got["t"]))

        print("a child that is better everywhere is accepted")
        real = opt.key_of(kids[1][0])
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 5000 * (opt.key_of(sched) == real))
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE)
        check("a real edit survives both stages", got["accepted"],
              "confirm=%+.1f t=%+.2f" % (got["mean"], got["t"]))
        check("and it is the right child", opt.key_of(got["child"]) == real)

        print("a child lucky on screen AND confirm is vetoed by the third draw")
        # The confirmation is unbiased for a child the screen named, but
        # accepting only when it comes out positive names the child a second
        # time, on the confirmation's own number. This is the child that
        # exploits exactly that: worth a fortune on the two sets the accept
        # rule looks at, worth nothing on the one it does not.
        rng = random.Random(11)
        screen, confirm, rep = opt.draw(rng, pool, 3, 8, [0, 1], 6)
        seen_seeds = set(s for s, _ in screen) | set(s for s, _ in confirm)
        rep_seeds = set(s for s, _ in rep)
        check("the third set is disjoint from the other two",
              not (rep_seeds & seen_seeds))
        kids = [opt.mutate(BASE, rng, 2) for _ in range(4)]
        fluke = opt.key_of(kids[0][0])
        opt.play = fake_play(
            lambda seed, side: (seed % 7) * 9000 + side * 400,
            lambda sched, seed: (80000 if opt.key_of(sched) == fluke
                                 and seed in seen_seeds else 0))
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE)
        check("without the third draw it is accepted", got["accepted"],
              "confirm=%+.1f" % got["mean"])
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE,
                       rep)
        check("with it the accept is withdrawn", not got["accepted"],
              "confirm=%+.1f rep=%+.1f" % (got["mean"], got["rep"]))
        check("and the reason is reported", got["rep"] <= 0,
              "rep=%+.1f" % got["rep"])

        print("the third draw does not veto an edit that is real")
        opt.play = fake_play(
            lambda seed, side: (seed % 7) * 9000 + side * 400,
            lambda sched, seed: 5000 * (opt.key_of(sched) == fluke))
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE,
                       rep)
        check("a real edit still passes three stages", got["accepted"],
              "confirm=%+.1f rep=%+.1f" % (got["mean"], got["rep"]))

        print("a rejected child never pays for the third draw")
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 0)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 9.0, WIDE,
                       rep)
        check("no accept, no third draw", not got["accepted"]
              and got["rep"] != got["rep"], "rep=%s" % got["rep"])

        print("the screen prefers a tight edit to a volatile one")
        # The volatile child reads three times better on the screen and is
        # worth nothing; the tight child is worth +2,000 every episode. A
        # plain argmax takes the volatile one, which is what the first climb
        # under the new rule was seen doing -- screen +7,247, confirmed
        # -8,965. Measured per operator, spread and harm go together here.
        volatile, tight = opt.key_of(kids[0][0]), opt.key_of(kids[3][0])

        # Half the screening seeds rich, half poor, so the volatile child's
        # screen mean lands above the tight child's while its spread is an
        # order of magnitude wider.
        seen_seeds = sorted({seed for seed, _side in screen})
        rich = set(seen_seeds[:len(seen_seeds) - 1])

        def two_kinds(sched, seed):
            k = opt.key_of(sched)
            if k == volatile:
                return 30000 if seed in rich else -40000
            return 2000 if k == tight else 0

        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             two_kinds)
        arena = opt.Arena(None, "x", 721)
        vals = {}
        for name, key in (("volatile", volatile), ("tight", tight)):
            cal = next(c for c, _a in kids if opt.key_of(c) == key)
            d = [m - b for (m, _t), (b, _bt) in
                 zip(arena.values(cal, screen), arena.values(BASE, screen))]
            vals[name] = (statistics.fmean(d), opt.lower_bound(d))
        check("the volatile child does read better on the mean",
              vals["volatile"][0] > vals["tight"][0],
              "volatile %+.0f vs tight %+.0f" % (vals["volatile"][0],
                                                 vals["tight"][0]))
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE)
        check("the pessimistic screen picks the tight one anyway",
              opt.key_of(got["child"]) == tight,
              "floors: volatile %+.0f tight %+.0f" % (vals["volatile"][1],
                                                      vals["tight"][1]))
        check("and it is accepted", got["accepted"],
              "confirm=%+.1f t=%+.2f" % (got["mean"], got["t"]))

        print("a child that plays the same season is dropped for one episode")
        # Two calendars in three behave identically here. The probe costs one
        # episode; leaving them in costs the generation.
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 0)
        arena = opt.Arena(None, "x", 721)
        kept, skipped = opt.live_children(arena, random.Random(5), BASE, 4, 2,
                                          screen[:1])
        check("every dead mutation is skipped", kept == [] and skipped == 16,
              "kept %d, skipped %d" % (len(kept), skipped))
        # At most one episode per attempt plus the incumbent's -- fewer when a
        # mutation happens to reproduce a calendar already in the cache.
        check("at most one probe episode each", arena.played <= 17,
              "played %d" % arena.played)

        marked = {}

        def only_marked(sched, seed):
            return 3000 if opt.key_of(sched) in marked else 0

        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             only_marked)
        arena = opt.Arena(None, "x", 721)
        rng2 = random.Random(5)
        for _ in range(64):                    # mark every calendar this seed makes
            marked[opt.key_of(opt.mutate(BASE, rng2, 2)[0])] = True
        kept, skipped = opt.live_children(arena, random.Random(5), BASE, 4, 2,
                                          screen[:1])
        check("live mutations are kept", len(kept) == 4 and skipped == 0,
              "kept %d, skipped %d" % (len(kept), skipped))

        print("a child too wide to measure is refused, not judged")
        # Same two children as above. The volatile one reads better on the
        # mean and is the one a screen picks by luck; at a cut it cannot
        # clear, it never reaches the confirmation at all.
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             two_kinds)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 0.0, 5000)
        check("the wide child is counted out", got["wild"] >= 1,
              "wild=%d" % got["wild"])
        check("and the tight one is what gets confirmed",
              got["child"] is not None and opt.key_of(got["child"]) == tight)

        # Every child swinging with the season, none of them measurable.
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 0
                             if opt.key_of(sched) == opt.key_of(BASE)
                             else (seed % 5) * 20000 - 40000)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 0.0, 1000)
        check("when every child is too wide, nothing is accepted",
              got["child"] is None and not got["accepted"],
              "child=%s" % (got["child"] is not None))
        check("and all of them are counted", got["wild"] == len(kids),
              "wild=%d of %d" % (got["wild"], len(kids)))

        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             two_kinds)
        arena = opt.Arena(None, "x", 721)

        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 0.0, WIDE)
        check("a generous cut refuses nobody", got["wild"] == 0,
              "wild=%d" % got["wild"])

        print("nothing at all is not an improvement")
        opt.play = fake_play(lambda seed, side: (seed % 7) * 9000 + side * 400,
                             lambda sched, seed: 0)
        arena = opt.Arena(None, "x", 721)
        got = opt.race(arena, BASE, kids, screen, confirm, "mean", 1.0, WIDE)
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


def steering():
    """The two columns the climb is steered by.

    Both were wrong until 2026-08-26. The objective defaulted to the farm's own
    money, and the arena played one fixed opponent -- and this game is not
    transitive, so those two choices compound. dist_weight 0.7 was adopted on
    exactly that pair: +3,443 +/- 1,136 over 512 games against a common third
    party, and the agent carrying it then lost the direct contest with its
    predecessor by -3,037 and -3,906 and rated 634.1 against 669.5.
    """
    print("the objective and the opponent pool")

    vals = [(100.0, 40.0), (50.0, 90.0)]
    check("margin is our money minus theirs",
          opt.per_episode(vals, "margin") == [60.0, -40.0])
    check("mean is our money alone",
          opt.per_episode(vals, "mean") == [100.0, 50.0])
    check("...and on this pair the two disagree in sign",
          opt.per_episode(vals, "margin")[1] < 0 < opt.per_episode(vals, "mean")[1])

    arena = opt.Arena(None, "a.py,b.py,c.py", 720)
    check("a comma list becomes a pool", arena.opponents == ["a.py", "b.py", "c.py"])
    check("the opponent is a function of the seed",
          [arena.opponent_for(s) for s in (3000, 3001, 3002, 3003)]
          == ["a.py", "b.py", "c.py", "a.py"])
    check("so the incumbent and every child meet the same mix",
          all(arena.opponent_for(s) == arena.opponent_for(s)
              for s in range(3000, 3020)))
    check("a single opponent still works",
          opt.Arena(None, "main.py", 720).opponents == ["main.py"])

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "optimize_schedule.py"), encoding="utf-8").read()
    check("the climb's default objective is margin, not own money",
          'ap.add_argument("--objective", default="margin"' in src)


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

    print("a pen is asked for while asking can still change something")
    rng = random.Random(43)
    starts, tops = [], []
    for _ in range(2000):
        rows = rows_of(BASE)
        name = opt.op_struct_when(rows, rng)
        if name is None:
            continue
        opt._tidy(rows)
        key = name.split(":")[1]
        col = [r.get(key, 0) for r in rows]
        raised = [d for d in range(sched_mod.DAYS) if col[d] > 0]
        if raised:
            starts.append(raised[0])
            tops.append(max(col))
    check("most targets start in the first half of the season",
          sum(1 for d in starts if d < sched_mod.DAYS // 2) > 0.6 * len(starts),
          "%d of %d" % (sum(1 for d in starts if d < sched_mod.DAYS // 2), len(starts)))
    check("a late start is still possible", max(starts) >= sched_mod.DAYS - 5,
          "latest %d" % max(starts))
    # Fourteen cows and sheep pull up more pens than four on their own, so a
    # target that stops at four is a target the farm has already met.
    check("the target can outrun what the herd builds by itself",
          max(tops) >= 8, "highest target %d" % max(tops))

    print("the budget goes where the measurement says")
    rng = random.Random(29)
    drawn = {}
    for _ in range(20000):
        name = opt.draw_operator(rng, opt.OPERATORS).__name__
        drawn[name] = drawn.get(name, 0) + 1
    total = float(sum(opt.OPERATOR_WEIGHTS.values()))
    worst = max(abs(drawn.get(k, 0) / 20000.0 - w / total)
                for k, w in opt.OPERATOR_WEIGHTS.items())
    check("draws match the weights", worst < 0.01, "off by %.4f" % worst)
    check("the measured lever outdraws the measured loser",
          drawn["op_task_dial"] > 8 * drawn["op_hands_scale"],
          "task_dial %d, hands_scale %d"
          % (drawn["op_task_dial"], drawn["op_hands_scale"]))
    # --only hands one operator to the same draw, and per-op calibration
    # would measure nothing at all if that came back empty.
    only = opt.draw_operator(random.Random(1), (opt.op_land_count,))
    check("a single operator is still drawable", only is opt.op_land_count)

    print("a rescheduled purchase clears the wait")
    rng = random.Random(31)
    far = near = 0
    for _ in range(4000):
        got = opt._nearby(12, 0, rng)
        if abs(got - 12) > 4:
            far += 1
        else:
            near += 1
    check("most moves reach past the old four days", far > 2 * near,
          "far %d, near %d" % (far, near))
    check("short moves are still drawn", near > 200, "near %d" % near)
    rng = random.Random(32)
    moved = []
    for _ in range(400):
        rows = rows_of(BASE)
        before = [r["land"] for r in rows]
        if opt.op_land_when(rows, rng) is None:
            continue
        opt._tidy(rows)
        after = [r["land"] for r in rows]
        if before != after:
            moved.append(max(abs(a - b) for a, b in zip(before, after)))
    check("land_when still fires and still moves the column",
          len(moved) > 200, "moved %d of 400" % len(moved))

    print("a dial the climb liked gets refined")
    rows = rows_of(BASE)
    for row in rows:
        row["PLANT_w"] = 160
        row["DIG_w"] = 40
    rng = random.Random(37)
    picks = [opt._task_key(rows, rng) for _ in range(4000)]
    hit = sum(1 for k in picks if k in ("PLANT_w", "DIG_w"))
    share = hit / 4000.0
    # Half the draws come from the two touched dials, the other half uniform
    # over thirteen: 0.5 + 0.5 * 2/13 = 0.577.
    check("touched dials take about four draws in seven",
          0.53 < share < 0.62, "share %.3f" % share)
    check("every dial is still reachable",
          len(set(picks)) == len(sched_mod.TASK_KEYS),
          "reached %d of %d" % (len(set(picks)), len(sched_mod.TASK_KEYS)))
    fresh = rows_of(BASE)
    rng = random.Random(41)
    picks = [opt._task_key(fresh, rng) for _ in range(2000)]
    top = max(picks.count(k) for k in set(picks)) / 2000.0
    check("an untouched calendar draws evenly", top < 0.12, "top %.3f" % top)

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
    steering()

    print("")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Compare several parameter settings of the agent on identical seeds.

Hand-tuning one knob per GitHub Actions round trip is slow and confounded by
season variance. This bakes each variant into its own copy of the agent, plays
every variant against the same opponent over the same seeds (both sides), and
ranks them by mean money with a 95% CI, so a difference inside the noise band
reads as inconclusive rather than as progress.

Usage:
    python sim/sweep.py --variants variants.json --b starter --episodes 6
    variants.json: [{"name": "base", "P": {}},
                    {"name": "hands16", "P": {"max_hands": 16}}]
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_keys(base_src, variants, agent):
    """Refuse to sweep a knob the agent does not have.

    `P.update()` happily adds a key nobody reads, so sweeping a knob against
    an agent emitted before that knob existed spends the whole job measuring
    nothing -- and it does not look like nothing, it looks like a clean tie.
    On 2026-08-25 six `sched_hands_scale` arms were run against v31, which
    predates the knob, and came back byte-identical to each other: same mean,
    same interval, same winrate, six times. That is the only fingerprint it
    leaves, and it is easy to read as "the knob does not matter".
    """
    have = set(re.findall(r'^\s*"([A-Za-z_]+)"\s*:', base_src, re.M))
    missing = {}
    for v in variants:
        for k in (v.get("P") or {}):
            if k not in have:
                missing.setdefault(k, []).append(v["name"])
    if missing:
        lines = ", ".join(f"{k} (in {', '.join(n)})" for k, n in missing.items())
        raise SystemExit(
            f"{agent} has no such knob: {lines}\n"
            "Emitted agents are frozen copies -- re-emit one from the current "
            "main.py before sweeping a knob that was added after it.")


def build_variant(base_src, name, overrides, outdir, globals_over=None):
    """Write a copy of the agent with `P` (and optionally module-level tables
    such as RATE) patched at the end of the file, where the agent reads them."""
    path = os.path.join(outdir, f"variant_{name}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(base_src)
        f.write("\n\n# --- sweep override ---\nP.update(")
        f.write(repr(overrides))
        f.write(")\n")
        for table, patch in (globals_over or {}).items():
            f.write(f"{table}.update({patch!r})\n")
    return path


def play(job):
    from kaggle_environments import make
    agent_a, agent_b, seed, steps, a_side = job
    order = [agent_a, agent_b] if a_side == 0 else [agent_b, agent_a]
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.run(order)
    final = env.steps[-1]
    money = [float(final[0].reward or 0), float(final[1].reward or 0)]
    return money[a_side], money[1 - a_side]


def run_all(jobs, workers):
    """Play every job, in this process when asked for one worker.

    The pool pickles `play` by reference and Windows re-imports the module in
    each child, so a stubbed `play` never reaches them -- which left the whole
    reporting path, the part that decides what gets adopted, only ever
    exercised by a sixty-minute job on a runner. One worker means one process
    and a stub that actually takes effect.
    """
    if workers <= 1:
        return [play(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(play, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="JSON file or inline JSON list")
    ap.add_argument("--agent", default="main.py")
    ap.add_argument("--b", default="starter")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=4)
    # Re-run the winner, alone against the reference, on a seed band the
    # sweep never touched. On 2026-08-25 three separate sweeps produced a
    # BETTER of +3,750, +2,731 and +1,075 and all three came back at or
    # below zero when asked again -- with a paired interval near 2,000 over
    # 96 games, the maximum of six or seven variants is worth about +3,000
    # before any of them does anything. The only knob that survived was the
    # one checked against its own neighbours and held over four bands. So
    # the sweep asks the second question itself now, for the price of two
    # more variants.
    ap.add_argument("--replicate", type=int, default=0,
                    help="episodes to re-test the winner on a fresh band (0 = off)")
    ap.add_argument("--replicate-gap", type=int, default=2000,
                    help="how far from seed0 the fresh band starts")
    args = ap.parse_args()

    spec = args.variants
    variants = json.loads(open(spec, encoding="utf-8").read()
                          if os.path.exists(spec) else spec)
    base_src = open(os.path.join(HERE, args.agent), encoding="utf-8").read()
    check_keys(base_src, variants, args.agent)
    outdir = tempfile.mkdtemp(prefix="sweep_", dir=HERE)

    jobs, owner = [], []
    for v in variants:
        path = build_variant(base_src, v["name"], v.get("P", {}), outdir, v.get("G"))
        rel = os.path.relpath(path, HERE)
        for i in range(args.episodes):
            for side in (0, 1):
                jobs.append((rel, args.b, args.seed0 + i, args.steps, side))
                owner.append((v["name"], args.seed0 + i, side))

    results = run_all(jobs, args.workers)

    # Keyed by (seed, side) as well as by variant, so a variant can be compared
    # with the reference one *within* the same season draw. Every variant plays
    # the identical seed list, and the season draw is what dominates the spread,
    # so an unpaired mean and its band cannot resolve a few thousand either way.
    per, cell, gap = {}, {}, {}
    for (name, seed, side), (ma, mb) in zip(owner, results):
        per.setdefault(name, []).append((ma, mb))
        cell[(name, seed, side)] = ma
        # The episode is not won by earning; it is won by out-earning. A
        # variant that lifts our own money while lifting the town's price
        # floor for both farms can read BETTER on `vs ref` and still lose the
        # game it was measured in, and the ladder pays for the game. Free to
        # record: both sides already come back from every episode.
        gap[(name, seed, side)] = ma - mb

    ref = variants[0]["name"]
    rows = []
    for name, pairs in per.items():
        mine = [a for a, _ in pairs]
        delta = [a - b for a, b in pairs]
        m = statistics.mean(mine)
        ci = 1.96 * statistics.stdev(delta) / math.sqrt(len(delta)) if len(delta) > 1 else float("nan")
        wins = sum(1 for a, b in pairs if a > b)
        vs = [cell[(name, s, d)] - cell[(ref, s, d)]
              for (nm, s, d) in cell if nm == name and (ref, s, d) in cell]
        gs = [gap[(name, s, d)] - gap[(ref, s, d)]
              for (nm, s, d) in gap if nm == name and (ref, s, d) in gap]
        if name == ref or len(vs) < 2:
            dm, dci, gm, gci = 0.0, float("nan"), 0.0, float("nan")
        else:
            dm = statistics.mean(vs)
            dci = 1.96 * statistics.stdev(vs) / math.sqrt(len(vs))
            # Its own interval, because the first table that carried this
            # column had all six arms negative and it was tempting to read a
            # pattern into six numbers that had never been asked how wide
            # they were.
            gm = statistics.mean(gs) if len(gs) > 1 else float("nan")
            gci = (1.96 * statistics.stdev(gs) / math.sqrt(len(gs))
                   if len(gs) > 1 else float("nan"))
        rows.append((m, ci, wins / len(pairs), name, len(pairs), dm, dci, gm, gci))
    # Rank on the margin against the reference, not on our own money. The two
    # disagree whenever a variant lifts both farms, which the shared elastic
    # market makes easy -- and it is the margin that decides the game, and the
    # ladder. The reference sits at exactly 0 by construction, so this puts
    # every variant that beats it above the line. Ties fall back to own money.
    rows.sort(key=lambda r: (0.0 if r[7] != r[7] else r[7], r[0]), reverse=True)

    head = "vs " + ref
    print(f"\n{'variant':<22}{'mean money':>12}{'+/-95%':>10}{'winrate':>9}"
          f"{'games':>7}{head:>16}{'+/-95%':>10}{'margin':>10}{'+/-95%':>9}"
          f"{'verdict':>9}")
    for m, ci, wr, name, n, dm, dci, gm, gci in rows:
        # The verdict follows the margin for the same reason the ranking does.
        # dist_weight 0.7 was adopted on own money against a third party --
        # +3,443 +/- 1,136 over 512 games across five bands, t=5.9 -- and the
        # agent carrying it then lost the direct contest with the same agent at
        # 1.0 by -3,037 and -3,906, and rated 634.1 against 669.5 on the
        # ladder. Nothing else differs between them: same calendar, and every
        # other difference between the frozen copies is inert at their
        # settings. "Out-earns a common third party" is not "beats this
        # opponent", and only the second one is a game.
        if name == ref:
            mark = "ref"
        elif gci != gci:
            mark = "?"
        elif gm - gci > 0:
            mark = "BETTER"
        elif gm + gci < 0:
            mark = "WORSE"
        else:
            mark = "tie"
        gtxt = "-" if gm != gm else f"{gm:.0f}"
        gctxt = "-" if gci != gci else f"{gci:.0f}"
        print(f"{name:<22}{m:>12.0f}{ci:>10.0f}{wr:>9.2f}"
              f"{n:>7}{dm:>16.0f}{dci:>10.0f}{gtxt:>10}{gctxt:>9}"
              f"{mark:>9}")
    print("\nSWEEP_BEST=" + json.dumps(
        {"name": rows[0][3], "mean": round(rows[0][0]),
         "margin": None if rows[0][7] != rows[0][7] else round(rows[0][7]),
         "ranked_on": "margin"}))

    winner = rows[0][3]
    if args.replicate and winner != ref:
        seed1 = args.seed0 + args.replicate_gap
        print()
        print(f"replicating {winner!r} against {ref!r} on seeds "
              f"{seed1}-{seed1 + args.replicate - 1}, both sides")
        pick = {v["name"]: v for v in variants}
        jobs2, owner2 = [], []
        for nm in (ref, winner):
            v = pick[nm]
            path = build_variant(base_src, "rep_" + nm, v.get("P", {}), outdir, v.get("G"))
            rel = os.path.relpath(path, HERE)
            for i in range(args.replicate):
                for side in (0, 1):
                    jobs2.append((rel, args.b, seed1 + i, args.steps, side))
                    owner2.append((nm, seed1 + i, side))
        res2 = run_all(jobs2, args.workers)
        cell2 = {k: ma for k, (ma, _mb) in zip(owner2, res2)}
        gap2 = {k: ma - mb for k, (ma, mb) in zip(owner2, res2)}
        vs = [cell2[(winner, sd, si)] - cell2[(ref, sd, si)]
              for (nm, sd, si) in cell2 if nm == winner and (ref, sd, si) in cell2]
        gs = [gap2[(winner, sd, si)] - gap2[(ref, sd, si)]
              for (nm, sd, si) in gap2 if nm == winner and (ref, sd, si) in gap2]
        if len(vs) > 1:
            dm = statistics.mean(vs)
            dci = 1.96 * statistics.stdev(vs) / math.sqrt(len(vs))
            # Judge on the margin, not on our own money. A game is won by
            # out-earning the other farm, and the market is shared: a variant
            # can lift its own money while handing the opponent more.
            #
            # Measured 2026-08-26. dist_weight 0.7 was adopted on own money
            # against v16_best -- +3,443 +/- 1,136 over 512 games across five
            # bands, t=5.9 -- and the agent carrying it lost the direct contest
            # with the same agent at 1.0 by -3,037 and -3,906 on two bands, and
            # rated 634.1 against 669.5 on the ladder. The two agents differ in
            # nothing else: same calendar, and every other difference between
            # the frozen copies is inert at their settings. "Out-earns a common
            # third party" and "beats this opponent" are different questions,
            # and the ladder asks the second one.
            gm = statistics.mean(gs) if len(gs) > 1 else float("nan")
            gci = (1.96 * statistics.stdev(gs) / math.sqrt(len(gs))
                   if len(gs) > 1 else float("nan"))
            held = ("HELD" if gm - gci > 0 else
                    "REVERSED" if gm + gci < 0 else "NOT CONFIRMED")
            print(f"  {winner} vs {ref}: margin {gm:+.0f} +/- {gci:.0f} "
                  f"over {len(vs)} games (own money {dm:+.0f} +/- {dci:.0f}) "
                  f"-> {held}")
            print("SWEEP_REPLICATE=" + json.dumps(
                {"name": winner, "margin": round(gm), "margin_ci95": round(gci),
                 "delta": round(dm), "ci95": round(dci),
                 "games": len(vs), "verdict": held, "judged_on": "margin"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

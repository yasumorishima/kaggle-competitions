#!/usr/bin/env python
"""Score a candidate against a panel shaped like the ladder, not against one rival.

Every offline number in this project so far came from one sparring partner at a
time, and that pairing was read as if it described the leaderboard. The 146 real
episodes our four submissions have played say it does not:

    their money   [     0, 40000)  n= 15   we win  80.0%
                  [ 40000, 55000)  n= 28           60.7%
                  [ 55000, 70000)  n= 38           71.1%
                  [ 70000, 90000)  n= 39           28.2%
                  [ 90000,      )  n= 26            0.0%

Two thirds of the grade is decided by farms unlike our own previous version, and
our own money hardly moves across the bands (59k to 72k) -- what changes is who
we are standing next to. A candidate that beats the incumbent by +3,730 in
self-play has been measured in one cell of that table, and the cell it was
measured in is not where the rating is won or lost.

So: play the candidate against a fixed panel that covers all five bands, report
the win rate per band next to the ladder's own, and weight the total by how
often the ladder actually sends each band. A band with no panel member is
reported as uncovered rather than quietly dropped.

Usage:
    python sim/panel.py --a agents/v38_sched.py --episodes 16 --seed0 40000
    python sim/panel.py --a main.py --panel starter,agents/v16_best.py --episodes 8
"""
import argparse
import json
import statistics
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import play, mean_ci  # noqa: E402

# Band edges and weights are the empirical opponent distribution of our own
# submissions, pulled with sim/ladder.py on 2026-08-27 (n=146 completed
# episodes across v18, v25, v31, v38). Re-pull and update when the sample grows.
BANDS = [(0, 40000), (40000, 55000), (55000, 70000), (70000, 90000), (90000, 10 ** 9)]
LADDER_N = [15, 28, 38, 39, 26]
LADDER_WIN = [0.800, 0.607, 0.711, 0.282, 0.000]

# opponents/ is gitignored -- a published notebook is not ours to redistribute --
# so the top band is named in the `kernel:` form the Eval workflow resolves at
# run time, and --print-panel exists so the workflow can resolve this list
# without a second copy of it living in YAML.
DEFAULT_PANEL = [
    "starter",
    "agents/v15_best.py",
    "agents/v16_best.py",
    "agents/v25_global.py",
    "kernel:boatlee/v16-rc5-high-score-8c-4s-premium-market-lead",
]


def band_of(money):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= money < hi:
            return i
    return len(BANDS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the candidate")
    ap.add_argument("--panel", default=",".join(DEFAULT_PANEL),
                    help="comma separated opponents covering the money bands")
    ap.add_argument("--episodes", type=int, default=16,
                    help="seeds per opponent; each is played from both sides")
    ap.add_argument("--seed0", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--print-panel", action="store_true",
                    help="print the panel and exit, so a caller can resolve it")
    args = ap.parse_args()

    panel = [p.strip() for p in args.panel.split(",") if p.strip()]
    if args.print_panel:
        print(",".join(panel))
        return
    jobs = []
    for oi, opp in enumerate(panel):
        # Disjoint seed block per opponent: a band must not be able to look good
        # because its seeds were kind.
        base = args.seed0 + oi * 1000
        for k in range(args.episodes):
            for side in (0, 1):
                jobs.append((args.a, opp, base + k, args.steps, side))

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(play, jobs))
    else:
        recs = [play(j) for j in jobs]

    broken = [r for r in recs if r["a_status"] not in ("ACTIVE", "DONE", "INACTIVE")
              or r["b_status"] not in ("ACTIVE", "DONE", "INACTIVE")]
    if broken:
        print(f"BROKEN n={len(broken)}  first={broken[0]['a_status']}/"
              f"{broken[0]['b_status']}  err={broken[0]['err']}")

    by_opp = {opp: [] for opp in panel}
    for job, rec in zip(jobs, recs):
        by_opp[job[1]].append(rec)

    print(f"\ncandidate: {args.a}   {args.episodes} seeds x 2 sides per opponent\n")
    print(f"{'opponent':52} {'band':>5} {'mine':>8} {'theirs':>8} "
          f"{'delta':>9} {'+/-':>7} {'win':>7}")
    per_band = {i: [0, 0] for i in range(len(BANDS))}
    rows = []
    for opp in panel:
        rs = by_opp[opp]
        mine = [r["a"] for r in rs]
        theirs = [r["b"] for r in rs]
        d, ci = mean_ci([r["delta"] for r in rs])
        wins = sum(1 for r in rs if r["delta"] > 0)
        b = band_of(statistics.mean(theirs))
        per_band[b][0] += wins
        per_band[b][1] += len(rs)
        rows.append({"opponent": opp, "band": b, "n": len(rs),
                     "mine": statistics.mean(mine), "theirs": statistics.mean(theirs),
                     "delta": d, "ci": ci, "wins": wins})
        print(f"{opp[-52:]:52} {b:>5} {statistics.mean(mine):>8.0f} "
              f"{statistics.mean(theirs):>8.0f} {d:>+9.0f} {ci:>7.0f} "
              f"{100 * wins / len(rs):>6.1f}%")

    total_w = sum(LADDER_N)
    print(f"\n{'band (their money)':22} {'ladder n':>9} {'ladder win':>11} "
          f"{'panel n':>8} {'panel win':>10}")
    weighted, covered = 0.0, 0.0
    for i, (lo, hi) in enumerate(BANDS):
        w = LADDER_N[i] / total_w
        wins, n = per_band[i]
        label = f"[{lo:>6},{hi if hi < 10 ** 9 else 0:>7})"
        if n:
            rate = wins / n
            weighted += w * rate
            covered += w
            print(f"{label:22} {LADDER_N[i]:>9} {100 * LADDER_WIN[i]:>10.1f}% "
                  f"{n:>8} {100 * rate:>9.1f}%")
        else:
            print(f"{label:22} {LADDER_N[i]:>9} {100 * LADDER_WIN[i]:>10.1f}% "
                  f"{'--':>8} {'UNCOVERED':>10}")
    # Renormalise over the covered mass and say so, rather than letting an
    # uncovered band silently count as a loss or a win.
    print(f"\nPANEL_WIN={100 * weighted / covered:.1f}%  "
          f"(over {100 * covered:.0f}% of the ladder's opponent mass)  "
          f"ladder_actual=45.9%")
    print("SUMMARY=" + json.dumps({"a": args.a, "rows": rows,
                                   "panel_win": weighted / covered,
                                   "covered": covered}, default=float))

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as fh:
            for job, rec in zip(jobs, recs):
                rec = dict(rec)
                rec["opponent"] = job[1]
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

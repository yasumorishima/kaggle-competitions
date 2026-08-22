#!/usr/bin/env python
"""Paired-seed evaluation harness for the Kaggriculture competition.

Why paired seeds: the season is highly stochastic (town shops are drawn with
replacement, weeds spawn randomly), so an unpaired A-vs-B mean is dominated by
seed variance. Every seed is therefore played twice with the sides swapped, and
the reported number is the *within-seed* money delta, which cancels the season
draw.

Usage:
    python evaluate.py --a main.py --b starter --episodes 20 --seed0 1000
    python evaluate.py --a main.py --b agents/v1.py --episodes 40 --workers 4

Agents may be a path to a .py file (with an `agent` function) or a builtin name
("random", "starter", "pass", ...). Prints a one-line SUMMARY= record plus a
per-episode table; nothing else parses stdout, so extra prints are safe.
"""
import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def play(job):
    """Run one episode. job = (agent_a, agent_b, seed, steps, a_side).

    Returns a record from A's point of view regardless of which side A played.
    Imports inside the worker so each process gets its own environment module.
    """
    from kaggle_environments import make

    agent_a, agent_b, seed, steps, a_side = job
    order = [agent_a, agent_b] if a_side == 0 else [agent_b, agent_a]
    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.run(order)
    final = env.steps[-1]
    money = [float(final[0].reward or 0), float(final[1].reward or 0)]
    ma, mb = money[a_side], money[1 - a_side]
    # A side that errored or timed out scores a clean zero, which reads exactly
    # like a farm that played badly. Carry the status and the first thing the
    # environment logged, so a broken agent announces itself instead of being
    # mistaken for a weak one.
    status = [str(getattr(final[i], "status", "")) for i in (0, 1)]
    err = ""
    for entry in (getattr(env, "logs", None) or []):
        for side in (entry if isinstance(entry, list) else [entry]):
            text = str((side or {}).get("stderr", "") or "").strip()
            if text:
                err = text.splitlines()[-1][:200]
                break
        if err:
            break
    return {"seed": seed, "a_side": a_side, "a": ma, "b": mb,
            "delta": ma - mb, "secs": round(time.time() - t0, 1),
            "a_status": status[a_side], "b_status": status[1 - a_side],
            "err": err}


def mean_ci(xs):
    """Mean and half-width of a 95% normal CI (paired deltas are near-normal)."""
    if len(xs) < 2:
        return (xs[0] if xs else 0.0), float("nan")
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs))
    return m, 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="agent A: path to .py or builtin name")
    ap.add_argument("--b", default="starter", help="agent B (default: builtin starter)")
    ap.add_argument("--episodes", type=int, default=10, help="number of seeds")
    ap.add_argument("--seed0", type=int, default=1000, help="first seed")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--no-swap", action="store_true", help="skip the mirrored game")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--jsonl", default="", help="also write per-episode records here")
    args = ap.parse_args()

    sides = [0] if args.no_swap else [0, 1]
    jobs = [(args.a, args.b, args.seed0 + i, args.steps, side)
            for i in range(args.episodes) for side in sides]

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(play, jobs))
    else:
        rows = [play(j) for j in jobs]
    wall = time.time() - t0

    rows.sort(key=lambda r: (r["seed"], r["a_side"]))
    for r in rows:
        print(f"seed {r['seed']} [A=p{r['a_side']}] A={r['a']:.0f} B={r['b']:.0f} "
              f"d={r['delta']:+.0f} ({r['secs']:.0f}s)", flush=True)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    deltas = [r["delta"] for r in rows]
    dm, dci = mean_ci(deltas)
    wins = sum(1 for r in rows if r["delta"] > 0)
    summary = {
        "a": args.a, "b": args.b, "games": len(rows),
        "a_mean": round(statistics.mean(r["a"] for r in rows), 1),
        "b_mean": round(statistics.mean(r["b"] for r in rows), 1),
        "delta_mean": round(dm, 1), "delta_ci95": round(dci, 1),
        "winrate": round(wins / len(rows), 3) if rows else 0.0,
        "wall_s": round(wall, 1),
    }
    bad = [r for r in rows if r.get("a_status") not in ("", "DONE", "ACTIVE")]
    if bad:
        summary["a_broken_games"] = len(bad)
        summary["a_status"] = bad[0]["a_status"]
        if bad[0].get("err"):
            summary["a_err"] = bad[0]["err"]
    print("\nSUMMARY=" + json.dumps(summary), flush=True)
    if bad:
        # Do not let a dead agent be reported as a weak one: an agent the
        # environment stopped scores zero, which is indistinguishable from a
        # farm that simply earned nothing.
        print(f"AGENT_BROKEN a={args.a} status={bad[0]['a_status']} "
              f"in {len(bad)}/{len(rows)} games "
              f"{bad[0].get('err', '')}".strip(), flush=True)
        print("VERDICT=A_BROKEN")
        return 0
    # A delta whose CI still spans 0 is noise, not an improvement.
    verdict = "INCONCLUSIVE" if not (abs(dm) > dci) else ("A_BETTER" if dm > 0 else "B_BETTER")
    print("VERDICT=" + verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())

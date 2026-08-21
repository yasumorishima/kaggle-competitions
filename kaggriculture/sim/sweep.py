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
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="JSON file or inline JSON list")
    ap.add_argument("--agent", default="main.py")
    ap.add_argument("--b", default="starter")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    spec = args.variants
    variants = json.loads(open(spec, encoding="utf-8").read()
                          if os.path.exists(spec) else spec)
    base_src = open(os.path.join(HERE, args.agent), encoding="utf-8").read()
    outdir = tempfile.mkdtemp(prefix="sweep_", dir=HERE)

    jobs, owner = [], []
    for v in variants:
        path = build_variant(base_src, v["name"], v.get("P", {}), outdir, v.get("G"))
        rel = os.path.relpath(path, HERE)
        for i in range(args.episodes):
            for side in (0, 1):
                jobs.append((rel, args.b, args.seed0 + i, args.steps, side))
                owner.append((v["name"], args.seed0 + i, side))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(play, jobs))

    # Keyed by (seed, side) as well as by variant, so a variant can be compared
    # with the reference one *within* the same season draw. Every variant plays
    # the identical seed list, and the season draw is what dominates the spread,
    # so an unpaired mean and its band cannot resolve a few thousand either way.
    per, cell = {}, {}
    for (name, seed, side), (ma, mb) in zip(owner, results):
        per.setdefault(name, []).append((ma, mb))
        cell[(name, seed, side)] = ma

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
        if name == ref or len(vs) < 2:
            dm, dci = 0.0, float("nan")
        else:
            dm = statistics.mean(vs)
            dci = 1.96 * statistics.stdev(vs) / math.sqrt(len(vs))
        rows.append((m, ci, wins / len(pairs), name, len(pairs), dm, dci))
    rows.sort(reverse=True)

    head = "vs " + ref
    print(f"\n{'variant':<22}{'mean money':>12}{'+/-95%':>10}{'winrate':>9}"
          f"{'games':>7}{head:>16}{'+/-95%':>10}{'verdict':>9}")
    for m, ci, wr, name, n, dm, dci in rows:
        if name == ref:
            mark = "ref"
        elif dci != dci:
            mark = "?"
        elif dm - dci > 0:
            mark = "BETTER"
        elif dm + dci < 0:
            mark = "WORSE"
        else:
            mark = "tie"
        print(f"{name:<22}{m:>12.0f}{ci:>10.0f}{wr:>9.2f}"
              f"{n:>7}{dm:>16.0f}{dci:>10.0f}{mark:>9}")
    print("\nSWEEP_BEST=" + json.dumps({"name": rows[0][3], "mean": round(rows[0][0])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

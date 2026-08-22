#!/usr/bin/env python
"""Measure the shape of a plan's labour: how far a unit walks per job it does.

The aggregate "63% of actions were steps" says a farm walks too much but not
why. What decides it is the run structure: a unit walks a stretch, then works a
stretch, and the plan is efficient exactly when the working stretches are long
-- feed a cow, care for it, collect its fertilizer, all without moving -- and
the walking stretches are short.

So this reports, per unit slot and per day, the mean length of a run of
consecutive non-move actions ("jobs per arrival") and of a run of moves ("steps
per trip"). Their ratio is the labour's efficiency, and it needs no simulation:
it is a property of the action list itself.

Accepts either a plan JSON (from record.py) or a published agent that embeds
one (decoded the same way decode_actions.py does, without importing it).

Usage:
    python sim/route_shape.py plan.json
    python sim/route_shape.py opponents/someone__their-notebook.py
"""
import ast
import base64
import json
import statistics
import sys
import zlib

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
IDLE = {"PASS"}
TURNS = 24


def load_plan(path):
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and len(node.value) > 5000:
            raw = node.value
            try:
                return json.loads(zlib.decompress(
                    base64.b85decode(raw)).decode("utf-8"))
            except Exception:
                return json.loads(raw)
    raise SystemExit(f"no embedded plan found in {path}")


def unit_streams(plan):
    """[(day, slot, [op, op, ...])] -- one op string per step of that day."""
    out = {}
    for step, action in enumerate(plan):
        day = step // TURNS
        units = [action.get("farmer") or ["PASS"]]
        units += [h or ["PASS"] for h in (action.get("hands") or [])]
        for slot, u in enumerate(units):
            name = u[0] if isinstance(u, (list, tuple)) else str(u)
            out.setdefault((day, slot), []).append(str(name))
    return out


def runs(ops, predicate):
    """Lengths of maximal runs of ops satisfying predicate."""
    lengths, current = [], 0
    for op in ops:
        if predicate(op):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    plan = load_plan(sys.argv[1])
    streams = unit_streams(plan)

    work_runs, move_runs = [], []
    per_day = {}
    for (day, _slot), ops in streams.items():
        w = runs(ops, lambda o: o not in MOVES and o not in IDLE)
        m = runs(ops, lambda o: o in MOVES)
        work_runs += w
        move_runs += m
        d = per_day.setdefault(day, {"work": [], "move": [], "idle": 0, "ops": 0})
        d["work"] += w
        d["move"] += m
        d["idle"] += sum(1 for o in ops if o in IDLE)
        d["ops"] += len(ops)

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    total = sum(len(o) for o in streams.values())
    moves = sum(1 for ops in streams.values() for o in ops if o in MOVES)
    idle = sum(1 for ops in streams.values() for o in ops if o in IDLE)
    print(f"steps {len(plan)}  unit-actions {total}")
    print(f"walking {100.0 * moves / total:.0f}%   idle {100.0 * idle / total:.0f}%")
    print(f"jobs per arrival  {mean(work_runs):.2f}  (n={len(work_runs)})")
    print(f"steps per trip    {mean(move_runs):.2f}  (n={len(move_runs)})")
    if move_runs:
        print(f"trips of 1 step   {100.0 * sum(1 for x in move_runs if x == 1) / len(move_runs):.0f}%")
        print(f"trips of 5+ steps {100.0 * sum(1 for x in move_runs if x >= 5) / len(move_runs):.0f}%")

    print("\nday  jobs/arrival  steps/trip  idle%")
    for day in sorted(per_day):
        d = per_day[day]
        print(f"{day:>3}  {mean(d['work']):>12.2f}  {mean(d['move']):>10.2f}"
              f"  {100.0 * d['idle'] / max(1, d['ops']):>5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

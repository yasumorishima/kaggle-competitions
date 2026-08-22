"""Decode the embedded action list out of a fetched public agent.

Reads the file as text and pulls the base85 literal out with `ast` -- the
module is never imported or executed. Writes a plain-JSON copy next to it so
the plan can be inspected, and prints a day-by-day summary.
"""
import ast
import base64
import collections
import json
import os
import sys
import zlib

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None

tree = ast.parse(open(SRC, encoding="utf-8").read())

blob = None
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str) \
            and len(node.value) > 5000:
        blob = node.value
        break
if blob is None:
    print("no large string literal found")
    raise SystemExit(1)

actions = json.loads(zlib.decompress(base64.b85decode(blob)).decode("utf-8"))
print("steps:", len(actions))
print("step keys:", sorted(actions[0].keys()))

if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(actions, f)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")

TURNS = 24
market_by_day = collections.defaultdict(collections.Counter)
unit_by_day = collections.defaultdict(collections.Counter)
hands_by_day = {}

for step, a in enumerate(actions):
    day = step // TURNS
    for order in (a.get("market") or []):
        if not order:
            continue
        op = order[0]
        if op in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL"):
            key = f"{op}:{order[1]}"
            qty = int(order[2]) if len(order) >= 3 else 1
        else:
            key, qty = op, 1
        market_by_day[day][key] += qty
    units = [a.get("farmer", ["PASS"])] + list(a.get("hands") or [])
    hands_by_day[day] = max(hands_by_day.get(day, 0), len(units) - 1)
    for u in units:
        if not u:
            continue
        name = u[0] if isinstance(u, (list, tuple)) else u
        if name in ("NORTH", "SOUTH", "EAST", "WEST"):
            unit_by_day[day]["MOVE"] += 1
        elif name == "PLANT":
            unit_by_day[day]["PLANT:" + str(u[1])] += 1
        else:
            unit_by_day[day][name] += 1

print("\nday  hands | market orders")
for day in sorted(market_by_day):
    m = market_by_day[day]
    top = "  ".join(f"{k}:{v}" for k, v in m.most_common(9))
    print(f"{day:>3}  {hands_by_day.get(day, 0):>5} | {top}")

print("\nday | unit actions")
for day in sorted(unit_by_day):
    u = unit_by_day[day]
    top = "  ".join(f"{k}:{v}" for k, v in u.most_common(9))
    print(f"{day:>3} | {top}")

tot = collections.Counter()
for u in unit_by_day.values():
    tot.update(u)
print("\ntotals:", "  ".join(f"{k}:{v}" for k, v in tot.most_common(20)))
moves = tot["MOVE"]
allops = sum(tot.values())
print(f"walking: {100.0 * moves / allops:.0f}% of {allops} unit actions")

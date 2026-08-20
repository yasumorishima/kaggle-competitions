#!/usr/bin/env python
"""Play one episode and print where each player's money actually came from.

A leaderboard delta says who won; it never says why. This walks the episode
step by step and reconstructs, per player: daily cash, the farm's composition,
how many units of each product left the shed (a sale, priced at the quote of
that turn), and how much was spent buying. That is the only way to see whether
an opponent is winning on volume, on price, or on something structural.

Usage:
    python sim/analyze.py --a main.py --b opponents/foo.py --seed 2000
"""
import argparse
import json
from collections import defaultdict


def obs_of(state, i):
    return state[i].observation


def get(o, k, d=None):
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()

    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": args.steps, "seed": args.seed})
    env.run([args.a, args.b])

    sold = [defaultdict(int), defaultdict(int)]
    revenue = [defaultdict(float), defaultdict(float)]
    ops = [defaultdict(int), defaultdict(int)]      # what the hands actually did
    market_ops = [defaultdict(int), defaultdict(int)]
    prev_shed = [None, None]
    prev_carried = [{}, {}]
    daily = [[], []]

    for step_idx, state in enumerate(env.steps):
        for p in (0, 1):
            act = getattr(state[p], "action", None)
            if isinstance(act, dict):
                farmer = act.get("farmer") or []
                if farmer:
                    ops[p][str(farmer[0])] += 1
                for h in (act.get("hands") or []):
                    if h:
                        ops[p][str(h[0])] += 1
                for m in (act.get("market") or []):
                    if m:
                        market_ops[p][str(m[0])] += 1
            o = obs_of(state, p)
            farms = get(o, "farms")
            if not farms:
                continue
            priv = get(o, "private", {}) or {}
            shed = dict(get(priv, "shed", {}) or {})
            prices = dict(get(get(o, "market", {}), "prices", {}) or {})
            carried = defaultdict(int)
            for iv in (get(priv, "inventories", []) or []):
                if isinstance(iv, dict):
                    for item, k in iv.items():
                        if isinstance(k, int):
                            carried[item] += k
            if prev_shed[p] is not None:
                for item, before in prev_shed[p].items():
                    drop = before - shed.get(item, 0)
                    # A shed drop is only a sale if the goods did not simply
                    # move into a farmhand's arms: PICKUP (feed, fertilizer, a
                    # goose to place) empties the shed without earning a coin.
                    picked = carried.get(item, 0) - prev_carried[p].get(item, 0)
                    drop -= max(0, picked)
                    if drop > 0 and item in prices:
                        sold[p][item] += drop
                        revenue[p][item] += drop * prices.get(item, 0)
            prev_shed[p] = shed
            prev_carried[p] = dict(carried)

            hour = int(get(o, "hour", 0))
            if hour == 0:
                farm = farms[p]
                tiles = get(farm, "tiles", []) or []
                comp = defaultdict(int)
                for row in tiles:
                    for t in row:
                        if isinstance(t, dict):
                            if t.get("kind") == "PLANT":
                                comp[t["crop"]] += 1
                            elif "animal" in t:
                                comp[t["animal"]] += 1
                            else:
                                comp[t.get("kind", "?")] += 1
                daily[p].append({
                    "day": int(get(o, "day", 0)),
                    "money": int(get(farm, "money", 0)),
                    "quads": len(get(farm, "unlocked_quadrants", []) or []),
                    "comp": dict(comp),
                })

    final = env.steps[-1]
    print(f"seed {args.seed}: A={args.a}  B={args.b}")
    for p, name in ((0, args.a), (1, args.b)):
        print(f"\n=== player {p} ({name})  final money {final[p].reward:.0f}")
        print("  day  money  quads  farm")
        for row in daily[p]:
            comp = " ".join(f"{k}:{v}" for k, v in sorted(row["comp"].items()))
            print(f"  {row['day']:>3}  {row['money']:>6}  {row['quads']:>4}   {comp}")
        print("  sales (units @ mean price = revenue):")
        for item in sorted(sold[p], key=lambda i: -revenue[p][i]):
            u = sold[p][item]
            r = revenue[p][item]
            print(f"    {item:<11} {u:>5} @ {r / max(1, u):>7.1f} = {r:>10.0f}")
        print(f"    TOTAL revenue ~ {sum(revenue[p].values()):.0f}")
        total_ops = sum(ops[p].values()) or 1
        moves = sum(v for k, v in ops[p].items() if k in ("NORTH", "SOUTH", "EAST", "WEST"))
        print(f"  unit actions ({total_ops} total, {100.0 * moves / total_ops:.0f}% walking):")
        print("    " + "  ".join(f"{k}:{v}" for k, v in
                                 sorted(ops[p].items(), key=lambda kv: -kv[1])))
        print("  market orders:")
        print("    " + "  ".join(f"{k}:{v}" for k, v in
                                 sorted(market_ops[p].items(), key=lambda kv: -kv[1])))

    last = obs_of(env.steps[-1], 0)
    print("\nfinal market prices:", json.dumps(dict(get(get(last, "market", {}), "prices", {}) or {})))
    print("final market inventory:", json.dumps(dict(get(get(last, "market", {}), "inventory", {}) or {})))
    print("town shops:", json.dumps(list(get(get(last, "town", {}), "unlocked_shops", []) or [])))


if __name__ == "__main__":
    main()

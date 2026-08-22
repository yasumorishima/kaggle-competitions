#!/usr/bin/env python
"""Reconstruct where a plan's work actually happens on the board.

An action list carries no coordinates, but it does not need to: a unit's
position is fully determined by where it spawned and the moves it made.
Reading the environment settles both. Moves always apply unless they would
leave the board, and are applied *before* the market is processed, so a hand
hired on a step exists from the next one. Hands spawn on the four shed-access
tiles in NWSE order, least occupied first, and the roster is cleared nightly;
the farmer keeps its position across days.

So the board a plan works on can be recovered exactly, and that matters here:
the top public plan does 2.07 jobs every time it stops and walks two steps
between stops, which is a farm laid out in a block. This prints which tiles it
does that work on.

Usage:
    python sim/layout.py opponents/someone__their-notebook.py
    python sim/layout.py plan.json
"""
import collections
import sys

from route_shape import load_plan  # noqa: E402  (same directory)

TURNS = 24
BOARD = 10
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
IDLE = {"PASS"}


def shed_access(board=BOARD):
    half = board // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def spawn(farmer, hands, board=BOARD):
    tiles = shed_access(board)
    occupancy = {t: 0 for t in tiles}
    for pos in [farmer] + list(hands):
        if pos in occupancy:
            occupancy[pos] += 1
    return min(tiles, key=lambda t: (occupancy[t], tiles.index(t)))


def walk(plan, board=BOARD):
    """Yield (step, unit_index, position, op) for every unit action."""
    farmer = (board // 2 - 1, board // 2 - 1)
    hands = []
    for step, action in enumerate(plan):
        units = [action.get("farmer") or ["PASS"]]
        units += [h or ["PASS"] for h in (action.get("hands") or [])]
        positions = [farmer] + hands
        for idx, act in enumerate(units):
            if idx >= len(positions):
                break
            op = act[0] if isinstance(act, (list, tuple)) else str(act)
            yield step, idx, positions[idx], op, act
            if op in MOVES:
                dx, dy = MOVES[op]
                nx, ny = positions[idx][0] + dx, positions[idx][1] + dy
                if 0 <= nx < board and 0 <= ny < board:
                    positions[idx] = (nx, ny)
        farmer, hands = positions[0], positions[1:]
        # The roster is cleared at nightfall; otherwise it grows to whatever the
        # next step's plan says, each new hand spawning at the shed.
        if (step + 1) % TURNS == 0:
            hands = []
        elif step + 1 < len(plan):
            want = len(plan[step + 1].get("hands") or [])
            while len(hands) < want:
                hands.append(spawn(farmer, hands, board))
            hands = hands[:want]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    plan = load_plan(sys.argv[1])

    work = collections.Counter()
    kinds = collections.defaultdict(collections.Counter)
    for _step, _idx, pos, op, _act in walk(plan):
        if op in MOVES or op in IDLE:
            continue
        work[pos] += 1
        kinds[pos][op] += 1

    total = sum(work.values())
    print(f"work actions placed: {total} over {len(work)} tiles")
    half = BOARD // 2
    quad = collections.Counter()
    for (x, y), n in work.items():
        name = ("N" if y < half else "S") + ("W" if x < half else "E")
        quad[name] += n
    print("by quadrant:", "  ".join(f"{k}:{v}" for k, v in quad.most_common()))

    print("\nwork per tile (blank = never worked, shed access marked #):")
    access = set(shed_access())
    print("     " + "".join(f"{x:>4}" for x in range(BOARD)))
    for y in range(BOARD):
        row = []
        for x in range(BOARD):
            n = work.get((x, y), 0)
            cell = str(n) if n else ("#" if (x, y) in access else ".")
            row.append(f"{cell:>4}")
        print(f"  {y:>2} " + "".join(row))

    print("\nbusiest tiles:")
    for pos, n in work.most_common(12):
        top = "  ".join(f"{k}:{v}" for k, v in kinds[pos].most_common(4))
        print(f"  {pos}  {n:>4}   {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

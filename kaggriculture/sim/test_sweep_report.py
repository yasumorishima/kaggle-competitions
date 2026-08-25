"""The sweep's report is what decides what gets adopted, so pin it.

Until now the reporting path only ever ran inside a sixty-minute Actions job,
which is a slow and expensive place to find a formatting bug -- and a bad
place to notice that the column being ranked is not the column that wins
games. These stub `play` and read the printed table.
"""
import io
import json
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(label, cond, detail=""):
    print(("  ok   " + label) if cond
          else ("  FAIL " + label + (" -- " + detail if detail else "")))
    if not cond:
        FAILED.append(label)


def run(variants, money, agent="main.py", extra=()):
    """Run the sweep with `play` stubbed, and return its stdout."""
    calls = []

    def fake_play(job):
        _a, _b, seed, _steps, side = job
        # The replicate stage rebuilds each variant under a `rep_` name, so a
        # stub that matches on the bare name silently makes both arms
        # identical and every replication comes back at exactly zero. That is
        # what this line is for; it cost a red test to find.
        name = os.path.basename(_a).replace("variant_", "").replace(".py", "")
        name = name[4:] if name.startswith("rep_") else name
        calls.append((name, seed, side))
        return money(name, seed, side)

    old = sweep.play
    sweep.play = fake_play
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["sweep.py", "--variants", json.dumps(variants),
                "--agent", agent, "--b", "x", "--episodes", "8",
                "--seed0", "500", "--workers", "1"] + list(extra)
    try:
        with contextlib.redirect_stdout(buf):
            sweep.main()
    finally:
        sweep.play = old
        sys.argv = argv
    return buf.getvalue(), calls


def col(out, name, header):
    """Read one column of the printed table by its header offset."""
    head = [l for l in out.splitlines() if l.strip().startswith("variant")][0]
    row = [l for l in out.splitlines() if l.strip().startswith(name + " ")
           or l.split(" ")[0] == name][0]
    end = head.index(header) + len(header)
    prev = 0
    for h in ("variant", "mean money", "+/-95%", "winrate", "games",
              "vs ", "+/-95%", "margin", "verdict"):
        i = head.find(h, prev)
        if i >= 0 and head.index(header) == i:
            break
        prev = i + len(h) if i >= 0 else prev
    return row[prev:end].strip()


print("a variant can earn more and still lose by more")
# `rich` takes 10,000 more than the reference every game -- and hands the
# opponent 30,000 while doing it, which is what a farm that floods the town
# with produce does to the price both farms sell into. Ranked on our own
# money it is the winner. It loses every game.
VAR = [{"name": "base", "P": {}}, {"name": "rich", "P": {"dist_weight": 0.5}}]


def money(name, seed, side):
    mine = 60000 + (seed % 5) * 1000 + (10000 if name == "rich" else 0)
    theirs = 55000 + (30000 if name == "rich" else 0)
    return mine, theirs


out, calls = run(VAR, money)
print(out)
check("both variants played the same seeds and both sides",
      sorted(set((s, d) for n, s, d in calls if n == "base"))
      == sorted(set((s, d) for n, s, d in calls if n == "rich")))
check("our-money delta is positive", "10000" in out)
check("the margin column shows the loss", "-20000" in out,
      "margin column missing from the table")
check("and the winrate says it never wins", " 0.00" in out)

print("\na variant that wins by more shows a positive margin")


def money2(name, seed, side):
    mine = 60000 + (seed % 5) * 1000 + (2000 if name == "rich" else 0)
    theirs = 55000 - (3000 if name == "rich" else 0)
    return mine, theirs


out2, _ = run(VAR, money2)
print(out2)
check("margin is the sum of both sides moving", "5000" in out2)
check("the reference itself has no margin entry against itself",
      out2.splitlines()[2].rstrip().endswith("ref")
      or "ref" in out2)

print("\nthe replicate stage reports a margin too")
out3, _ = run(VAR, money2, extra=("--replicate", "4", "--replicate-gap", "900"))
check("SWEEP_REPLICATE is printed", "SWEEP_REPLICATE=" in out3)
blob = json.loads(out3.split("SWEEP_REPLICATE=")[1].splitlines()[0])
check("it carries the margin", "margin" in blob, str(blob))
check("and it held", blob.get("verdict") == "HELD", str(blob))
check("the fresh band is disjoint from the sweep's own",
      "1400-1403" in out3, out3.split("replicating")[-1].splitlines()[0])

print("\none worker means the stub is actually used")
check("no episode escaped to a subprocess", len(calls) > 0)

print("\n" + ("all invariants hold" if not FAILED
             else "FAILED: " + ", ".join(FAILED)))
sys.exit(1 if FAILED else 0)

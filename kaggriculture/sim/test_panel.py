"""The panel's report is the new basis for adoption, so pin it.

Three things have to hold or the panel is worse than the single-opponent Eval
it replaces:

* an opponent lands in the band its *measured* money puts it in, not the band
  we assumed when we picked it;
* a band with no panel member is announced as UNCOVERED and the weighted total
  is renormalised over the covered mass -- an uncovered band must never read
  as a clean sweep or as a wipeout;
* each opponent gets its own seed block, so a band cannot look strong because
  its seeds were kind.

These stub `play`, so no episode is run.
"""
import io
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel  # noqa: E402

FAILED = []


def check(label, cond, detail=""):
    print(("  ok   " + label) if cond
          else ("  FAIL " + label + (" -- " + detail if detail else "")))
    if not cond:
        FAILED.append(label)


def run(money, argv):
    """Run the panel with `play` stubbed. `money` maps opponent -> (mine, theirs)."""
    seen = []

    def fake_play(job):
        agent_a, agent_b, seed, steps, side = job
        seen.append((agent_b, seed))
        mine, theirs = money[agent_b]
        return {"seed": seed, "a_side": side, "a": float(mine), "b": float(theirs),
                "delta": float(mine - theirs), "secs": 0.0,
                "a_status": "DONE", "b_status": "DONE", "err": ""}

    real_play, panel.play = panel.play, fake_play
    real_argv, sys.argv = sys.argv, ["panel.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            panel.main()
    finally:
        panel.play = real_play
        sys.argv = real_argv
    return buf.getvalue(), seen


def main():
    print("band_of")
    check("39,999 is band 0", panel.band_of(39999) == 0)
    check("40,000 is band 1", panel.band_of(40000) == 1)
    check("89,999 is band 3", panel.band_of(89999) == 3)
    check("90,000 is band 4", panel.band_of(90000) == 4)
    check("a huge farm stays in band 4", panel.band_of(10 ** 7) == 4)

    print("\nan opponent is placed by its measured money, not by our label")
    # "weak" is named as if it were feeble but earns 95k: it must be graded in
    # the top band, where we lose, or the panel would flatter the candidate.
    out, _ = run({"weak": (60000, 95000), "strong": (60000, 30000)},
                 ["--a", "cand", "--panel", "weak,strong",
                  "--episodes", "2", "--workers", "1"])
    top = [ln for ln in out.splitlines() if ln.startswith("weak")]
    check("the 95k opponent is graded in band 4", top and top[0].split()[1] == "4",
          top[0] if top else "row missing")
    check("we lose to it", "  0.0%" in top[0])

    print("\nan uncovered band is announced, and the total is renormalised")
    check("UNCOVERED appears for the three empty bands",
          out.count("UNCOVERED") == 3, f"count={out.count('UNCOVERED')}")
    # Covered mass is band 0 (15/146) and band 4 (26/146) = 41/146. We win all
    # of band 0 and none of band 4, so the weighted rate is 15/41.
    want = 100 * 15 / 41
    line = [ln for ln in out.splitlines() if ln.startswith("PANEL_WIN=")]
    got = float(line[0].split("=")[1].split("%")[0]) if line else -1.0
    check("PANEL_WIN is renormalised over the covered mass",
          abs(got - want) < 0.05, f"got {got}, want {want:.1f}")
    check("the covered mass is stated",
          "28% of the ladder's opponent mass" in out, line[0] if line else "")

    print("\neach opponent gets its own seed block")
    _, seen = run({"a": (50000, 40000), "b": (50000, 40000)},
                  ["--a", "cand", "--panel", "a,b", "--episodes", "3",
                   "--seed0", "40000", "--workers", "1"])
    seeds_a = sorted({s for o, s in seen if o == "a"})
    seeds_b = sorted({s for o, s in seen if o == "b"})
    check("first opponent starts at seed0", seeds_a == [40000, 40001, 40002], str(seeds_a))
    check("second opponent is 1000 away", seeds_b == [41000, 41001, 41002], str(seeds_b))
    check("both sides are played", len(seen) == 12, str(len(seen)))

    print("\n" + ("FAILED: " + ", ".join(FAILED) if FAILED else "all panel checks pass"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

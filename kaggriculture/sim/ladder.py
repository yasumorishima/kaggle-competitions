#!/usr/bin/env python
"""What the leaderboard is actually made of: the episodes our submission played.

Every measurement in this project up to now was taken against one opponent --
the strongest published replay -- and read as if it described the leaderboard.
It does not. Against that replay this farm makes about 52k while the replay
makes 126k, a 75k hole that no hill climb was ever going to close, and that
number is what made a +589 improvement look pointless.

The ladder is a different world, and Kaggle will tell us so. The episode
service returns, per episode, both agents' final money, both teams' ratings
before and after, and who the opponent was. Pulled for our own submissions it
gives the two things offline sparring cannot:

* the opponent distribution we are actually graded against (median money near
  60k, not 126k), and
* the exchange rate between money and rating, because every episode is a
  paired head-to-head and we can ask what would have flipped.

That second number is the point of this file. Re-scoring the real episodes with
our money shifted by a constant says how many wins a given improvement buys.
On the sample as of 2026-08-25 it is steep -- +2,500 turns 45% into 52% -- so
small, real gains are worth chasing after all, provided they are real.

API notes, found by probing rather than from documentation:

* `ListEpisodes` needs an *ID filter*. `teamId` is rejected with "You must
  specify at least one ID filter"; `submissionId` works; `ids` is denied
  outright for a competition we do not own.
* Submission IDs are not in the CLI output. `GET /api/v1/competitions/
  submissions/list/<slug>` returns them as `ref`.
* `reward` is the agent's final money, and it is None for an episode that is
  still running or that errored, so both sides are checked before use.

Usage:
    python sim/ladder.py                       # our submissions, full report
    python sim/ladder.py --json out.json       # ...and keep the raw episodes
"""
import argparse
import base64
import json
import os
import statistics as st
import urllib.request

SLUG = "kaggriculture"
EPISODE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
SUBMISSION_URL = f"https://www.kaggle.com/api/v1/competitions/submissions/list/{SLUG}"

# Deltas the report prices in win rate. The low end matters most: a hill climb
# generation is worth a few hundred, and the question is whether that is noise
# or a rung.
DELTAS = (0, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 50000)
SCALES = (1.0, 1.05, 1.10, 1.20, 1.35, 1.50, 2.00)


def _auth():
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), encoding="utf-8") as f:
        k = json.load(f)
    return "Basic " + base64.b64encode(f"{k['username']}:{k['key']}".encode()).decode()


def _get(url, auth):
    req = urllib.request.Request(url, headers={"Authorization": auth, "User-Agent": "kg"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read().decode())


def _post(url, body, auth):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": auth,
                 "User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read().decode())


def submissions(auth):
    """Our submissions, newest first, with the IDs the episode service wants."""
    return [{"id": s["ref"], "date": s["date"], "score": s.get("publicScore"),
             "desc": (s.get("description") or "")[:60]}
            for s in _get(SUBMISSION_URL, auth)]


def episodes_for(submission_id, auth):
    """Rows of (mine, theirs, their_rating, my_rating_before, my_rating_after).

    Self-play episodes -- the seeding game a new submission plays against its
    own team -- carry one agent only and are dropped: they are a tie by
    construction and would flatter every statistic here.
    """
    data = _post(EPISODE_URL, {"submissionId": submission_id}, auth)
    rows = []
    for e in data.get("episodes", []):
        ours = [a for a in e["agents"] if a.get("submissionId") == submission_id]
        if not ours:
            continue
        me = ours[0]
        others = [a for a in e["agents"] if a is not me]
        if not others:
            continue
        them = others[0]
        if me.get("reward") is None or them.get("reward") is None:
            continue
        rows.append({
            "time": e["createTime"],
            "mine": float(me["reward"]),
            "theirs": float(them["reward"]),
            "their_rating": them.get("initialScore"),
            "before": me.get("initialScore"),
            "after": me.get("updatedScore"),
            "their_team": them.get("teamId"),
        })
    rows.sort(key=lambda r: r["time"])
    return rows


def describe(rows, label):
    n = len(rows)
    if not n:
        print(f"{label}: no completed episodes")
        return
    mine = [r["mine"] for r in rows]
    theirs = [r["theirs"] for r in rows]
    diff = [a - b for a, b in zip(mine, theirs)]
    wins = sum(1 for d in diff if d > 0)
    sem = st.stdev(diff) / n ** 0.5 if n > 1 else float("nan")
    print(f"\n{label}  n={n}  W-L = {wins}-{n - wins}  ({100 * wins / n:.1f}%)")
    print(f"  my money    mean {st.mean(mine):8.0f}  median {st.median(mine):8.0f}"
          f"  sd {st.stdev(mine) if n > 1 else 0:7.0f}"
          f"  min {min(mine):7.0f}  max {max(mine):7.0f}")
    print(f"  their money mean {st.mean(theirs):8.0f}  median {st.median(theirs):8.0f}"
          f"  sd {st.stdev(theirs) if n > 1 else 0:7.0f}"
          f"  min {min(theirs):7.0f}  max {max(theirs):7.0f}")
    print(f"  paired diff mean {st.mean(diff):+8.0f}  sem {sem:7.0f}"
          f"  t {st.mean(diff) / sem if sem == sem and sem else float('nan'):+.2f}")


def exchange_rate(rows):
    """How many of these very episodes flip if our money moves by a constant.

    The counterfactual holds the opponent's money fixed, which is only honest
    while our changes do not move theirs. The market is shared, so that is an
    assumption, not a fact -- but the measured correlation between the two
    rewards is weak (-0.17 over the 2026-08-25 sample), so a constant shift is
    close enough to price a lever with.
    """
    n = len(rows)
    base = sum(1 for r in rows if r["mine"] > r["theirs"])
    print(f"\nwhat a money gain buys, re-scored on these {n} real episodes")
    print(f"  {'delta':>8}  {'win rate':>9}  {'flips':>6}")
    for d in DELTAS:
        w = sum(1 for r in rows if r["mine"] + d > r["theirs"])
        print(f"  {d:>+8}  {100 * w / n:>8.1f}%  {w - base:>+6}")
    print(f"  {'scale':>8}  {'win rate':>9}")
    for f in SCALES:
        w = sum(1 for r in rows if r["mine"] * f > r["theirs"])
        print(f"  {'x%.2f' % f:>8}  {100 * w / n:>8.1f}%")


def by_opponent_band(rows):
    """Where the losses live. Splitting by the opponent's money separates
    "they are a replay and we were never in it" from "we lost a close one",
    and only the second kind is worth spending a sweep on."""
    print("\nwin rate by the opponent's money")
    bands = [(0, 40000), (40000, 55000), (55000, 70000), (70000, 90000), (90000, 10 ** 9)]
    for lo, hi in bands:
        g = [r for r in rows if lo <= r["theirs"] < hi]
        if not g:
            continue
        w = sum(1 for r in g if r["mine"] > r["theirs"])
        print(f"  their money [{lo:>6},{hi if hi < 10 ** 9 else 0:>7}) "
              f" n={len(g):>3}  my mean {st.mean([r['mine'] for r in g]):>7.0f}"
              f"  win {w:>3}/{len(g):<3} ({100 * w / len(g):>5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the raw rows here")
    ap.add_argument("--limit", type=int, default=4, help="most recent N submissions")
    args = ap.parse_args()

    auth = _auth()
    subs = submissions(auth)[: args.limit]
    dump = {}
    pooled = []
    for s in subs:
        rows = episodes_for(s["id"], auth)
        dump[str(s["id"])] = rows
        describe(rows, f"submission {s['id']}  LB {s['score']}  {s['desc']}")
        pooled.extend(rows)

    if pooled:
        describe(pooled, "POOLED")
        by_opponent_band(pooled)
        exchange_rate(pooled)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dump, f)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

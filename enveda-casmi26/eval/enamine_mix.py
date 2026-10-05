"""Fast comparison of ways to mix Enamine-tier candidates into the pool ranking.

Reads the cache written by enamine_slot.py (per molecule: pool ranking + scores, Enamine-only
ranking + analog scores, gated flag) and reports MRR@25 per class for
  - fixed slots (1-based ranks), only when no library match reaches the gate
  - score merge: Enamine candidates compete with the pool on F * analog score (pool score =
    analog + fp2 term after demotion; gated pool hits stay on top)

    python enamine_mix.py
"""
import pickle

import pandas as pd

from common import DATA, mrr25

SLOTS = {"1": [1], "1.3.5": [1, 3, 5], "2.4.6": [2, 4, 6], "1.2.3": [1, 2, 3], "2.3.4": [2, 3, 4],
         "3.5.7": [3, 5, 7], "2.4.6.8.10": [2, 4, 6, 8, 10], "1.3.5.7.9": [1, 3, 5, 7, 9]}
FS = [0.6, 0.8, 1.0, 1.2, 1.5]
K_MERGE = 10   # Enamine candidates allowed into the merge


def main():
    rows = []
    for ik, cls, ranked, sc, e_rank, e_sc, gated in pickle.load(open(DATA + "/enamine_slot_cache.pkl", "rb")):
        rows.append(("none", mrr25(ranked, ik), cls))
        for name, sl in SLOTS.items():
            out = list(ranked)
            if not gated:
                for r, k in zip(sl, e_rank):
                    out.insert(r - 1, k)
            rows.append(("slot " + name, mrr25(out, ik), cls))
        for f in FS:
            if gated:
                out = list(ranked)
            else:
                both = list(zip(sc, ranked)) + [(f * s, k) for s, k in zip(e_sc[:K_MERGE], e_rank[:K_MERGE])]
                out = [k for _, k in sorted(both, key=lambda x: -x[0])]
            rows.append((f"merge F{f:g}", mrr25(out, ik), cls))
    r = pd.DataFrame(rows, columns=["method", "mrr", "cls"]).pivot_table("mrr", "method", "cls")
    # LB-like weighting: hidden class 1 (library) 16%, answers outside the pool (local class 3) 55-80%
    r["w16/27/55"] = 0.16 * r[1] + 0.27 * r[2] + 0.55 * r[3]
    r["w16/0/84"] = 0.16 * r[1] + 0.84 * r[3]
    print(r.round(3).sort_values("w16/27/55").to_string())


if __name__ == "__main__":
    main()

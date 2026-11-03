"""Statistical review of every scored submission (2026-11-03 user rule: analyse past results regularly).

What it does
  enveda  : each submission = a public base (E006 / a451 / v15 / v22 / f465 / p477 / m479 / m480) + our parts.
            The part table is enveda-casmi26/eval/lb_parts.csv (one row per submission, hand-kept).
            1) OLS  score ~ base fixed effect + CPU flag + one coefficient per part.
               The public score of each base notebook is added as an "anchor" row (base with no parts).
               Residual SD = how far the parts are from adding up (interaction + rounding).
            2) exact pairs: two submissions on the same base and device whose part sets differ by one part.
            Effects are also shown in molecules (public LB is about 130 molecules; 1 molecule top-1 = 0.0077).
  gemma   : completed vs over-time runs against the config (minutes / calls / thinking / temperature / guard),
            and the score of the completed runs. Public LB is about 58 tasks (1 task = 0.017).
  arc / kaggriculture are finished; they are only listed.

Usage: python tools/stats/sub_stats.py [--cache DIR]   (fetches www.kaggle.com/api/v1 submissions/list when no cache)
"""
import argparse
import csv
import json
import os
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPS = {"enveda": "enveda-CASMI26-molecule-id-mass-spectra", "gemma": "gemma-4-developer-agent"}
ANCHOR = {"E006": 0.440, "a451": 0.451, "f465": 0.465, "p477": 0.477, "m479": 0.479, "m480": 0.480}
# kinds of part: new candidates into the list / re-scoring of the existing list / a prior on the whole list
CATS = {
    "inject": {"mmp", "dn", "pcl", "bt", "ownpc", "n1neg", "pcfix", "ownpcnp", "pi", "ownco", "fi"},
    "rescore": {"fg", "fgwide", "nl", "ion", "ionb", "decoy", "lfrag", "bf", "fp2", "nn", "fpnpc", "fpneng", "gg",
                "pcvote", "c1r", "arb", "rk", "dnlib", "dnslot", "pclcap", "pcguard", "latecut"},
    "prior": {"npprior", "tx", "txb", "np"},
}
MOL = 1 / 130          # one public molecule moved from wrong to rank 1
TASK = 1 / 58          # one public Gemma task


def fetch(slug, cache):
    path = os.path.join(cache, slug + ".json") if cache else None
    if path and os.path.exists(path):
        return json.load(open(path))
    d = json.load(urllib.request.urlopen(
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{slug}?pageSize=500", timeout=60))
    if path:
        json.dump(d, open(path, "w"))
    return d


def name_of(s):
    return (s.get("description") or "").split(":")[0].strip()


def enveda(subs):
    score = {name_of(s): float(s["publicScore"]) for s in subs if s.get("publicScore")}
    rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, "enveda-casmi26/eval/lb_parts.csv"))):
        y = ANCHOR.get(r["base"]) if r["name"].startswith("anchor:") else score.get(r["name"])
        if y is None:
            continue
        rows.append(dict(name=r["name"], base=r["base"], cpu=int(r["cpu"]),
                         parts=frozenset(p for p in r["parts"].split(";") if p), y=y))
    bases = sorted({r["base"] for r in rows})
    parts = sorted({p for r in rows for p in r["parts"]})
    cols = [f"base:{b}" for b in bases] + ["cpu"] + parts
    X = np.array([[r["base"] == b for b in bases] + [r["cpu"]] + [p in r["parts"] for p in parts] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    # tiny ridge only on the part columns so parts seen once stay estimable; bases and cpu are free
    lam = np.diag([0.0] * (len(bases) + 1) + [1e-6] * len(parts))
    beta = np.linalg.solve(X.T @ X + lam, X.T @ y)
    res = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    sig = np.sqrt(res @ res / dof)
    cov = sig ** 2 * np.linalg.pinv(X.T @ X + lam)
    se = np.sqrt(np.diag(cov))
    print(f"ENVEDA  rows {len(rows)} (scored submissions on a public base + anchors)  columns {len(cols)}  dof {dof}")
    print(f"  residual SD {sig:.4f} = {sig / MOL:.2f} molecules  (non-additivity + rounding; below this an effect is noise)")
    print(f"  {'column':12s} {'coef':>8s} {'se':>7s} {'t':>6s} {'mol':>6s}  n")
    order = sorted(range(len(cols)), key=lambda i: -beta[i] if i > len(bases) else -1e9 + i)
    for i in order:
        n = int(X[:, i].sum())
        if i < len(bases):
            print(f"  {cols[i]:12s} {beta[i]:8.4f}")
            continue
        print(f"  {cols[i]:12s} {beta[i]:+8.4f} {se[i]:7.4f} {beta[i] / se[i]:+6.2f} {beta[i] / MOL:+6.2f}  {n}")
    # empirical-Bayes shrinkage of the part effects toward their category mean (method of moments)
    pi = [i for i in range(len(bases) + 1, len(cols)) if se[i] < 0.05]
    for cat, members in CATS.items():
        idx = [i for i in pi if cols[i] in members]
        if not idx:
            continue
        b, s2 = beta[idx], se[idx] ** 2
        mu = np.average(b, weights=1 / s2)
        tau2 = max(np.var(b) - s2.mean(), 0.0)
        shr = mu + tau2 / (tau2 + s2) * (b - mu)
        print(f"\n  category {cat:8s} parts {len(idx):2d}  precision-weighted mean {mu:+.4f} ({mu / MOL:+.2f} mol)  "
              f"spread of true effects sd {np.sqrt(tau2):.4f}")
        print("    shrunken: " + "  ".join(f"{cols[i]} {v:+.4f}" for i, v in sorted(zip(idx, shr), key=lambda t: -t[1])))
    print("\n  exact pairs (same base and device, part sets differ by one part):  delta = with - without")
    pairs = {}
    for a in rows:
        for b in rows:
            if a["base"] == b["base"] and a["cpu"] == b["cpu"] and len(a["parts"] - b["parts"]) == 1 and b["parts"] < a["parts"]:
                p = next(iter(a["parts"] - b["parts"]))
                pairs.setdefault(p, []).append((round(a["y"] - b["y"], 3), a["name"], b["name"]))
    for p in sorted(pairs, key=lambda k: -np.mean([d for d, *_ in pairs[k]])):
        ds = [d for d, *_ in pairs[p]]
        print(f"  {p:9s} mean {np.mean(ds):+.4f} ({np.mean(ds) / MOL:+.1f} mol)  n {len(ds)}  "
              + "  ".join(f"{d:+.3f} [{a} vs {b}]" for d, a, b in pairs[p][:4]))
    return beta, cols


GEMMA_CFG = {  # version: (minutes per task, calls, thinking, temperature, own time guard)
    "g4v2": (4, 50, 0, 0.2, 0), "g4v3": (4, 50, 0, 0.2, 0), "g4v4": (8, 100, 4096, 0.2, 0), "g4v5": (5, 80, 0, 0.2, 0),
    "g4v6": (5, 80, 0, 0.2, 0), "g4v7": (5, 80, 0, 0.2, 0), "g4v12": (5, 80, 0, 0.2, 0), "g4v13": (5, 60, 2048, 0.2, 0),
    "g4v14": (5, 60, 2048, 0.2, 0), "g4v15": (8, 28, 4096, 0.2, 0), "g4v16": (7, 28, 4096, 0.2, 1),
    "g4v17": (5, 28, 4096, 0.2, 1), "g4v18": (8, 28, 4096, 0.2, 1), "g4v19": (8, 28, 4096, 0.2, 1),
    "g4v20": (8, 28, 4096, 0.2, 1), "g4v21": (8, 28, 4096, 0.2, 1), "g4v22": (8, 28, 4096, 0.2, 1),
    "g4v23": (8, 28, 4096, 0.2, 1), "g4v24p": (8, 28, 4096, 0.2, 1), "g4v25": (8, 28, 1024, 0.2, 1),
    "g4v26": (7, 20, 6144, 0.2, 1), "g4v27": (8, 28, 4096, 1.0, 0), "g4v28": (7, 28, 4096, 1.0, 1),
}


def gemma(subs):
    print("\nGEMMA  (public about 58 tasks; 1 task = 0.017)")
    done, over = [], []
    for s in reversed(subs):
        n = name_of(s)
        v = n.split("-")[0]
        cfg = GEMMA_CFG.get(v)
        err = (s.get("errorDescription") or "")
        st = "pending" if s.get("status") == "pending" else (
            "OVERTIME" if "runtime" in err else "ERROR" if err else s.get("publicScore"))
        print(f"  {n:10s} cfg(min,calls,think,temp,guard) {cfg}  -> {st}")
        if cfg and st == "OVERTIME":
            over.append(cfg)
        elif cfg and s.get("publicScore"):
            done.append((cfg, float(s["publicScore"])))
    for k, lab in ((0, "minutes/task"), (3, "temperature"), (4, "own time guard")):
        vals = sorted({c[k] for c, _ in done} | {c[k] for c in over})
        print(f"  {lab:15s} " + "  ".join(
            f"{v}: done {sum(c[k] == v for c, _ in done)} / over {sum(c[k] == v for c in over)}" for v in vals))
    sc = np.array([y for _, y in done])
    p = sc.mean()
    print(f"  completed runs {len(sc)}: mean {p:.3f}  SD {sc.std(ddof=1):.3f}  "
          f"binomial SD of one run at p={p:.2f}, 58 tasks: {np.sqrt(p * (1 - p) / 58):.3f}")
    for m in sorted({c[0] for c, _ in done} | {c[0] for c in over}):
        ok = [y for c, y in done if c[0] == m]
        n_over = sum(c[0] == m for c in over)
        if ok:
            pc = len(ok) / (len(ok) + n_over)
            print(f"  {m} min/task: finish rate {pc:.2f} ({len(ok)}/{len(ok) + n_over})  mean score when finished "
                  f"{np.mean(ok):.3f}  expected score (finish rate x score) {pc * np.mean(ok):.3f}")
    late = sc[-8:]
    print(f"  last 8 completed (same 28-call family): mean {late.mean():.3f}  SD {late.std(ddof=1):.3f} "
          f"= {late.std(ddof=1) / TASK:.1f} tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    a = ap.parse_args()
    enveda(fetch(COMPS["enveda"], a.cache))
    gemma(fetch(COMPS["gemma"], a.cache))


if __name__ == "__main__":
    main()

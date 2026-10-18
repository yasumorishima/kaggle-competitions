"""Validation report plus per-molecule selection rules between the post-fusion list and the late list.

Runs val_report.py first, then needs val_pconly.json (build.py pcdump=1: PubChem-only candidates of the merged list,
{molecule: {smiles: merged rank}}). Stages: S = first snapshot (gated merge), F = first snapshot that differs from S
(the fusion), L = second-to-last snapshot (the last base stage, before our final pcguard cell). Numbers only.
usage: python val_rules.py <dir>
"""
import glob, json, os, sys
import numpy as np, pandas as pd

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_report.py")).read())

pco = json.load(open(os.path.join(d, "val_pconly.json")))
st = sorted(glob.glob(os.path.join(d, "val_stage*.csv")))
rd = lambda f: {m: [x for x in s.split(";") if x and x != "CCO"] for m, s in
                pd.read_csv(f, dtype=str, keep_default_na=False).itertuples(index=False)}
S = rd(st[0])
iF = next(i for i, f in enumerate(st) if rd(f) != S)
F, L, Z = rd(st[iF]), rd(st[-2]), rd(st[-1])
print("RULES S", os.path.basename(st[0]), "| F", os.path.basename(st[iF]), "| L", os.path.basename(st[-2]),
      "| last", os.path.basename(st[-1]), "| last differs from L in", sum(Z[m] != L[m] for m in L), "molecules")
key = lambda s: info(s)[0] or s


def pk(m):
    return set(key(c) for c in pco.get(m, {}))


def fixto(lst, ref, m):  # PubChem-only candidates return to their rank in ref when lst has them lower or not at all
    P = pk(m)
    new = list(lst)
    for r1, c in sorted((i + 1, c) for i, c in enumerate(ref) if key(c) in P):
        cur = next((i for i, x in enumerate(new) if key(x) == key(c)), None)
        if cur is not None and cur + 1 <= r1:
            continue
        it = new.pop(cur) if cur is not None else c
        new.insert(min(r1 - 1, len(new)), it)
    return new[:25]


def top1pc(lst, m):
    return bool(lst) and key(lst[0]) in pk(m)


rules = {
    "L (base late)": lambda m: L[m],
    "F (fused)": lambda m: F[m],
    "guard: F if F top1 PC-only": lambda m: F[m] if top1pc(F[m], m) else L[m],
    "guard2: F if F top1 PC-only and S top1 same": lambda m: F[m] if top1pc(F[m], m) and S[m][:1] == F[m][:1] else L[m],
    "guardS: F if S top1 PC-only": lambda m: F[m] if top1pc(S[m], m) else L[m],
    "lift: L with F top1 put first if PC-only": lambda m: ([F[m][0]] + [x for x in L[m] if key(x) != key(F[m][0])])[:25] if top1pc(F[m], m) else L[m],
    "fixF: PC-only back to F rank in L": lambda m: fixto(L[m], F[m], m),
    "fixS: PC-only back to S rank in L": lambda m: fixto(L[m], S[m], m),
    "fixS on F": lambda m: fixto(F[m], S[m], m),
}
for nm, fn in rules.items():
    out = {c: [] for c in ("all", "c1", "c2")}
    nch = 0
    for m in L:
        if m not in tk:
            continue
        l = fn(m)
        nch += l[:25] != L[m][:25]
        k = [key(x) for x in l[:25]]
        rr = k.index(tk[m]) + 1 if tk[m] in k else 0
        v = 1 / rr if rr else 0.0
        out["all"].append(v); out[cls[m]].append(v)
    print(f"RULE {nm:46s} all {np.mean(out['all']):.4f} c1 {np.mean(out['c1']):.4f} c2 {np.mean(out['c2']):.4f} | changed {nch}")

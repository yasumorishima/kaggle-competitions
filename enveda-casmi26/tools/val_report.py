"""Score the stage snapshots of a validation run (build.py val=N) against val_truth.csv.

Run by the enveda workflow's "output" action on the downloaded kernel output (the files stay out of the repo:
val_truth.csv is drawn from the competition's train.parquet). Prints numbers only, no structures:
per stage and class (c1 = reference spectra in other libraries, c2 = enveda-180 only):
MRR@25, top1, hit@5, hit@25, top-1 formula correct, truth formula anywhere in the list; then a per-molecule
rank table (molecule id, class, rank per stage).
usage: python val_report.py <dir with val_truth.csv and val_stage*.csv>
"""
import glob, os, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
d = sys.argv[1]
_c = {}


def info(s):
    if s not in _c:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        _c[s] = (Chem.MolToInchiKey(m)[:14], CalcMolFormula(m)) if m is not None else (None, None)
    return _c[s]


t = pd.read_csv(os.path.join(d, "val_truth.csv"))
tk = {m: (info(s)[0] or ik) for m, s, ik in zip(t.molecule_id, t.normalized_smiles, t.inchikey14)}
tf = dict(zip(t.molecule_id, t.molecular_formula))
cls = dict(zip(t.molecule_id, t.cls))
files = sorted(glob.glob(os.path.join(d, "val_stage*.csv")))
for extra in ("submission.csv",):
    if os.path.isfile(os.path.join(d, extra)):
        files.append(os.path.join(d, extra))
ranks = {}
print("VALREPORT", len(t), "molecules | c1", sum(v == "c1" for v in cls.values()), "c2", sum(v == "c2" for v in cls.values()))
for f in files:
    s = pd.read_csv(f, dtype=str, keep_default_na=False)
    r, f1, fin = {}, {}, {}
    for m, l in zip(s.molecule_id, s.smiles):
        if m not in tk:
            continue
        inf = [info(x) for x in l.split(";")[:25]]
        r[m] = next((i + 1 for i, x in enumerate(inf) if x[0] == tk[m]), 0)
        f1[m] = bool(inf) and inf[0][1] == tf[m]
        fin[m] = any(x[1] == tf[m] for x in inf)
    nm = os.path.basename(f).replace(".csv", "")
    ranks[nm] = r
    for c in ("all", "c1", "c2", "c3"):
        ms = [m for m in r if c == "all" or cls[m] == c]
        a = np.array([r[m] for m in ms])
        if not len(a):
            continue
        print(f"{nm:32s} {c:3s} n{len(a):4d} mrr {np.mean(np.where(a > 0, 1 / np.maximum(a, 1), 0)):.4f} "
              f"top1 {np.mean(a == 1):.3f} hit5 {np.mean((a > 0) & (a <= 5)):.3f} hit25 {np.mean(a > 0):.3f} "
              f"form1 {np.mean([f1[m] for m in ms]):.3f} formAny {np.mean([fin[m] for m in ms]):.3f}")
rk = pd.DataFrame(ranks)
rk.insert(0, "cls", [cls[m] for m in rk.index])
rk.columns = ["cls"] + [c.replace("val_stage", "s").split("_after")[0] for c in rk.columns[1:]]
print("RANKS (0 = not in the top 25)")
print(rk.sort_index().to_csv())

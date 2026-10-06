"""Does a tighter precursor mass window (or a soft mass-error penalty) help?

enveda-180 (timsTOF, same instrument as the hidden test) has |precursor error| median 1.0 ppm,
99% below 4.4 ppm, max 7.1 ppm, while the kernel's candidate window is +-10 ppm. Candidates
far out in the window are rarely the answer, so dropping or down-weighting them should lift
every class a little.

Uses the cache from enamine_slot.py (pool ranking + scores per split molecule). The molecule's
neutral mass is the median over its split rows, as in the kernel.

    python ppm_window.py
"""
import pickle

import numpy as np
import pandas as pd

from common import DATA, mrr25, neutral_mass


def main():
    split = pd.read_parquet(DATA + "/split.parquet")
    tr = pd.read_parquet(DATA + "/train.parquet", columns=["precursor_mz"])
    split["M"] = [neutral_mass(m, a) for m, a in zip(tr.precursor_mz.values[split.row.values], split.adduct)]
    M = split.groupby("inchikey14").M.median()
    S = pd.read_parquet(DATA + "/structures.parquet", columns=["inchikey14", "fM"])
    coco = pd.read_parquet(DATA + "/coconut.parquet", columns=["inchikey14", "fM"])
    fM = {**dict(zip(coco.inchikey14, coco.fM)), **dict(zip(S.inchikey14, S.fM))}

    errs, rows = [], []
    for ik, cls, ranked, sc, *_ in pickle.load(open(DATA + "/enamine_slot_cache.pkl", "rb")):
        m0 = M.get(ik)
        if m0 is None or not np.isfinite(m0):
            continue
        ppm = np.array([abs(fM.get(c, np.nan) - m0) / m0 * 1e6 for c in ranked])
        if ik in fM:
            errs.append((cls, abs(fM[ik] - m0) / m0 * 1e6))
        sc = np.asarray(sc, float)
        gated = sc >= 1000
        rows.append(("none", cls, mrr25(ranked, ik)))
        for cut in (3, 4, 5, 6, 7):
            keep = [c for c, p in zip(ranked, ppm) if not p > cut]
            rows.append((f"cut{cut}", cls, mrr25(keep, ik)))
        for sig in (2, 3, 4):
            for a in (0.1, 0.3, 1.0):
                pen = np.where(gated, sc, sc - a * (np.nan_to_num(ppm, nan=10) / sig) ** 2)
                out = [ranked[i] for i in np.argsort(-pen, kind="stable")]
                rows.append((f"soft s{sig} a{a}", cls, mrr25(out, ik)))
    e = pd.DataFrame(errs, columns=["cls", "ppm"])
    print("answer |error| ppm by class (molecule median mass):")
    print(e.groupby("cls").ppm.describe(percentiles=[.5, .9, .99]).round(2).to_string())
    r = pd.DataFrame(rows, columns=["method", "cls", "mrr"]).pivot_table("mrr", "method", "cls")
    print(r.round(4).to_string())


if __name__ == "__main__":
    main()

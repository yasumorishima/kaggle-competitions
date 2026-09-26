"""B0: mass-window candidates from train structures, ranked by library entropy similarity.

For each query molecule: neutral mass = median over its spectra; candidates are
pool structures within +-PPM of it; each candidate scores the best entropy
similarity between any query spectrum and any of its library spectra (0 when it
has none). Prints MRR@25 by class and how often the truth is in the window.

    python baseline_lib.py [n_per_class]
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import calculate_entropy_similarity, clean_spectrum

from common import DATA, mrr25, neutral_mass

PPM = 10.0
TOL_DA = 0.02


def peaks(mz, it):
    p = np.column_stack([np.asarray(mz, np.float32), np.asarray(it, np.float32)])
    return clean_spectrum(p, min_ms2_difference_in_da=2 * TOL_DA)


def score_one(args):
    qspecs, cands = args          # cands: list of (ik14, [lib peaks])
    out = []
    for ik, libs in cands:
        best = 0.0
        for q in qspecs:
            for l in libs:
                s = calculate_entropy_similarity(q, l, ms2_tolerance_in_da=TOL_DA, clean_spectra=False)
                if s > best:
                    best = s
        out.append((ik, best))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time()
    meta = pd.read_parquet(DATA + "/train_meta.parquet",
                           columns=["ingest_lib", "inchikey14", "precursor_mz", "adduct"])
    split = pd.read_parquet(DATA + "/split.parquet")
    split = split[split.inchikey14.isin(
        split.drop_duplicates("inchikey14").groupby("cls").head(n).inchikey14)]
    S = pd.read_parquet(DATA + "/structures.parquet").dropna(subset=["fM"])
    c3 = set(split.loc[split.cls == 3, "inchikey14"])
    pool = S[~S.inchikey14.isin(c3)].sort_values("fM")
    pm, pk = pool.fM.values, pool.inchikey14.values

    # library = train minus every enveda-180 spectrum of a query molecule
    held = set(split.inchikey14)
    drop = meta.inchikey14.isin(held) & (meta.ingest_lib == "enveda-180")
    lib_rows = np.flatnonzero(~drop.values)

    q = split.merge(meta[["precursor_mz"]], left_on="row", right_index=True)
    q["M"] = [neutral_mass(m, a) for m, a in zip(q.precursor_mz, q.adduct)]
    qm = q.groupby("inchikey14").agg(M=("M", "median"), cls=("cls", "first"), rows=("row", list))

    cand = {}
    for ik, r in qm.iterrows():
        lo, hi = np.searchsorted(pm, [r.M * (1 - PPM * 1e-6), r.M * (1 + PPM * 1e-6)])
        cand[ik] = list(pk[lo:hi])
    need = set(c for v in cand.values() for c in v)
    lib_meta = meta.iloc[lib_rows]
    lib_rows = lib_rows[lib_meta.inchikey14.isin(need).values]
    rows_needed = np.union1d(lib_rows, np.concatenate(qm.rows.values))
    print(f"queries {len(qm)}  candidates/query median {np.median([len(v) for v in cand.values()]):.0f}"
          f"  library spectra to load {len(lib_rows)}  ({time.time()-t0:.0f}s)")

    tab = pq.read_table(DATA + "/train.parquet", columns=["ms2_mzs", "ms2_normalized_intensities"])
    tab = tab.take(rows_needed)
    mz, it = tab.column(0).to_pylist(), tab.column(1).to_pylist()
    spec = {r: peaks(a, b) for r, a, b in zip(rows_needed, mz, it)}
    lib_by_ik = {}
    for r, ik in zip(lib_rows, meta.inchikey14.values[lib_rows]):
        lib_by_ik.setdefault(ik, []).append(spec[r])
    print(f"spectra loaded ({time.time()-t0:.0f}s)")

    jobs = [([spec[r] for r in qm.rows[ik]], [(c, lib_by_ik.get(c, [])) for c in cand[ik]]) for ik in qm.index]
    with Pool(4) as p:
        res = p.map(score_one, jobs, chunksize=4)
    rec = []
    for ik, sc in zip(qm.index, res):
        ranked = [c for c, s in sorted(sc, key=lambda x: -x[1])]
        rec.append((qm.cls[ik], ik in cand[ik], mrr25(ranked, ik)))
    r = pd.DataFrame(rec, columns=["cls", "in_window", "mrr"])
    print(r.groupby("cls").agg(n=("mrr", "size"), in_window=("in_window", "mean"), mrr=("mrr", "mean")).round(3))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

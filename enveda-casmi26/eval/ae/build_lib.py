"""ae library index (ours, 11-04): one compact spectrum table for the analog search of the ae part.

For every (structure inchikey14, ionization mode, adduct) of the 10 test adducts, keep up to 2 train spectra
(prefer one from enveda-180 = timsTOF like the hidden test, then the one with most peaks). Peaks <= precursor + 1,
top 32 by intensity, sqrt(intensity / max). Output (npz + parquet) is what the kernel cell reads.
Usage: python build_lib.py TRAIN_PARQUET OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
K = 32


def main(src, out):
    f = pq.ParquetFile(src)
    cols = ["inchikey14", "normalized_smiles", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs",
            "ms2_normalized_intensities", "ingest_lib"]
    keep = {}          # (ik, grp) -> list of (prio, npk, mz, it, prec)
    smiles = {}
    grp_id = {}
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=cols).to_pandas()
        t = t[t.adduct.isin(ADDUCTS)]
        for r in t.itertuples(index=False):
            mz = np.asarray(r.ms2_mzs, np.float64)
            it = np.asarray(r.ms2_normalized_intensities, np.float64)
            prec = float(r.precursor_mz)
            m = (mz <= prec + 1.0) & (it > 0)
            if m.sum() < 3:
                continue
            mz, it = mz[m], it[m]
            o = np.argsort(-it)[:K]
            mz, it = mz[o], np.sqrt(it[o] / it[o[0]])
            g = f"{r.ionization_mode}|{r.adduct}"
            gi = grp_id.setdefault(g, len(grp_id))
            ik = str(r.inchikey14)
            smiles.setdefault(ik, str(r.normalized_smiles))
            prio = (0 if r.ingest_lib == "enveda-180" else 1, -len(mz))
            lst = keep.setdefault((ik, gi), [])
            lst.append((prio, mz.astype(np.float32), it.astype(np.float32), prec))
            if len(lst) > 2:
                lst.sort(key=lambda x: x[0])
                # keep the best enveda-180 one and the best other one when both exist
                e = [x for x in lst if x[0][0] == 0][:1]
                o_ = [x for x in lst if x[0][0] == 1][:1]
                lst[:] = (e + o_) if e and o_ else lst[:2]
        print("row group", rg, "/", f.num_row_groups, "kept", len(keep), flush=True)
    iks = sorted(smiles)
    sidx = {k: i for i, k in enumerate(iks)}
    rows = [(sidx[ik], gi, x) for (ik, gi), lst in keep.items() for x in lst]
    n = len(rows)
    PM = np.zeros((n, K), np.float32)
    PI = np.zeros((n, K), np.float32)
    PN = np.zeros(n, np.int16)
    PR = np.zeros(n, np.float64)
    G = np.zeros(n, np.int16)
    SI = np.zeros(n, np.int32)
    for j, (si, gi, (_, mz, it, prec)) in enumerate(rows):
        PM[j, :len(mz)], PI[j, :len(mz)], PN[j], PR[j], G[j], SI[j] = mz, it, len(mz), prec, gi, si
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "ae_lib.npz"), mz=PM, it=PI, n=PN, prec=PR, grp=G, sidx=SI)
    pd.DataFrame({"inchikey14": iks, "smiles": [smiles[k] for k in iks]}).to_parquet(os.path.join(out, "ae_struct.parquet"))
    pd.DataFrame({"group": sorted(grp_id, key=grp_id.get)}).to_parquet(os.path.join(out, "ae_groups.parquet"))
    print("spectra", n, "structures", len(iks), "groups", grp_id)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

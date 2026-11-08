"""ae library + extlib (ours, 11-08): append takumuhata/casmi26-extlib (CC0, 200k spectra of structures not in train)
to the ae library index, so analog-edit can start from analogs outside train.

Same compaction as build_lib.py (peaks <= precursor + 1, top 32, sqrt(intensity / max), up to 2 spectra per structure
and group); extlib adduct names are mapped to the test names; other adducts are dropped.
Usage: python build_lib_ext.py AE_LIB_DIR EXTLIB_DIR OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

K = 32
MAP = {"[M+H]+": "[M+H]+", "[M+NH4]+": "[M+NH4]+", "[M+H-H2O]+": "[M-H2O+H]+", "[M-H2O+H]+": "[M-H2O+H]+",
       "[M-2H2O+H]+": "[M-2H2O+H]+", "[M+H-2H2O]+": "[M-2H2O+H]+", "[M+Na]+": "[M+Na]+", "[M+K]+": "[M+K]+",
       "[M-H]-": "[M-H]-", "[M-H-H2O]-": "[M-H2O-H]-", "[M-H2O-H]-": "[M-H2O-H]-", "[M+HCOO]-": "[M+CH2O2-H]-",
       "[M+CH2O2-H]-": "[M+CH2O2-H]-", "[M+FA-H]-": "[M+CH2O2-H]-", "[M+Cl]-": "[M+Cl]-"}


def main(ae, ext, out):
    z = dict(np.load(os.path.join(ae, "ae_lib.npz")))
    st = pd.read_parquet(os.path.join(ae, "ae_struct.parquet"))
    groups = pd.read_parquet(os.path.join(ae, "ae_groups.parquet")).group.tolist()
    gid = {g: i for i, g in enumerate(groups)}
    sidx = {k: i for i, k in enumerate(st.inchikey14)}
    iks, smis = list(st.inchikey14), list(st.smiles)
    meta = pd.read_parquet(os.path.join(ext, "ext_meta.parquet"))
    mz = np.load(os.path.join(ext, "ext_mz.npy"), mmap_mode="r")
    it = np.load(os.path.join(ext, "ext_in.npy"), mmap_mode="r")
    cnt, rows = {}, []
    for j, r in enumerate(meta.itertuples(index=False)):
        a = MAP.get(r.adduct)
        if a is None:
            continue
        g = f"{r.ionization_mode}|{a}"
        if g not in gid:
            continue
        ik = str(r.inchikey14)
        if ik not in sidx:
            sidx[ik] = len(iks)
            iks.append(ik)
            smis.append(str(r.normalized_smiles))
        key = (sidx[ik], gid[g])
        if cnt.get(key, 0) >= 2:
            continue
        m_, i_ = np.asarray(mz[j], np.float64), np.asarray(it[j], np.float64)
        prec = float(r.precursor_mz)
        k = (m_ > 0) & (i_ > 0) & (m_ <= prec + 1.0)
        if k.sum() < 3:
            continue
        m_, i_ = m_[k], i_[k]
        o = np.argsort(-i_)[:K]
        rows.append((key[0], key[1], m_[o], np.sqrt(i_[o] / i_[o[0]]), prec))
        cnt[key] = cnt.get(key, 0) + 1
    n0, n = len(z["prec"]), len(rows)
    PM = np.zeros((n0 + n, K), np.float32)
    PI = np.zeros((n0 + n, K), np.float32)
    PM[:n0], PI[:n0] = z["mz"], z["it"]
    PN = np.r_[z["n"], np.zeros(n, np.int16)].astype(np.int16)
    PR = np.r_[z["prec"], np.zeros(n)]
    G = np.r_[z["grp"], np.zeros(n, np.int16)].astype(np.int16)
    SI = np.r_[z["sidx"], np.zeros(n, np.int32)].astype(np.int32)
    for j, (si, gi, m_, i_, prec) in enumerate(rows):
        t = n0 + j
        PM[t, :len(m_)], PI[t, :len(m_)], PN[t], PR[t], G[t], SI[t] = m_, i_, len(m_), prec, gi, si
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "ae_lib.npz"), mz=PM, it=PI, n=PN, prec=PR, grp=G, sidx=SI)
    pd.DataFrame({"inchikey14": iks, "smiles": smis}).to_parquet(os.path.join(out, "ae_struct.parquet"))
    pd.DataFrame({"group": groups}).to_parquet(os.path.join(out, "ae_groups.parquet"))
    print(f"train spectra {n0}  extlib spectra added {n}  structures {len(st)} -> {len(iks)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

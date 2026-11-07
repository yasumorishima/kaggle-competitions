"""ae v1 rows (ours, 11-07): one row per analog-edit product of simulated molecules, for a learned pick (ours_aesite).

Same simulation as sim.py: N enveda-180 structures (seed S) are hidden from the ae library and queried with their own
enveda-180 spectra (<= 4); every new product of the gated molecules gets the ours_aesite features and label = (product
== answer, inchikey14). Molecules are split over P worker processes.
Usage: python site.py TRAIN_PARQUET LIB_DIR N SEED OUT_PARQUET [P]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels", "e3"))
import ours_ae   # noqa: E402
import ours_aesite   # noqa: E402
import ours_frag   # noqa: E402

ADD = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
G = {}


def _init(libdir, hidden, tab):
    st = pd.read_parquet(os.path.join(libdir, "ae_struct.parquet"))
    gr = pd.read_parquet(os.path.join(libdir, "ae_groups.parquet")).group.tolist()
    lib = ours_ae.Library(os.path.join(libdir, "ae_lib.npz"), st.smiles.tolist(), gr)
    sidx = {k: i for i, k in enumerate(st.inchikey14)}
    lib.mask = np.zeros(len(st), bool)
    lib.mask[[sidx[k] for k in hidden if k in sidx]] = True
    G.update(lib=lib, tab=tab)


def _one(ik):
    rows = G["tab"].get(ik)
    specs = ours_ae.query_specs(zip(rows.ionization_mode, rows.adduct, rows.precursor_mz, rows.ms2_mzs, rows.ms2_normalized_intensities))
    fr = [(np.c_[np.asarray(a, float), np.asarray(b, float)], float(p), ours_frag.polarity_of(ad, md))
          for md, ad, p, a, b in zip(rows.ionization_mode, rows.adduct, rows.precursor_mz, rows.ms2_mzs, rows.ms2_normalized_intensities)]
    try:
        X, info = ours_aesite.product_rows(G["lib"], specs, fr, ours_frag, tau=0.95)
    except Exception as e:   # noqa: BLE001
        return ik, None, dict(gate=0, error=repr(e)[:100])
    if X is None:
        return ik, None, info
    y = np.array([int(k == ik) for k in info["keys"]])
    return ik, (X, y, info["keys"]), {k: v for k, v in info.items() if k in ("gate", "n_new", "n_analogs", "modcos")}


def main(src, libdir, n, seed, out, procs=4):
    t = pq.read_table(src, columns=["inchikey14", "ingest_lib", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs",
                                    "ms2_normalized_intensities"], filters=[("ingest_lib", "=", "enveda-180")]).to_pandas()
    t = t[t.adduct.isin(ADD)]
    rng = np.random.default_rng(seed)
    iks = rng.choice(t.inchikey14.unique(), n, replace=False)
    sub = t[t.inchikey14.isin(set(iks))]
    tab = {k: g.head(4) for k, g in sub.groupby("inchikey14")}
    del t
    t0 = time.time()
    recs, gate, orc = [], 0, 0
    with Pool(procs, initializer=_init, initargs=(libdir, list(iks), tab)) as pool:
        for j, (ik, res, info) in enumerate(pool.imap_unordered(_one, list(iks), chunksize=8)):
            gate += int(bool(info.get("gate")))
            if res is not None:
                X, y, keys = res
                orc += int(y.any())
                d = pd.DataFrame(X, columns=ours_aesite.COLS)
                d.insert(0, "label", y)
                d.insert(0, "key", keys)
                d.insert(0, "mol", ik)
                recs.append(d)
            if (j + 1) % 500 == 0:
                print(f"{j + 1} molecules  gate {gate}  oracle {orc}  {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(recs, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"N {n} seed {seed}: gate {gate} ({gate / n:.3f})  with products {df.mol.nunique()}  oracle {orc} "
          f"({orc / max(gate, 1):.3f} of gated)  rows {len(df)}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), int(a[4]), a[5], int(a[6]) if len(a) > 6 else 4)

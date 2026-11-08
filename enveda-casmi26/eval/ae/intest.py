"""Test-internal analog edit (ours, 11-08; idea from the 11-08 plan audit): link molecules of the same test set.

The hidden ~400 molecules come from the same extracts and may contain analog series. For a test molecule B, look for
another test molecule A whose spectrum matches B's at one modification shift (modified cosine >= tau, same groups as
ae), take A's structure as known (A confidently identified, e.g. by a strong library match) and propose its one-step
edits for B. Here A's true structure stands in for the confident rank 1.
We draw R random sets of N enveda-180 molecules (the visible test is such a draw, so this is the pessimistic case for
series) and report: the pair rate (B with any test-internal analog), the oracle (answer among the edits) and the
product counts.
Usage: python intest.py TRAIN_PARQUET N R SEED
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels", "e3"))
import ours_ae as _ae   # noqa: E402

ADD = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]


def mini_lib(tab, iks, smiles):
    rows, groups = [], {}
    for si, ik in enumerate(iks):
        r = tab[ik]
        for g, prec, mz, it in _ae.query_specs(zip(r.ionization_mode, r.adduct, r.precursor_mz, r.ms2_mzs,
                                                    r.ms2_normalized_intensities)):
            rows.append((si, groups.setdefault(g, len(groups)), prec, mz, it))
    lib = object.__new__(_ae.Library)
    n = len(rows)
    lib.mz = np.zeros((n, 32), np.float32)
    lib.it = np.zeros((n, 32), np.float32)
    lib.n, lib.prec = np.zeros(n, np.int16), np.zeros(n)
    lib.grp, lib.sidx = np.zeros(n, np.int16), np.zeros(n, np.int32)
    for j, (si, gi, prec, mz, it) in enumerate(rows):
        lib.mz[j, :len(mz)], lib.it[j, :len(mz)], lib.n[j], lib.prec[j], lib.grp[j], lib.sidx[j] = mz, it, len(mz), prec, gi, si
    lib.smiles = [smiles[k] for k in iks]
    lib.gid = groups
    lib.by_g = {}
    for g in range(len(groups)):
        r = np.flatnonzero(lib.grp == g)
        o = np.argsort(lib.prec[r], kind="stable")
        lib.by_g[g] = (r[o], lib.prec[r[o]])
    lib.mask = None
    return lib


def main(src, n, reps, seed, tau=0.95):
    t = pq.read_table(src, columns=["inchikey14", "normalized_smiles", "ingest_lib", "ionization_mode", "adduct",
                                    "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"],
                      filters=[("ingest_lib", "=", "enveda-180")]).to_pandas()
    t = t[t.adduct.isin(ADD)]
    smiles = t.groupby("inchikey14").normalized_smiles.first().to_dict()
    allik = np.array(sorted(smiles))
    rng = np.random.default_rng(seed)
    tot = dict(mol=0, pair=0, orc=0, prods=[])
    for rep in range(reps):
        iks = list(rng.choice(allik, n, replace=False))
        sub = t[t.inchikey14.isin(set(iks))]
        tab = {k: g.head(4) for k, g in sub.groupby("inchikey14")}
        lib = mini_lib(tab, iks, smiles)
        for bi, ik in enumerate(iks):
            r = tab[ik]
            specs = _ae.query_specs(zip(r.ionization_mode, r.adduct, r.precursor_mz, r.ms2_mzs, r.ms2_normalized_intensities))
            lib.mask = np.zeros(len(iks), bool)
            lib.mask[bi] = True
            an = lib.analogs(specs, tau, top=5)
            tot["mol"] += 1
            if not an:
                continue
            tot["pair"] += 1
            prods = {}
            for s, si, kind, direction, *_ in an:
                prods.update(_ae.edits(lib.smiles[si], kind, direction))
            tot["prods"].append(len(prods))
            tot["orc"] += int(ik in prods)
        print(f"rep {rep}: molecules {tot['mol']}  with a test-internal analog {tot['pair']}  oracle {tot['orc']}", flush=True)
    m, p = tot["mol"], tot["pair"]
    print(f"N {n} x {reps}: pair rate {p / m:.4f}  oracle {tot['orc']} ({tot['orc'] / m:.4f} of all, "
          f"{tot['orc'] / max(p, 1):.3f} of paired)  products median {np.median(tot['prods']) if tot['prods'] else 0}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), int(a[4]))

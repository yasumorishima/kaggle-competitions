"""ae two-step oracle (ours, 11-08): would two modifications per analog widen ae?

Same simulation as site.py: N enveda-180 structures (seed S) are hidden from the ae library and queried with their own
spectra (<= 4). For each molecule we record
  - the one-step gate (an analog at a single modification shift with modified cosine >= tau) and whether the answer is
    among its one-step edits (the oracle of ae v0 / v1),
  - the two-step gate: an analog at the shift of a pair of modifications (common ones, both orders of the edits), and
    whether the answer is among the two-step edits, with the product count.
Usage: python oracle2.py TRAIN_PARQUET LIB_DIR N SEED [P]
"""
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels", "e3"))
import ours_ae as _ae   # noqa: E402

ADD = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
KINDS = ["CH2", "O", "H2", "C2H2O", "C6H10O5", "C5H8O4", "C6H10O4", "C6H8O6", "CO2"]
TAU = 0.95
CAP1, CAP2 = 40, 3000
G = {}


def _mass(k):
    return _ae.H2 if k == "H2" else _ae.MODS[k][0]


def pair_targets():
    items = [(k, d) for k in KINDS for d in ("add", "remove")]
    singles = [abs(t[1]) for t in _ae.TARGETS]
    out = []
    for (k1, d1), (k2, d2) in itertools.combinations_with_replacement(items, 2):
        m = (_mass(k1) if d1 == "add" else -_mass(k1)) + (_mass(k2) if d2 == "add" else -_mass(k2))
        if abs(m) < 1.0 or min(abs(abs(m) - s) for s in singles) < 0.005:
            continue
        out.append((f"{k1}|{d1}|{k2}|{d2}", m, "pair"))
    return out


def _init(libdir, hidden, tab):
    st = pd.read_parquet(os.path.join(libdir, "ae_struct.parquet"))
    gr = pd.read_parquet(os.path.join(libdir, "ae_groups.parquet")).group.tolist()
    lib = _ae.Library(os.path.join(libdir, "ae_lib.npz"), st.smiles.tolist(), gr)
    sidx = {k: i for i, k in enumerate(st.inchikey14)}
    lib.mask = np.zeros(len(st), bool)
    lib.mask[[sidx[k] for k in hidden if k in sidx]] = True
    G.update(lib=lib, tab=tab, single=list(_ae.TARGETS), pair=pair_targets())


def _two(smi, k1, d1, k2, d2):
    out = {}
    for (a, da), (b, db) in (((k1, d1), (k2, d2)), ((k2, d2), (k1, d1))):
        for s1 in list(_ae.edits(smi, a, da).values())[:CAP1]:
            out.update(_ae.edits(s1, b, db))
            if len(out) >= CAP2:
                return out
    return out


def _one(ik):
    rows = G["tab"][ik]
    specs = _ae.query_specs(zip(rows.ionization_mode, rows.adduct, rows.precursor_mz, rows.ms2_mzs, rows.ms2_normalized_intensities))
    lib = G["lib"]
    r = dict(g1=0, o1=0, n1=0, g2=0, o2=0, n2=0, cos2=0.0)
    try:
        _ae.TARGETS[:] = G["single"]
        an = lib.analogs(specs, TAU, top=5)
        if an:
            r["g1"] = 1
            prods = {}
            for s, si, kind, direction, *_ in an:
                prods.update(_ae.edits(lib.smiles[si], kind, direction))
            r["n1"], r["o1"] = len(prods), int(ik in prods)
        _ae.TARGETS[:] = G["pair"]
        an = lib.analogs(specs, TAU, top=3)
        if an:
            r["g2"], r["cos2"] = 1, round(float(an[0][0]), 3)
            prods = {}
            for s, si, kind, direction, *_ in an:
                k1, d1, k2, d2 = kind.split("|")
                prods.update(_two(lib.smiles[si], k1, d1, k2, d2))
            r["n2"], r["o2"] = len(prods), int(ik in prods)
    except Exception as e:   # noqa: BLE001
        r["err"] = repr(e)[:80]
    finally:
        _ae.TARGETS[:] = G["single"]
    return ik, r


def main(src, libdir, n, seed, procs=4):
    t = pq.read_table(src, columns=["inchikey14", "ingest_lib", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs",
                                    "ms2_normalized_intensities"], filters=[("ingest_lib", "=", "enveda-180")]).to_pandas()
    t = t[t.adduct.isin(ADD)]
    rng = np.random.default_rng(seed)
    iks = rng.choice(t.inchikey14.unique(), n, replace=False)
    sub = t[t.inchikey14.isin(set(iks))]
    tab = {k: g.head(4) for k, g in sub.groupby("inchikey14")}
    del t
    t0 = time.time()
    recs = []
    with Pool(procs, initializer=_init, initargs=(libdir, list(iks), tab)) as pool:
        for j, (ik, r) in enumerate(pool.imap_unordered(_one, list(iks), chunksize=4)):
            recs.append(dict(mol=ik, **r))
            if (j + 1) % 250 == 0:
                print(f"{j + 1} molecules  {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(recs)
    d.to_csv(f"oracle2_{n}_{seed}.csv", index=False)
    n = len(d)
    print(f"N {n}  errors {d.get('err', pd.Series(dtype=str)).notna().sum()}  {time.time() - t0:.0f}s")
    print(f"one-step: gate {d.g1.sum()} ({d.g1.mean():.3f})  oracle {d.o1.sum()} ({d.o1.sum() / max(d.g1.sum(), 1):.3f} of gated)  "
          f"products median {d.n1[d.g1 == 1].median()}")
    new = d[(d.g1 == 0) & (d.g2 == 1)]
    print(f"two-step gate {d.g2.sum()} ({d.g2.mean():.3f}); new molecules (no one-step gate) {len(new)} ({len(new) / n:.3f})  "
          f"oracle {new.o2.sum()} ({new.o2.sum() / max(len(new), 1):.3f})  products median {new.n2.median()}")
    both = d[(d.g1 == 1) & (d.o1 == 0) & (d.g2 == 1)]
    print(f"one-step gated but missed, two-step gated {len(both)}  two-step oracle {both.o2.sum()}")
    for lo, hi in ((0, 20), (20, 100), (100, 400), (400, 10 ** 6)):
        s = new[(new.n2 > lo) & (new.n2 <= hi)]
        print(f"  new, products ({lo},{hi}]: {len(s)}  oracle {s.o2.sum()}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), int(a[4]), int(a[5]) if len(a) > 5 else 4)

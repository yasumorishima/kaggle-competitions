"""Validation queries that mimic the hidden test's classes, built from train.

Every query is a timsTOF molecule (as in the test), restricted to the ten test
adducts, with its own enveda-180 spectra as the query spectra:

  c1  structure also has spectra in a public library -> those stay in the library
  c2  structure only in enveda-180 -> all its spectra leave the library, the
      structure stays in the candidate pool (as a PubChem/COCONUT hit would)
  c3  as c2, and the structure also leaves the pool (novel)

    python split.py            # writes $CASMI_DATA/split.parquet
"""
import numpy as np
import pandas as pd

from common import ADDUCTS, DATA

TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
N_PER_CLASS = 400
SEED = 2026


def main():
    t = pd.read_parquet(DATA + "/train_meta.parquet",
                        columns=["ingest_lib", "inchikey14", "instrument_type", "adduct"])
    t["row"] = np.arange(len(t))
    ev = t[(t.ingest_lib == "enveda-180") & t.adduct.isin(TEST_ADDUCTS)]
    public = set(t.loc[~t.ingest_lib.str.startswith("enveda"), "inchikey14"])
    mols = ev.inchikey14.unique()
    rng = np.random.default_rng(SEED)
    in_pub = np.array([m in public for m in mols])
    c1 = rng.choice(mols[in_pub], min(N_PER_CLASS, in_pub.sum()), replace=False)
    rest = rng.permutation(mols[~in_pub])
    c2, c3 = rest[:N_PER_CLASS], rest[N_PER_CLASS:2 * N_PER_CLASS]
    cls = {**{m: 1 for m in c1}, **{m: 2 for m in c2}, **{m: 3 for m in c3}}
    q = ev[ev.inchikey14.isin(cls)].copy()
    q["cls"] = q.inchikey14.map(cls)
    # at most 16 spectra per molecule, like the test
    q = q.sample(frac=1, random_state=SEED).groupby("inchikey14").head(16)
    q[["row", "inchikey14", "cls", "adduct"]].to_parquet(DATA + "/split.parquet")
    print(q.groupby("cls").agg(mols=("inchikey14", "nunique"), spectra=("row", "size")))
    assert set(ADDUCTS) >= set(TEST_ADDUCTS)


if __name__ == "__main__":
    main()

"""Training matrix for the spectrum -> fingerprint model.

X: per spectrum, sqrt intensities binned at 1 Da for fragments (0..MZ_BINS) and
for neutral losses from the precursor (0..NL_BINS), plus a one-hot adduct block.
Y: Morgan radius-2 2048-bit fingerprint of the structure (packed bits).
Every molecule of the validation split is held out. At most PER_STRUCT spectra
per structure, preferring timsTOF.

    python fpnet_data.py      # writes $CASMI_DATA/fpnet_{X,Y,meta}
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA

MZ_BINS, NL_BINS = 1000, 500
ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
N_FEAT = MZ_BINS + NL_BINS + len(ADDUCT_LIST) + 1   # +1: other adduct
PER_STRUCT = 4
FP_BITS = 2048


def featurize(mz, it, prec, adduct):
    """mz/it: 1-D arrays of one spectrum. Returns a float32 vector of N_FEAT."""
    v = np.zeros(N_FEAT, np.float32)
    s = np.sqrt(np.clip(it, 0, None))
    b = np.floor(mz).astype(np.int64)
    k = (b >= 0) & (b < MZ_BINS)
    np.maximum.at(v, b[k], s[k])
    nl = np.floor(prec - mz).astype(np.int64)
    k = (nl >= 1) & (nl < NL_BINS)
    np.maximum.at(v, MZ_BINS + nl[k], s[k])
    a = ADDUCT_LIST.index(adduct) if adduct in ADDUCT_LIST else len(ADDUCT_LIST)
    v[MZ_BINS + NL_BINS + a] = 1.0
    return v


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = np.zeros((len(smiles), FP_BITS // 8), np.uint8)
    ok = np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        out[i] = np.packbits(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
        ok[i] = True
    return out, ok


def main():
    meta = pd.read_parquet(DATA + "/train_meta.parquet",
                           columns=["inchikey14", "normalized_smiles", "precursor_mz", "adduct", "instrument_type"])
    meta["row"] = np.arange(len(meta))
    held = set(pd.read_parquet(DATA + "/split.parquet").inchikey14)
    m = meta[~meta.inchikey14.isin(held)].copy()
    m["pri"] = (m.instrument_type == "timsTOF").astype(int)
    m = m.sample(frac=1, random_state=0).sort_values("pri", ascending=False, kind="stable")
    m = m.groupby("inchikey14").head(PER_STRUCT).sort_values("row")
    print("spectra", len(m), "structures", m.inchikey14.nunique())

    S = m.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles"]]
    fp, ok = fingerprints(S.normalized_smiles.values)
    fp_by = dict(zip(S.inchikey14[ok], fp[ok]))
    m = m[m.inchikey14.isin(fp_by)]
    rows = m.row.values
    X = np.zeros((len(m), N_FEAT), np.float16)
    f = pq.ParquetFile(DATA + "/train.parquet")
    start, j = 0, 0
    prec, add = m.precursor_mz.values, m.adduct.values
    for g in range(f.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(rows, [start, start + n])
        if hi > lo:
            t = f.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(rows[lo:hi] - start)
            a, b = t.column(0).combine_chunks(), t.column(1).combine_chunks()
            for i, (x, y) in enumerate(zip(a, b)):
                X[lo + i] = featurize(x.values.to_numpy(zero_copy_only=False),
                                      y.values.to_numpy(zero_copy_only=False), prec[lo + i], add[lo + i])
            j = hi
        start += n
        print("row group", g, j, flush=True)
    Y = np.stack([fp_by[k] for k in m.inchikey14.values])
    np.save(DATA + "/fpnet_X.npy", X)
    np.save(DATA + "/fpnet_Y.npy", Y)
    m[["row", "inchikey14"]].to_parquet(DATA + "/fpnet_meta.parquet")
    print("saved", X.shape, Y.shape)


if __name__ == "__main__":
    main()

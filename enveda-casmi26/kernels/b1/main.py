"""CASMI26 B1: mass-window candidates (train structures + COCONUT), ranked by
direct library similarity and analog propagation. No internet; wheels come
from the casmi26-offline-wheels dataset.

Locally: CASMI_COMP=<folder with train/test parquet> CASMI_COCO=<folder with
coco_meta.pkl, coco_mass.npy> python main.py
"""
import glob
import os
import pickle
import re
import subprocess
import sys
import time

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def first(pattern, default=None):
    hits = glob.glob(pattern, recursive=True)
    return os.path.dirname(hits[0]) if hits else default


COMP = os.environ.get("CASMI_COMP") or first("/kaggle/input/**/train.parquet")
COCO = os.environ.get("CASMI_COCO") or first("/kaggle/input/**/coco_meta.pkl")
WHEELS = first("/kaggle/input/**/rdkit-*.whl")
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
                    *glob.glob(WHEELS + "/*.whl")], check=True)
log("comp", COMP, "coco", COCO, "wheels", WHEELS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from ms_entropy import FlashEntropySearch, calculate_entropy_similarity, clean_spectrum  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PPM, PPM_WIDE = 10.0, 30.0
TOL_DA = 0.02
N_ANALOG = 100
POW = 3.0
MAX_PEAKS = 64
LIB_GATE = float(os.environ.get("CASMI_LIB_GATE", "0.95"))
W_ANALOG = 0.9

PROTON = 1.007276467
ADDUCTS = {
    "[M+H]+": (1, PROTON), "[M+NH4]+": (1, 18.033823), "[M-H2O+H]+": (1, PROTON - 18.010565),
    "[M-2H2O+H]+": (1, PROTON - 2 * 18.010565), "[M+Na]+": (1, 22.989218), "[M+K]+": (1, 38.963158),
    "[M-H]-": (1, -PROTON), "[M-H2O-H]-": (1, -PROTON - 18.010565),
    "[M+CH2O2-H]-": (1, 44.998201), "[M+Cl]-": (1, 34.969402),
    "[2M+H]+": (2, PROTON), "[2M+Na]+": (2, 22.989218), "[2M-H]-": (2, -PROTON),
}
MONO = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268,
        "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325, "B": 11.0093054,
        "Se": 79.9165213, "Na": 22.9897692809, "K": 38.96370668}
_FORM = re.compile(r"([A-Z][a-z]?)(\d*)")


def neutral_mass(mz, adduct):
    a = ADDUCTS.get(adduct)
    return np.nan if a is None else (mz - a[1]) / a[0]


def formula_mass(f):
    if not isinstance(f, str) or any(c in f for c in "+-"):
        return np.nan
    m = 0.0
    for el, n in _FORM.findall(f):
        if el not in MONO:
            return np.nan
        m += MONO[el] * (int(n) if n else 1)
    return m


def peaks(mz, it):
    p = np.column_stack([np.asarray(mz, np.float32), np.asarray(it, np.float32)])
    return clean_spectrum(p, min_ms2_difference_in_da=2 * TOL_DA)


def load_peaks(path, rows):
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    f = pq.ParquetFile(path)
    out, start = {}, 0
    for g in range(f.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(rows, [start, start + n])
        if hi > lo:
            t = f.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(rows[lo:hi] - start)
            a, b = t.column(0).combine_chunks(), t.column(1).combine_chunks()
            for r, x, y in zip(rows[lo:hi], a, b):
                out[int(r)] = peaks(x.values.to_numpy(zero_copy_only=False), y.values.to_numpy(zero_copy_only=False))
            del t
        start += n
    return out


_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return _gen.GetFingerprint(m) if m is not None else None


def main():
    meta = pd.read_parquet(COMP + "/train.parquet",
                           columns=["inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz",
                                    "adduct", "instrument_type", "ionization_mode", "num_peaks"])
    test = pd.read_parquet(COMP + "/test.parquet")
    log("train", len(meta), "test", len(test), test.molecule_id.nunique(), "molecules")

    S = meta.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles", "molecular_formula"]].copy()
    S["fM"] = S.molecular_formula.map(formula_mass)
    S = S.dropna(subset=["fM"]).rename(columns={"normalized_smiles": "smiles"})[["inchikey14", "smiles", "fM"]]
    if COCO:
        cm = pickle.load(open(COCO + "/coco_meta.pkl", "rb"))
        coco = pd.DataFrame({"inchikey14": cm["keys"], "smiles": cm["smiles"],
                             "fM": np.load(COCO + "/coco_mass.npy")})
        S = pd.concat([S, coco[~coco.inchikey14.isin(set(S.inchikey14))]], ignore_index=True)
    S = S.sort_values("fM").reset_index(drop=True)
    pm, pk = S.fM.values, S.inchikey14.values
    smi = dict(zip(S.inchikey14, S.smiles))
    log("pool", len(S))

    test["M"] = [neutral_mass(m, a) for m, a in zip(test.precursor_mz, test.adduct)]
    tm = test.groupby("molecule_id").agg(M=("M", "median"))
    cand = {}
    for mid, r in tm.iterrows():
        c = []
        if np.isfinite(r.M):
            for ppm in (PPM, PPM_WIDE):
                lo, hi = np.searchsorted(pm, [r.M * (1 - ppm * 1e-6), r.M * (1 + ppm * 1e-6)])
                c = list(pk[lo:hi])
                if c:
                    break
        cand[mid] = c
    need = set(x for v in cand.values() for x in v)
    log("candidates/molecule median", np.median([len(v) for v in cand.values()]))

    meta["row"] = np.arange(len(meta))
    lib_rows = meta.row.values[meta.inchikey14.isin(need).values]
    m2 = meta.copy()
    m2["pri"] = (m2.instrument_type == "timsTOF").astype(int) * 2 + m2.adduct.isin(["[M+H]+", "[M-H]-"]).astype(int)
    reps = m2.sort_values(["pri", "num_peaks"], ascending=False).drop_duplicates(["inchikey14", "ionization_mode"])
    del m2
    spec = load_peaks(COMP + "/train.parquet", np.union1d(lib_rows, reps.row.values))
    log("library spectra", len(spec))
    lib_by_ik = {}
    for r, ik in zip(lib_rows, meta.inchikey14.values[lib_rows]):
        lib_by_ik.setdefault(ik, []).append(spec[r])

    engines = {}
    for mode, g in reps.groupby("ionization_mode"):
        lib = [{"precursor_mz": float(p), "peaks": spec[r], "ik": ik}
               for r, p, ik in zip(g.row, g.precursor_mz, g.inchikey14) if len(spec[r])]
        e = FlashEntropySearch(max_ms2_tolerance_in_da=TOL_DA)
        lib = e.build_index(lib, max_indexed_mz=1500.0, precursor_ions_removal_da=1.6, noise_threshold=0.0,
                            min_ms2_difference_in_da=2 * TOL_DA, max_peak_num=MAX_PEAKS)
        engines[mode] = (e, [x["ik"] for x in lib])
    log("indexes", {k: len(v[1]) for k, v in engines.items()})

    fps = {}

    def getfp(ik):
        if ik not in fps:
            fps[ik] = fp(smi.get(ik))
        return fps[ik]

    out = []
    for mid, g in test.groupby("molecule_id"):
        cands = cand[mid]
        lib = dict.fromkeys(cands, 0.0)
        ana = dict.fromkeys(cands, 0.0)
        cfp = [getfp(c) for c in cands]
        okc = [i for i, f in enumerate(cfp) if f is not None]
        for _, s in g.iterrows():
            qs = peaks(s.ms2_mzs, s.ms2_normalized_intensities)
            if not len(qs):
                continue
            for c in cands:
                for sp in lib_by_ik.get(c, []):
                    v = calculate_entropy_similarity(qs, sp, ms2_tolerance_in_da=TOL_DA, clean_spectra=False)
                    if v > lib[c]:
                        lib[c] = v
            eng = engines.get(s.ionization_mode)
            if eng is None or not okc:
                continue
            e, iks = eng
            res = e.search(precursor_mz=float(s.precursor_mz), peaks=qs, method="hybrid",
                           ms2_tolerance_in_da=TOL_DA, noise_threshold=0.0, max_peak_num=MAX_PEAKS)["hybrid_search"]
            for j in np.argsort(-res)[:N_ANALOG]:
                w = float(res[j])
                if w <= 0:
                    break
                afp = getfp(iks[j])
                if afp is None:
                    continue
                tan = DataStructs.BulkTanimotoSimilarity(afp, [cfp[i] for i in okc])
                w = w ** POW
                for i, tv in zip(okc, tan):
                    if w * tv > ana[cands[i]]:
                        ana[cands[i]] = w * tv
        # Local split (150/class): analog alone c1 0.924 c2 0.799; a direct library
        # hit only helps when it is near-identical, so it is gated. On natural products
        # (class 4 of analog.py, the LB-like c2) gate 0.8 -> 0.526, 0.95 -> 0.570, c1 unchanged.
        score = {c: (lib[c] + 1.0 if lib[c] >= LIB_GATE else 0.0) + ana[c] for c in cands}
        ranked = sorted(cands, key=lambda c: -score[c])[:25]
        smiles = [smi[c] for c in ranked if isinstance(smi.get(c), str)]
        out.append((mid, ";".join(smiles) if smiles else "CCO"))
    sub = pd.DataFrame(out, columns=["molecule_id", "smiles"])
    sub.to_csv("submission.csv", index=False)
    log("wrote", len(sub), "rows")


if __name__ == "__main__":
    main()

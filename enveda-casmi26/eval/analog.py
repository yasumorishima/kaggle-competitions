"""B1: B0 plus analog propagation for candidates that have no library spectrum.

Library for analogs: one representative spectrum per (structure, polarity),
preferring timsTOF and the plain protonated/deprotonated adduct, held-out
molecules removed exactly as in B0. Every query spectrum runs a FlashEntropy
hybrid search (shifted matches allowed) against the reps of its polarity; the
top N_ANALOG hits are the analogs. A candidate scores

    analog = max over analogs of sim ** POW * tanimoto(candidate, analog)

and the final score is max(lib, W_ANALOG * analog), lib being B0's direct
library similarity.

    python analog.py [n_per_class]
"""
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import FlashEntropySearch, calculate_entropy_similarity
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from baseline_lib import PPM, TOL_DA, peaks
from common import DATA, load_peaks, mrr25, neutral_mass

RDLogger.DisableLog("rdApp.*")
N_ANALOG = 100
POW = 3.0
W_ANALOG = 0.9
MAX_PEAKS = 64
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return _gen.GetFingerprint(m) if m is not None else None


def pick_reps(meta, keep_rows):
    m = meta.iloc[keep_rows].copy()
    m["row"] = keep_rows
    m["pri"] = ((m.instrument_type == "timsTOF").astype(int) * 2
                + m.adduct.isin(["[M+H]+", "[M-H]-"]).astype(int))
    m = m.sort_values(["pri", "num_peaks"], ascending=False)
    return m.drop_duplicates(["inchikey14", "ionization_mode"])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time()
    meta = pd.read_parquet(DATA + "/train_meta.parquet",
                           columns=["ingest_lib", "inchikey14", "precursor_mz", "adduct",
                                    "instrument_type", "ionization_mode"])
    meta["num_peaks"] = pq.read_table(DATA + "/train.parquet", columns=["num_peaks"]).column(0).to_numpy()
    split = pd.read_parquet(DATA + "/split.parquet")
    split = split[split.inchikey14.isin(
        split.drop_duplicates("inchikey14").groupby("cls").head(n).inchikey14)]
    S = pd.read_parquet(DATA + "/structures.parquet").dropna(subset=["fM"])
    smi = dict(zip(S.inchikey14, S.normalized_smiles))
    c3 = set(split.loc[split.cls == 3, "inchikey14"])
    pool = S[~S.inchikey14.isin(c3)].sort_values("fM")
    pm, pk = pool.fM.values, pool.inchikey14.values

    held = set(split.inchikey14)
    drop = meta.inchikey14.isin(held) & (meta.ingest_lib == "enveda-180")
    keep = np.flatnonzero(~drop.values)
    reps = pick_reps(meta, keep)
    print(f"reps {len(reps)} ({time.time()-t0:.0f}s)")

    q = split.merge(meta[["precursor_mz", "ionization_mode"]], left_on="row", right_index=True)
    q["M"] = [neutral_mass(m, a) for m, a in zip(q.precursor_mz, q.adduct)]
    qm = q.groupby("inchikey14").agg(M=("M", "median"), cls=("cls", "first"), rows=("row", list))
    cand = {}
    for ik, r in qm.iterrows():
        lo, hi = np.searchsorted(pm, [r.M * (1 - PPM * 1e-6), r.M * (1 + PPM * 1e-6)])
        cand[ik] = list(pk[lo:hi])
    need = set(c for v in cand.values() for c in v)
    lib_rows = keep[meta.inchikey14.values[keep].astype(object) != None]  # noqa: E711
    lib_rows = lib_rows[np.isin(meta.inchikey14.values[lib_rows], list(need))]
    rows_needed = np.union1d(np.union1d(lib_rows, reps.row.values), np.concatenate(qm.rows.values))
    spec = load_peaks(rows_needed, peaks)
    lib_by_ik = {}
    for r, ik in zip(lib_rows, meta.inchikey14.values[lib_rows]):
        lib_by_ik.setdefault(ik, []).append(spec[r])
    print(f"spectra {len(spec)} ({time.time()-t0:.0f}s)")

    engines = {}
    for mode, g in reps.groupby("ionization_mode"):
        lib = [{"precursor_mz": float(p), "peaks": spec[r], "ik": ik}
               for r, p, ik in zip(g.row, g.precursor_mz, g.inchikey14) if len(spec[r])]
        e = FlashEntropySearch(max_ms2_tolerance_in_da=TOL_DA)
        lib = e.build_index(lib, max_indexed_mz=1500.0, precursor_ions_removal_da=1.6,
                            noise_threshold=0.0, min_ms2_difference_in_da=2 * TOL_DA, max_peak_num=MAX_PEAKS)
        engines[mode] = (e, [x["ik"] for x in lib])
    print(f"indexes built ({time.time()-t0:.0f}s)")

    fps = {}

    def getfp(ik):
        if ik not in fps:
            fps[ik] = fp(smi.get(ik))
        return fps[ik]

    rec, dump = [], []
    for ik, r in qm.iterrows():
        cands = cand[ik]
        lib = {c: 0.0 for c in cands}
        ana = {c: 0.0 for c in cands}
        cfp = [getfp(c) for c in cands]
        okc = [i for i, f in enumerate(cfp) if f is not None]
        hits = []
        for row in r.rows:
            qs = spec[row]
            if not len(qs):
                continue
            for c in cands:
                for l in lib_by_ik.get(c, []):
                    s = calculate_entropy_similarity(qs, l, ms2_tolerance_in_da=TOL_DA, clean_spectra=False)
                    lib[c] = max(lib[c], s)
            e, iks = engines[meta.ionization_mode.values[row]]
            res = e.search(precursor_mz=float(meta.precursor_mz.values[row]), peaks=qs, method="hybrid",
                           ms2_tolerance_in_da=TOL_DA, noise_threshold=0.0, max_peak_num=MAX_PEAKS)["hybrid_search"]
            top = np.argsort(-res)[:N_ANALOG]
            hits.append([(float(res[j]), iks[j]) for j in top if res[j] > 0])
            for j in top:
                s = float(res[j])
                if s <= 0:
                    break
                afp = getfp(iks[j])
                if afp is None or not okc:
                    continue
                tan = DataStructs.BulkTanimotoSimilarity(afp, [cfp[i] for i in okc])
                w = s ** POW
                for i, tv in zip(okc, tan):
                    v = w * tv
                    if v > ana[cands[i]]:
                        ana[cands[i]] = v
        dump.append((ik, r.cls, cands, [lib[c] for c in cands], [ana[c] for c in cands], hits))
        score = {c: max(lib[c], W_ANALOG * ana[c]) for c in cands}
        for name, sc in (("lib", lib), ("analog", ana), ("both", score)):
            ranked = sorted(cands, key=lambda c: -sc[c])
            rec.append((name, r.cls, mrr25(ranked, ik)))
    import pickle
    pickle.dump(dump, open(DATA + f"/scores_analog_{n}.pkl", "wb"))
    res = pd.DataFrame(rec, columns=["chan", "cls", "mrr"])
    print(res.pivot_table(index="cls", columns="chan", values="mrr", aggfunc="mean").round(3))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

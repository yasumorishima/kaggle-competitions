"""ae simulation (ours, 11-04): how often the analog-edit gate fires and how often its product is the answer.

N enveda-180 structures (timsTOF, test adducts) are hidden from the ae library (all their spectra); their own
enveda-180 spectra (<= 4) are the queries. g = gate rate (modified cosine >= tau with a one-modification analog),
h = picked product == answer (inchikey14), o = answer among the products (oracle).
No learned model is used, so nothing has seen these molecules except the library itself, which hides them.
Usage: python sim.py TRAIN_PARQUET LIB_DIR N [tau]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels", "e3"))
import ours_ae   # noqa: E402
import ours_frag   # noqa: E402


def main(src, libdir, n, tau=0.85):
    st = pd.read_parquet(os.path.join(libdir, "ae_struct.parquet"))
    gr = pd.read_parquet(os.path.join(libdir, "ae_groups.parquet")).group.tolist()
    lib = ours_ae.Library(os.path.join(libdir, "ae_lib.npz"), st.smiles.tolist(), gr)
    t = pq.read_table(src, columns=["inchikey14", "ingest_lib", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs",
                                    "ms2_normalized_intensities"], filters=[("ingest_lib", "=", "enveda-180")]).to_pandas()
    t = t[t.adduct.isin(ours_ae_adducts())]
    rng = np.random.default_rng(0)
    iks = rng.choice(t.inchikey14.unique(), n, replace=False)
    sidx = {k: i for i, k in enumerate(st.inchikey14)}
    lib.mask = np.zeros(len(st), bool)
    lib.mask[[sidx[k] for k in iks if k in sidx]] = True
    g = h = o = 0
    kinds = {}
    t0 = time.time()
    for j, ik in enumerate(iks):
        rows = t[t.inchikey14 == ik].head(4)
        specs = ours_ae.query_specs(zip(rows.ionization_mode, rows.adduct, rows.precursor_mz, rows.ms2_mzs, rows.ms2_normalized_intensities))
        fr = [(np.c_[np.asarray(a, float), np.asarray(b, float)], float(p), ours_frag.polarity_of(ad, md))
              for md, ad, p, a, b in zip(rows.ionization_mode, rows.adduct, rows.precursor_mz, rows.ms2_mzs, rows.ms2_normalized_intensities)]
        smi, info = (ours_ae.propose(lib, specs, fr, ours_frag, tau=tau) if MODE == 'one' else
                     ours_ae.propose_v(lib, specs, fr, ours_frag, tau=tau, mode=MODE))
        if info.get("gate"):
            REC.append((info.get('votes', 1), info.get('modcos'), info.get('n_analogs', 1), info.get('n_new', 0), int((info.get('pick') or '') == ik), int(ik in (info.get('keys') or []))))
            g += 1
            kinds[info["kind"] + info["direction"][0]] = kinds.get(info["kind"] + info["direction"][0], 0) + 1
            if ik in (info.get("keys") or []):
                o += 1
            if smi is not None and (info.get("pick") or (info.get("keys") or [None])[0]) == ik:
                h += 1
        if (j + 1) % 50 == 0:
            print(f"{j + 1} molecules  gate {g}  oracle {o}  hit {h}  {time.time() - t0:.0f}s", flush=True)
    print(f"N {n} tau {tau}: gate rate g {g / n:.3f} ({g})  oracle among products {o} ({o / max(g, 1):.3f} of gated)  "
          f"hit h {h} ({h / max(g, 1):.3f} of gated, {h / n:.3f} of all)  sec/mol {(time.time() - t0) / n:.2f}")
    if os.environ.get('AE_REC'):
        pd.DataFrame(REC, columns=['votes', 'modcos', 'n_analogs', 'n_new', 'hit', 'oracle']).to_csv(os.environ['AE_REC'], index=False)
    print("kinds of gated analogs:", dict(sorted(kinds.items(), key=lambda x: -x[1])))
    # expected MRR change for the gated: +0.5 per hit (rank 2), about -0.025 per gated molecule (answers in 2-25 move down)
    print(f"dMRR per molecule ~ 0.5*{h / n:.3f} - 0.025*{g / n:.3f} = {0.5 * h / n - 0.025 * g / n:+.4f}  "
          f"(x130 public = {(0.5 * h / n - 0.025 * g / n) * 130:+.2f} molecules if every molecule were class-3-like)")


def ours_ae_adducts():
    return ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]


MODE = os.environ.get('AE_MODE', 'one')
REC = []

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.85)

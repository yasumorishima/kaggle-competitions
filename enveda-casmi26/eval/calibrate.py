"""Make the local c2 as hard as the hidden one, then compare channels there.

The random enveda-180 c2 is easy: close analogs of the query sit in the library.
The LB says the real c2 is harder (b1 = 0.275 -> c2 ~ 0.47 at the public class
shares). This recomputes the analog score from the dumped hits with every analog
whose Tanimoto to the TRUE structure is >= T removed, for several T, and adds the
fingerprint channel (fpnet.pt) at several weights.

    python calibrate.py
"""
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs

from analog import POW, fp
from common import DATA, load_peaks, mrr25
from fpnet_data import featurize, fingerprints
from fpnet_train import net

THRESH = [1.01, 0.9, 0.8, 0.7, 0.6, 0.5]
W_FP = [0.0, 0.1, 0.3, 1.0, 3.0]


def main():
    dump = pickle.load(open(DATA + "/scores_analog_150.pkl", "rb"))
    S = pd.read_parquet(DATA + "/structures.parquet")
    smi = dict(zip(S.inchikey14, S.normalized_smiles))
    split = pd.read_parquet(DATA + "/split.parquet")
    meta = pd.read_parquet(DATA + "/train_meta.parquet", columns=["precursor_mz", "adduct"])
    q = split[split.inchikey14.isin(set(d[0] for d in dump))]
    spec = load_peaks(q.row.values, lambda a, b: (a, b))
    model = net()
    model.load_state_dict(torch.load(DATA + "/fpnet.pt"))
    model.eval()
    fps = {}

    def gfp(ik):
        if ik not in fps:
            fps[ik] = fp(smi.get(ik))
        return fps[ik]

    rows = []
    for ik, cls, cands, lib, _, hits in dump:
        if cls == 3:
            continue
        g = q[q.inchikey14 == ik]
        with torch.no_grad():
            x = np.stack([featurize(*spec[r], meta.precursor_mz.values[r], meta.adduct.values[r]) for r in g.row])
            p = torch.sigmoid(model(torch.from_numpy(x))).mean(0).clamp(1e-4, 1 - 1e-4).numpy()
        cfp_bits, ok = fingerprints([smi.get(c) for c in cands])
        bits = np.unpackbits(cfp_bits, axis=1).astype(np.float32)
        ll = bits @ np.log(p) + (1 - bits) @ np.log(1 - p)
        ll[~ok] = -1e9
        fpz = np.exp((ll - ll.max()) / 50.0)
        truth = gfp(ik)
        cf = [gfp(c) for c in cands]
        okc = [i for i, f in enumerate(cf) if f is not None]
        # per analog: tanimoto to truth, to every candidate
        allh = {}
        for spec_hits in hits:
            for s, a in spec_hits:
                allh[a] = max(allh.get(a, 0.0), s)
        alist = [a for a in allh if gfp(a) is not None]
        t_truth = np.array(DataStructs.BulkTanimotoSimilarity(truth, [gfp(a) for a in alist])) if truth and alist else np.zeros(len(alist))
        tan = np.zeros((len(alist), len(cands)))
        for k, a in enumerate(alist):
            if okc:
                tan[k, okc] = DataStructs.BulkTanimotoSimilarity(gfp(a), [cf[i] for i in okc])
        w = np.array([allh[a] for a in alist]) ** POW
        lib = np.asarray(lib)
        gate = np.where(lib >= 0.8, lib + 1.0, 0.0)
        for T in THRESH:
            keep = t_truth < T
            # a library hit on the truth itself would also be gone at this hardness
            ana = (w[keep, None] * tan[keep]).max(0) if keep.any() else np.zeros(len(cands))
            for wf in W_FP:
                sc = (gate if T > 1 else 0) + ana + wf * fpz
                ranked = [cands[i] for i in np.argsort(-sc, kind="stable")]
                rows.append((cls, T, wf, mrr25(ranked, ik)))
    r = pd.DataFrame(rows, columns=["cls", "T", "w_fp", "mrr"])
    print(r[r.cls == 2].pivot_table(index="T", columns="w_fp", values="mrr", aggfunc="mean").round(3))
    print(r[r.cls == 1].pivot_table(index="T", columns="w_fp", values="mrr", aggfunc="mean").round(3))
    print("fp alone c2:", round(np.mean([m for c, T, wf, m in rows if c == 2 and T == 0.5 and wf == 3.0]), 3))


if __name__ == "__main__":
    main()

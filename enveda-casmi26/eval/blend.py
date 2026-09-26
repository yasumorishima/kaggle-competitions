"""Blend the channels of an analog.py dump with the fingerprint model, by class.

    python blend.py scores_analog_150_coco_np.pkl

score = gate(lib) + analog + w_fp * exp((ll - max ll) / 50), gate = lib + 1 when
lib >= G else 0. Prints MRR@25 per class for a grid of G and w_fp.
"""
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from common import DATA, load_peaks, mrr25
from fpnet_data import featurize, fingerprints
from fpnet_train import net

GATES = [0.8, 0.95, 9.0]
W_FP = [0.0, 0.05, 0.1, 0.2, 0.3]


def main():
    dump = pickle.load(open(DATA + "/" + sys.argv[1], "rb"))
    S = pd.read_parquet(DATA + "/structures.parquet")
    coco = pd.read_parquet(DATA + "/coconut.parquet")
    smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
    meta = pd.read_parquet(DATA + "/train_meta.parquet", columns=["ingest_lib", "inchikey14", "precursor_mz", "adduct"])
    split = pd.read_parquet(DATA + "/split.parquet")
    e = meta[meta.ingest_lib == "enveda-np-examples"]
    e = e.assign(row=e.index.values).sample(frac=1, random_state=0).groupby("inchikey14").head(16)
    q = pd.concat([split[["row", "inchikey14"]], e[["row", "inchikey14"]]])
    q = q[q.inchikey14.isin(set(d[0] for d in dump))]
    spec = load_peaks(q.row.values, lambda a, b: (a, b))
    model = net()
    model.load_state_dict(torch.load(DATA + "/fpnet.pt"))
    model.eval()
    rows = []
    for ik, cls, cands, lib, ana, *_ in dump:
        if cls == 3:
            continue
        g = q[q.inchikey14 == ik]
        with torch.no_grad():
            x = np.stack([featurize(*spec[r], meta.precursor_mz.values[r], meta.adduct.values[r]) for r in g.row])
            p = torch.sigmoid(model(torch.from_numpy(x))).mean(0).clamp(1e-4, 1 - 1e-4).numpy()
        bits, ok = fingerprints([smi.get(c) for c in cands])
        bits = np.unpackbits(bits, axis=1).astype(np.float32)
        ll = bits @ np.log(p) + (1 - bits) @ np.log(1 - p)
        ll[~ok] = -1e9
        fpz = np.exp((ll - ll.max()) / 50.0)
        lib, ana = np.asarray(lib), np.asarray(ana)
        rows.append((cls, "fp only", 0, mrr25([cands[i] for i in np.argsort(-ll, kind="stable")], ik)))
        for G in GATES:
            for wf in W_FP:
                sc = np.where(lib >= G, lib + 1.0, 0.0) + ana + wf * fpz
                rows.append((cls, G, wf, mrr25([cands[i] for i in np.argsort(-sc, kind="stable")], ik)))
    r = pd.DataFrame(rows, columns=["cls", "G", "w_fp", "mrr"])
    for c in sorted(r.cls.unique()):
        print(f"class {c}")
        print(r[r.cls == c].pivot_table(index="G", columns="w_fp", values="mrr", aggfunc="mean").round(3))


if __name__ == "__main__":
    main()

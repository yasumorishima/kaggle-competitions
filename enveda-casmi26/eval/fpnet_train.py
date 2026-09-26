"""Train the spectrum -> fingerprint MLP and measure it as a ranking channel.

    python fpnet_train.py [epochs]

Writes $CASMI_DATA/fpnet.pt. Then, on the validation molecules saved by
analog.py (scores_analog_150.pkl), ranks candidates by the fingerprint
log-likelihood alone and added to the analog score, and prints MRR@25 by class.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import DATA, load_peaks, mrr25, neutral_mass  # noqa: F401
from fpnet_data import FP_BITS, N_FEAT, featurize, fingerprints

torch.set_num_threads(4)
HID = 1024


def net():
    return nn.Sequential(nn.Linear(N_FEAT, HID), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(HID, HID), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(HID, FP_BITS))


def train(epochs):
    X = np.load(DATA + "/fpnet_X.npy", mmap_mode="r")
    Y = np.load(DATA + "/fpnet_Y.npy", mmap_mode="r")
    n = len(X)
    rng = np.random.default_rng(0)
    model = net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    steps = epochs * (n // 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    k = 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        model.train()
        tot = 0.0
        for i in range(0, n - 511, 512):
            idx = np.sort(perm[i:i + 512])
            xb = torch.from_numpy(np.asarray(X[idx], np.float32))
            yb = torch.from_numpy(np.unpackbits(np.asarray(Y[idx]), axis=1).astype(np.float32))
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            k += 1
            if k % 500 == 0:
                print(f"ep {ep} step {k}/{steps} loss {tot / (i // 512 + 1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), DATA + "/fpnet.pt")
    return model


def evaluate(model):
    dump = pickle.load(open(DATA + "/scores_analog_150.pkl", "rb"))
    split = pd.read_parquet(DATA + "/split.parquet")
    meta = pd.read_parquet(DATA + "/train_meta.parquet", columns=["precursor_mz", "adduct"])
    S = pd.read_parquet(DATA + "/structures.parquet")
    coco = pd.read_parquet(DATA + "/coconut.parquet")
    smi = {**dict(zip(coco.inchikey14, coco.smiles)), **dict(zip(S.inchikey14, S.normalized_smiles))}
    iks = set(ik for d in dump for ik in [d[0]])
    q = split[split.inchikey14.isin(iks)]
    spec = load_peaks(q.row.values, lambda a, b: (a, b))
    model.eval()
    logp = {}
    with torch.no_grad():
        for ik, g in q.groupby("inchikey14"):
            x = np.stack([featurize(*spec[r], meta.precursor_mz.values[r], meta.adduct.values[r]) for r in g.row])
            p = torch.sigmoid(model(torch.from_numpy(x))).mean(0).clamp(1e-4, 1 - 1e-4).numpy()
            logp[ik] = (np.log(p), np.log(1 - p))
    rows = []
    for ik, cls, cands, lib, ana in dump:
        fp, ok = fingerprints([smi.get(c) for c in cands])
        bits = np.unpackbits(fp, axis=1).astype(np.float32)
        lp1, lp0 = logp[ik]
        ll = bits @ lp1 + (1 - bits) @ lp0
        ll[~ok] = -1e9
        z = (ll - ll.max()) / 50.0          # log-likelihood per 50 nats -> comparable scale
        ana = np.asarray(ana)
        lib = np.asarray(lib)
        for name, s in (("fp", ll), ("analog", ana), ("an+fp", ana + 0.3 * np.exp(z)),
                        ("an+fp.6", ana + 0.6 * np.exp(z)), ("an+fp1", ana + 1.0 * np.exp(z))):
            s = s + np.where(lib >= 0.8, lib + 1.0, 0.0)
            ranked = [cands[i] for i in np.argsort(-s, kind="stable")]
            rows.append((name, cls, mrr25(ranked, ik)))
    r = pd.DataFrame(rows, columns=["chan", "cls", "mrr"])
    print(r.pivot_table(index="cls", columns="chan", values="mrr", aggfunc="mean").round(3))


if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if ep > 0:
        m = train(ep)
    else:
        m = net()
        m.load_state_dict(torch.load(DATA + "/fpnet.pt"))
    evaluate(m)

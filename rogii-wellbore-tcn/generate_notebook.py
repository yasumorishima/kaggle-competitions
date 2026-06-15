"""
Generate the ROGII Wellbore TVT sequence-model (TCN) notebook.

Approach (from-scratch frontier lever, NOT a fork of the GBT clones):
  The toe TVT is a *long-range extrapolation* along the horizontal well. A flat
  per-row GBT cannot exploit the trajectory continuity; a dilated 1D-CNN (TCN)
  over the MD-ordered log can. We predict the per-row delta (TVT - last_known_tvt)
  as a sequence, masking the loss to the toe (where TVT_input is NaN).

Inputs available at BOTH train and test time (verified on the raw CSVs):
  horizontal_well.csv: [MD, X, Y, Z, GR, TVT_input]  (+ train-only TVT & formation
  tops, which are NOT used as inputs because test lacks them).
  -> 7 channels: MD/X/Y/Z relative to the heel/toe boundary, GR (interp + well
     z-score), known_value, known_mask.
  Target (train): (TVT - last_known_tvt) / 100.

SMOKE flag is baked in (Kaggle cannot inject env vars at push time). SMOKE=True
runs 6 train / 3 test wells, a tiny TCN, 3 epochs -> only proves the pipeline
(data -> delta -> submission format) end to end. FLIP to False, regenerate,
commit, push for the full run.

The generated notebook is written next to this file as the code_file named in
kernel-metadata.json. kaggle-push.yml runs `python generate_notebook.py` then
`kaggle kernels push`.
"""
import json
import os

# ---------------------------------------------------------------------------
# Cell sources. Keep each cell self-contained and ordered.
# ---------------------------------------------------------------------------

CELL_CONFIG = r'''
# === Config + torch GPU check (P100 sm_60 may be incompatible -> CPU fallback) ===
import os, glob, math, time
import numpy as np
import pandas as pd

SMOKE = True          # FLIP to False (regenerate + commit + push) for the full run
SEED = 42
N_SPLITS = 5
EPOCHS   = 3   if SMOKE else 40
CHANNELS = 32  if SMOKE else 96
N_BLOCKS = 3   if SMOKE else 8
LR = 1e-3
N_CH = 7
print(f"SMOKE={SMOKE} EPOCHS={EPOCHS} CHANNELS={CHANNELS} N_BLOCKS={N_BLOCKS}")

import torch
import torch.nn as nn
print("torch", torch.__version__)

DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        _t = torch.zeros(8, device="cuda")
        _ = float((_t + 1).sum().item())   # force a real kernel on the GPU
        DEVICE = "cuda"
        print("GPU OK:", torch.cuda.get_device_name(0))
    else:
        print("cuda not available")
except Exception as e:
    print("GPU unusable, falling back to CPU:", repr(e)[:160])
    DEVICE = "cpu"
print("DEVICE =", DEVICE)

torch.manual_seed(SEED)
np.random.seed(SEED)
'''

CELL_DATA_PATHS = r'''
# === Locate competition data (mounted under /kaggle/input/competitions/<slug>/) ===
ROOT = None
for base in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
             "/kaggle/input/rogii-wellbore-geology-prediction"]:
    if os.path.isdir(os.path.join(base, "train")) and os.path.isdir(os.path.join(base, "test")):
        ROOT = base
        break
if ROOT is None:
    for d, _, _ in os.walk("/kaggle/input"):
        if d.endswith(os.sep + "train") and os.path.isdir(d[:-6] + "test"):
            ROOT = d[:-6].rstrip(os.sep)
            break
assert ROOT is not None, "competition data root not found under /kaggle/input"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR  = os.path.join(ROOT, "test")
print("ROOT =", ROOT)

def list_wells(dirpath):
    fs = sorted(glob.glob(os.path.join(dirpath, "*__horizontal_well.csv")))
    return [(os.path.basename(f).replace("__horizontal_well.csv", ""), f) for f in fs]

train_wells = list_wells(TRAIN_DIR)
test_wells  = list_wells(TEST_DIR)
if SMOKE:
    train_wells = train_wells[:6]
    test_wells  = test_wells[:3]
print(f"train wells={len(train_wells)}  test wells={len(test_wells)}")
'''

CELL_BUILD = r'''
# === Per-well channel builder ===
# Rows are MD-ordered as given (MD is monotonic); row_index = position in the file
# (this is what the submission id f"{wid}_{row_index}" refers to).
def build_well(wid, path, is_train):
    df = pd.read_csv(path)
    n = len(df)
    md = df["MD"].values.astype(np.float64)
    x  = df["X"].values.astype(np.float64)
    y  = df["Y"].values.astype(np.float64)
    z  = df["Z"].values.astype(np.float64)
    ti = df["TVT_input"].values.astype(np.float64)
    known = ~np.isnan(ti)
    if known.sum() == 0:
        b = 0
        last = float(z[0])
    else:
        b = int(np.where(known)[0].max())   # heel/toe boundary (last known TVT_input)
        last = float(ti[b])

    gr = pd.to_numeric(df["GR"], errors="coerce").interpolate(limit_direction="both")
    gr = gr.fillna(gr.median()).values.astype(np.float64)
    grm = float(np.nanmean(gr)); grs = float(np.nanstd(gr)) + 1e-6

    ch = np.zeros((N_CH, n), dtype=np.float32)
    ch[0] = (md - md[b]) / 1000.0
    ch[1] = (x - x[b]) / 1000.0
    ch[2] = (y - y[b]) / 1000.0
    ch[3] = (z - z[b]) / 100.0
    ch[4] = (gr - grm) / grs
    ch[5] = np.where(known, (np.nan_to_num(ti) - last) / 100.0, 0.0)   # known delta value
    ch[6] = known.astype(np.float32)                                    # known mask

    toe = (~known).astype(np.float32)
    out = {"wid": wid, "ch": ch, "toe": toe, "last": last, "n": n,
           "row_idx": np.arange(n)}
    if is_train:
        tvt = df["TVT"].values.astype(np.float64)
        out["target"] = ((tvt - last) / 100.0).astype(np.float32)      # scaled delta
    return out

def load_set(wells, is_train):
    return [build_well(wid, path, is_train) for wid, path in wells]

train_data = load_set(train_wells, True)
print("loaded train wells:", len(train_data), "| example n rows:", train_data[0]["n"])
'''

CELL_MODEL = r'''
# === TCN: dilated residual 1D-CNN, length-preserving ===
class TCNBlock(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.ReLU()
    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.bn(self.c2(y))
        return self.act(x + y)

class TCN(nn.Module):
    def __init__(self, cin, c=32, nb=3):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.blocks = nn.ModuleList([TCNBlock(c, 2 ** i) for i in range(nb)])
        self.out = nn.Conv1d(c, 1, 1)
    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)

def huber(pred, tgt, delta=1.0):
    e = pred - tgt
    a = e.abs()
    return torch.where(a < delta, 0.5 * e * e, delta * (a - 0.5 * delta))
'''

CELL_TRAIN = r'''
# === GroupKFold(by well) training; loss masked to the toe ===
from sklearn.model_selection import GroupKFold

def train_one(tr):
    model = TCN(N_CH, CHANNELS, N_BLOCKS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train()
        order = list(range(len(tr)))
        np.random.shuffle(order)
        tot = 0.0
        for i in order:
            d = tr[i]
            x = torch.from_numpy(d["ch"]).unsqueeze(0).to(DEVICE)
            t = torch.from_numpy(d["target"]).view(1, 1, -1).to(DEVICE)
            m = torch.from_numpy(d["toe"]).view(1, 1, -1).to(DEVICE)
            opt.zero_grad()
            p = model(x)
            loss = (huber(p, t) * m).sum() / (m.sum() + 1e-6)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f"    ep{ep} train_loss={tot / max(1, len(tr)):.4f}")
    return model

groups = np.array([d["wid"] for d in train_data])
nsp = min(N_SPLITS, len(set(groups)))
gkf = GroupKFold(n_splits=nsp)
models, oof_rmse = [], []
for fi, (tri, vai) in enumerate(gkf.split(train_data, groups=groups)):
    tr = [train_data[i] for i in tri]
    va = [train_data[i] for i in vai]
    print(f"fold{fi}: train={len(tr)} valid={len(va)}")
    mdl = train_one(tr)
    models.append(mdl)
    mdl.eval()
    sq = []
    with torch.no_grad():
        for d in va:
            x = torch.from_numpy(d["ch"]).unsqueeze(0).to(DEVICE)
            p = mdl(x).cpu().numpy().reshape(-1) * 100.0
            toe = d["toe"].astype(bool)
            pred_tvt = d["last"] + p
            true_tvt = d["target"] * 100.0 + d["last"]
            if toe.sum() > 0:
                sq.append((pred_tvt[toe] - true_tvt[toe]) ** 2)
    if sq:
        rmse = float(np.sqrt(np.concatenate(sq).mean()))
        oof_rmse.append(rmse)
        print(f"fold{fi} toe RMSE = {rmse:.4f}")
    if SMOKE:
        break
if oof_rmse:
    print("CV toe RMSE mean =", float(np.mean(oof_rmse)))
'''

CELL_INFER = r'''
# === Inference (fold-averaged) + submission ===
test_data = load_set(test_wells, False)
rows = []
for d in test_data:
    x = torch.from_numpy(d["ch"]).unsqueeze(0).to(DEVICE)
    preds = []
    for mdl in models:
        mdl.eval()
        with torch.no_grad():
            preds.append(mdl(x).cpu().numpy().reshape(-1))
    p = np.mean(preds, axis=0) * 100.0
    tvt = d["last"] + p
    toe = d["toe"].astype(bool)
    for ri in np.where(toe)[0]:
        rows.append((f"{d['wid']}_{int(d['row_idx'][ri])}", float(tvt[ri])))

pred_df = pd.DataFrame(rows, columns=["id", "tvt"])
samp = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
sub = samp[["id"]].merge(pred_df, on="id", how="left")
fill = float(np.nanmean([d["last"] for d in test_data])) if test_data else 0.0
sub["tvt"] = sub["tvt"].fillna(fill)
sub.to_csv("submission.csv", index=False)
print("submission:", sub.shape, "| NaN:", int(sub["tvt"].isna().sum()),
      "| covered (non-fill):", int(pred_df.shape[0]))
print(sub.head())
'''

CELLS = [
    CELL_CONFIG,
    CELL_DATA_PATHS,
    CELL_BUILD,
    CELL_MODEL,
    CELL_TRAIN,
    CELL_INFER,
]


def make_code_cell(src):
    src = src.strip("\n") + "\n"
    lines = src.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def main():
    nb = {
        "cells": [make_code_cell(c) for c in CELLS],
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "rogii-tcn-seq.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {out} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()

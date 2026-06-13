"""Generate S6E6 NN-diversity notebook (.ipynb).

Lever: model-class diversity (standard playground top-solution move).
GBT champion = two-stage pseudo-label ensemble (LB 0.95944, STAGE2 OOF
weighted 0.95796). LB-verified closed levers: external SDSS17 data
(LB -0.00143), pseudo round 2 / stage3 blend (LB -0.00037).

This kernel mounts the baseline kernel npy artifacts
(oof_stage2 / test_probs_stage2 / y_train) via kernel_sources, trains
a categorical-embedding MLP on the same features (numeric + pairwise
diffs, per-fold quantile normalization, balanced CE loss), then
searches blend = a*STAGE2 + (1-a)*NN on OOF with per-class weight
re-tuning. The alpha grid includes 1.0 (= pure STAGE2) so the blend
pick can never lose to STAGE2 on OOF; prints SKIP_SUBMIT when pure
STAGE2 wins (submission would be identical to the submitted best).

SMOKE=True: 3 folds x 3 epochs plumbing check (artifact mount sanity,
training loop, blend + submission path). SMOKE=False: 5 folds x up to
40 epochs, early stop patience 6 on val macro-F1.
"""

import json

SMOKE = True

CODE = r'''
import os, time, numpy as np, pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight

SMOKE = __SMOKE__
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

IN = None
ART = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "sample_submission.csv" in files:
        IN = root
    if "oof_stage2.npy" in files and "test_probs_stage2.npy" in files:
        ART = root
if IN is None:
    raise SystemExit("train.csv not found under /kaggle/input")
if ART is None:
    raise SystemExit("baseline npy artifacts not mounted via kernel_sources")
print("data dir:", IN, "| artifact dir:", ART)

train = pd.read_csv(IN + "/train.csv")
test = pd.read_csv(IN + "/test.csv")
sub = pd.read_csv(IN + "/sample_submission.csv")

id_col = sub.columns[0]
target = sub.columns[1]
classes = sorted(train[target].astype(str).unique())
cls2idx = {c: i for i, c in enumerate(classes)}
idx2cls = {i: c for c, i in cls2idx.items()}
n_class = len(classes)

features = [c for c in train.columns if c not in (id_col, target)]
cat_cols = [c for c in features if str(train[c].dtype) in ("object", "category")]
num_cols = [c for c in features if c not in cat_cols]

for c in cat_cols:
    cats = pd.concat([train[c], test[c]], axis=0).astype("category").cat.categories
    train[c] = pd.Categorical(train[c], categories=cats).codes
    test[c] = pd.Categorical(test[c], categories=cats).codes

diff_cols = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        a, b = num_cols[i], num_cols[j]
        name = "d_" + a + "_" + b
        train[name] = train[a] - train[b]
        test[name] = test[a] - test[b]
        diff_cols.append(name)

NUMS = num_cols + diff_cols
CATS = cat_cols
y = train[target].astype(str).map(cls2idx).values
print("train:", train.shape, "| test:", test.shape,
      "| nums:", len(NUMS), "| cats:", len(CATS))

def tune_weights(oof, y):
    """Per-class probability weights via coordinate ascent on OOF macro-F1."""
    def mf1(w):
        return f1_score(y, (oof * w[None, :]).argmax(1), average="macro")
    best_w = np.ones(oof.shape[1])
    best_f1 = mf1(best_w)
    for it in range(3):
        improved = False
        for c in range(oof.shape[1]):
            for v in np.linspace(0.6, 1.6, 21):
                w = best_w.copy(); w[c] = v
                f = mf1(w)
                if f > best_f1 + 1e-6:
                    best_f1, best_w, improved = f, w, True
        if not improved:
            break
    return best_w, best_f1

y_art = np.load(ART + "/y_train.npy")
assert np.array_equal(y, y_art), "y mismatch vs baseline artifacts"
oof2 = np.load(ART + "/oof_stage2.npy")
pred2 = np.load(ART + "/test_probs_stage2.npy")
w2, f2 = tune_weights(oof2, y)
print("STAGE2 (loaded) | OOF weighted:", round(f2, 5), "| expect ~0.95796")
assert f2 > 0.9578, "stage2 artifact sanity failed"

cls_w = compute_class_weight("balanced", classes=np.arange(n_class), y=y)

N_SPLITS = 3 if SMOKE else 5
MAX_EPOCHS = 3 if SMOKE else 40
PATIENCE = 2 if SMOKE else 6
BS = 4096
cards = [int(max(train[c].max(), test[c].max())) + 1 for c in CATS]
print("cat cardinalities:", cards)

class MLP(nn.Module):
    def __init__(self, n_num, cards, n_class):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(card, min(16, (card + 1) // 2)) for card in cards])
        in_dim = n_num + sum(e.embedding_dim for e in self.embs)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, n_class))

    def forward(self, xn, xc):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        return self.net(torch.cat(parts, 1))

def predict(model, xn, xc):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xn), 65536):
            out.append(torch.softmax(model(xn[i:i + 65536], xc[i:i + 65536]), 1).cpu().numpy())
    return np.concatenate(out)

Xc_train = torch.tensor(train[CATS].values, dtype=torch.long, device=DEVICE)
Xc_test = torch.tensor(test[CATS].values, dtype=torch.long, device=DEVICE)

oof_nn = np.zeros((len(train), n_class))
pred_nn = np.zeros((len(test), n_class))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
t0 = time.time()
for fold, (tr, va) in enumerate(skf.split(train[NUMS], y)):
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=200000, random_state=SEED)
    tn_tr = torch.tensor(qt.fit_transform(train[NUMS].iloc[tr]).astype(np.float32), device=DEVICE)
    tn_va = torch.tensor(qt.transform(train[NUMS].iloc[va]).astype(np.float32), device=DEVICE)
    tn_te = torch.tensor(qt.transform(test[NUMS]).astype(np.float32), device=DEVICE)
    tc_tr = Xc_train[torch.from_numpy(tr).to(DEVICE)]
    tc_va = Xc_train[torch.from_numpy(va).to(DEVICE)]
    ty_tr = torch.tensor(y[tr], dtype=torch.long, device=DEVICE)
    yva = y[va]

    torch.manual_seed(SEED + fold)
    model = MLP(len(NUMS), cards, n_class).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(cls_w, dtype=torch.float32, device=DEVICE))
    best_f1, best_state, bad = -1.0, None, 0
    n_tr = len(tr)
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BS):
            idx = perm[i:i + BS]
            if len(idx) < 32:
                continue
            opt.zero_grad()
            loss = lossf(model(tn_tr[idx], tc_tr[idx]), ty_tr[idx])
            loss.backward()
            opt.step()
        sched.step()
        f = f1_score(yva, predict(model, tn_va, tc_va).argmax(1), average="macro")
        if f > best_f1 + 1e-5:
            best_f1, bad = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    oof_nn[va] = predict(model, tn_va, tc_va)
    pred_nn += predict(model, tn_te, tc_te)
    print("NN fold", fold, "| best val macro-F1:", round(best_f1, 5),
          "| epochs:", ep + 1, "| elapsed:", round(time.time() - t0), "s")
pred_nn /= N_SPLITS

w_nn, f_nn = tune_weights(oof_nn, y)
print("NN | OOF macro-F1 argmax:", round(f1_score(y, oof_nn.argmax(1), average="macro"), 5),
      "| weighted:", round(f_nn, 5),
      "| weights:", dict(zip(classes, np.round(w_nn, 3))))

np.save("oof_nn.npy", oof_nn)
np.save("test_probs_nn.npy", pred_nn)
print("saved npy artifacts: oof_nn, test_probs_nn")

best = ("STAGE2", 1.0, w2, f2)
for a in np.arange(0.5, 1.0001, 0.05):
    blended = a * oof2 + (1 - a) * oof_nn
    w, f = tune_weights(blended, y)
    print("blend a=" + str(round(a, 2)), "| OOF weighted:", round(f, 5))
    if f > best[3] + 1e-6:
        best = ("BLEND", float(a), w, f)
name, a, w, f = best
print("best:", name, "| a:", round(a, 2), "| OOF weighted:", round(f, 5))
if name == "STAGE2":
    print("SKIP_SUBMIT: pure STAGE2 wins on OOF (file identical to submitted best)")

final_pred = a * pred2 + (1 - a) * pred_nn
sub[target] = [idx2cls[i] for i in (final_pred * w[None, :]).argmax(1)]
sub.to_csv("submission.csv", index=False)
print("saved submission.csv", sub.shape, "| from:", name, "| a:", round(a, 2))
'''

code = CODE.replace("__SMOKE__", "True" if SMOKE else "False")

title = (
    "# S6E6 NN Diversity - categorical-embedding MLP + STAGE2 blend\n\n"
    "Mounts the baseline GBT kernel artifacts (oof/test_probs stage2, "
    "y_train), trains an MLP on the same features (numeric + pairwise "
    "diffs), then searches blend = a*STAGE2 + (1-a)*NN on OOF with "
    "per-class weight re-tuning (grid includes a=1.0 = pure STAGE2).\n\n"
    + ("**SMOKE plumbing check** (3 folds x 3 epochs)."
       if SMOKE else
       "Full run: 5 folds, up to 40 epochs, early stop on val macro-F1.")
)

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": title.split("\n")},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": code.lstrip("\n").splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("s6e6-nn-diversity.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Generated s6e6-nn-diversity.ipynb (SMOKE=%s)" % SMOKE)
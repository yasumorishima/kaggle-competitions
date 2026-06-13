"""Generate S6E6 feature-engineering A/B notebook (.ipynb).

Champion = two-stage pseudo-label GBT ensemble (LB 0.95944). LB-verified
closed levers: external SDSS17 data (-0.00143), pseudo round 2 / stage3
blend (-0.00037), NN-diversity MLP blend (OOF SKIP, NN 0.94071 << GBT).

A/B-tests physics-motivated INTERACTION features on top of the proven
DIFF set (single-feature monotone transforms are invisible to trees, so
only products a tree cannot cheaply represent are added): redshift x
adjacent color, color x color, sign-folding squares, mean magnitude.
Sky coords (alpha/delta) excluded. Same folds, balanced weights,
per-class OOF weight tuning, LGB-only 3-fold early stopping. If
DIFF+FE >= DIFF on OOF, the winning FE block is ported into the
baseline champion generator.
"""

import json

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

SEED = 42
N_SPLITS = 3
MAX_EST = 1000
ES_ROUNDS = 100

IN = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "sample_submission.csv" in files:
        IN = root
if IN is None:
    raise SystemExit("train.csv not found under /kaggle/input")
print("data dir:", IN)

train = pd.read_csv(IN + "/train.csv")
test = pd.read_csv(IN + "/test.csv")
sub = pd.read_csv(IN + "/sample_submission.csv")

id_col = sub.columns[0]
target = sub.columns[1]
classes = sorted(train[target].astype(str).unique())
cls2idx = {c: i for i, c in enumerate(classes)}
n_class = len(classes)
print("train:", train.shape, "| classes:", classes)

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
print("n_diff:", len(diff_cols))

def add_fe(df):
    fe = []
    have = set(num_cols)
    pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
    colors = {}
    for a, b in pairs:
        if a in have and b in have:
            colors[a + b] = df[a] - df[b]
    ckeys = list(colors)
    z = df["redshift"] if "redshift" in have else None
    if z is not None:
        for k in ckeys:
            n = "fe_z_x_" + k
            df[n] = z * colors[k]; fe.append(n)
        df["fe_z2"] = z * z; fe.append("fe_z2")
    cc = [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]
    for a, b in cc:
        if a < len(ckeys) and b < len(ckeys):
            ka, kb = ckeys[a], ckeys[b]
            n = "fe_" + ka + "_x_" + kb
            df[n] = colors[ka] * colors[kb]; fe.append(n)
    for k in ckeys[:2]:
        n = "fe_" + k + "2"
        df[n] = colors[k] * colors[k]; fe.append(n)
    mags = [m for m in ["u", "g", "r", "i", "z"] if m in have]
    if mags:
        df["fe_meanmag"] = df[mags].mean(axis=1); fe.append("fe_meanmag")
    return fe

fe_cols = add_fe(train)
add_fe(test)
print("n_fe:", len(fe_cols), "|", fe_cols)

y = train[target].astype(str).map(cls2idx).values

def tune_weights(oof, y):
    def mf1(w):
        return f1_score(y, (oof * w[None, :]).argmax(1), average="macro")
    bw = np.ones(oof.shape[1]); bf = mf1(bw)
    for it in range(3):
        imp = False
        for c in range(oof.shape[1]):
            for v in np.linspace(0.6, 1.6, 21):
                w = bw.copy(); w[c] = v; f = mf1(w)
                if f > bf + 1e-6:
                    bf, bw, imp = f, w, True
        if not imp:
            break
    return bw, bf

def run_lgb(tag, feats):
    X = train[feats]
    oof = np.zeros((len(train), n_class))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_class,
            n_estimators=MAX_EST, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])
        print(" ", tag, "fold", fold, "best_iter", m.best_iteration_,
              "elapsed", round(time.time() - t0), "s")
    w, fw = tune_weights(oof, y)
    print(tag, "| OOF macro-F1 argmax:",
          round(f1_score(y, oof.argmax(1), average="macro"), 5),
          "| weighted:", round(fw, 5))
    return fw

base = features + diff_cols
f_diff = run_lgb("DIFF", base)
f_fe = run_lgb("DIFF+FE", base + fe_cols)
print("A/B result: diff", round(f_diff, 5), "vs diff+fe", round(f_fe, 5),
      "| delta", round(f_fe - f_diff, 5))
print("decision: PORT FE" if f_fe - f_diff >= -0.0002 else "decision: REJECT FE")
'''

title = (
    "# S6E6 Feature-Engineering A/B - physics interaction features\n\n"
    "A/B test (LGB 3-fold, early stopping, balanced weights, per-class "
    "OOF weight tuning): DIFF vs DIFF + physics interaction features "
    "(redshift x color, color x color products, squares, mean magnitude). "
    "Sky coords excluded.\n"
)

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": title.split("\n")},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": CODE.lstrip("\n").splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("s6e6-fe-ab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Generated s6e6-fe-ab.ipynb")
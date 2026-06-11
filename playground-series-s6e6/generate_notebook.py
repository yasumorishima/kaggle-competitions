"""Generate S6E6 baseline notebook (.ipynb).

Playground Series S6E6 = 3-class classification (GALAXY / QSO / STAR,
SDSS-style stellar object classification). Submission is the predicted
class LABEL ([id, class]) -> accuracy-style metric. Target/id columns and
the class set are auto-detected, so this is robust to the exact schema.

SMOKE=True runs a fast pipeline (LightGBM only, 3-fold, few trees) to
validate the Kaggle-side plumbing + submission.csv output before the full
LGB/XGB/CatBoost multi-seed ensemble. Flip SMOKE to False for the full run.
"""

import json

SMOKE = True

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

SMOKE = __SMOKE__

# Locate competition data: competition_sources mounts at
# /kaggle/input/competitions/<slug>/, so walk to be robust.
IN = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "sample_submission.csv" in files:
        IN = root
if IN is None:
    raise SystemExit("train.csv not found under /kaggle/input")
print("Using data dir:", IN)

train = pd.read_csv(IN + "/train.csv")
test = pd.read_csv(IN + "/test.csv")
sub = pd.read_csv(IN + "/sample_submission.csv")

id_col = sub.columns[0]
target = sub.columns[1]
classes = sorted(train[target].astype(str).unique())
cls2idx = {c: i for i, c in enumerate(classes)}
idx2cls = {i: c for c, i in cls2idx.items()}
n_class = len(classes)
print("id_col:", id_col, "| target:", target, "| classes:", classes)
print("train:", train.shape, "| test:", test.shape)
print(train[target].value_counts(normalize=True))

features = [c for c in train.columns if c not in (id_col, target)]
cat_cols = [c for c in features if str(train[c].dtype) in ("object", "category")]
print("n_features:", len(features), "| n_categorical:", len(cat_cols), cat_cols)

for c in cat_cols:
    cats = pd.concat([train[c], test[c]], axis=0).astype("category").cat.categories
    train[c] = pd.Categorical(train[c], categories=cats).codes
    test[c] = pd.Categorical(test[c], categories=cats).codes

X = train[features]
y = train[target].astype(str).map(cls2idx).values
Xtest = test[features]

N_SPLITS = 3 if SMOKE else 10
SEEDS = [42] if SMOKE else [42, 1, 2025]
LGB_EST = 400 if SMOKE else 4000
XGB_EST = 400 if SMOKE else 4000
CAT_EST = 400 if SMOKE else 4000

oof = np.zeros((len(train), n_class))
pred = np.zeros((len(test), n_class))
n_oof = np.zeros(len(train))
n_pred = 0

import lightgbm as lgb
use_xgb = not SMOKE
use_cat = not SMOKE
if use_xgb:
    import xgboost as xgb
if use_cat:
    from catboost import CatBoostClassifier

t0 = time.time()
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva, ytr = X.iloc[tr], X.iloc[va], y[tr]

        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_class,
            n_estimators=LGB_EST, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        oof[va] += m.predict_proba(Xva)
        pred += m.predict_proba(Xtest)
        n_oof[va] += 1; n_pred += 1

        if use_xgb:
            mx = xgb.XGBClassifier(
                objective="multi:softprob", num_class=n_class,
                n_estimators=XGB_EST, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, tree_method="hist")
            mx.fit(Xtr, ytr)
            oof[va] += mx.predict_proba(Xva)
            pred += mx.predict_proba(Xtest)
            n_oof[va] += 1; n_pred += 1

        if use_cat:
            mc = CatBoostClassifier(
                loss_function="MultiClass", classes_count=n_class,
                iterations=CAT_EST, learning_rate=0.03, depth=6,
                l2_leaf_reg=3.0, random_seed=seed, verbose=0)
            mc.fit(Xtr, ytr)
            oof[va] += mc.predict_proba(Xva)
            pred += mc.predict_proba(Xtest)
            n_oof[va] += 1; n_pred += 1

        print("seed", seed, "fold", fold, "done", round(time.time() - t0), "s")

oof /= np.maximum(n_oof, 1)[:, None]
pred /= max(n_pred, 1)
oof_pred = oof.argmax(1)
print("OOF accuracy:", round(accuracy_score(y, oof_pred), 5))
print("OOF macro-F1:", round(f1_score(y, oof_pred, average="macro"), 5))

test_pred = pred.argmax(1)
sub[target] = [idx2cls[i] for i in test_pred]
sub.to_csv("submission.csv", index=False)
print("saved submission.csv", sub.shape)
print(sub[target].value_counts(normalize=True))
print(sub.head())
'''

code = CODE.replace("__SMOKE__", "True" if SMOKE else "False")

title = (
    "# Playground Series S6E6 — Stellar Classification (GALAXY / QSO / STAR)\n\n"
    "3-class classification, label submission. LightGBM"
    + ("" if SMOKE else " + XGBoost + CatBoost")
    + " multiclass, StratifiedKFold, probability-averaged ensemble (argmax).\n\n"
    + ("**SMOKE run** (LGB-only, 3-fold)." if SMOKE else "Full multi-seed ensemble.")
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

with open("playground-series-s6e6-baseline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Generated playground-series-s6e6-baseline.ipynb (SMOKE=%s)" % SMOKE)

"""Generate S6E6 baseline notebook (.ipynb).

Playground Series S6E6 = binary classification, metric AUC, tabular.
Target/id columns are auto-detected from sample_submission.csv so this is
robust to the exact schema. SMOKE=True runs a fast pipeline (LightGBM only,
3-fold, few trees) to validate the Kaggle-side plumbing + submission.csv
output before the full LGB/XGB/CatBoost multi-seed ensemble.

Flip SMOKE to False (one-line edit) for the full run.
"""

import json

SMOKE = True

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SMOKE = {smoke}
IN = "/kaggle/input/playground-series-s6e6"

train = pd.read_csv(f"{{IN}}/train.csv")
test = pd.read_csv(f"{{IN}}/test.csv")
sub = pd.read_csv(f"{{IN}}/sample_submission.csv")

id_col = sub.columns[0]
target = sub.columns[1]
print("id_col:", id_col, "| target:", target)
print("train:", train.shape, "| test:", test.shape)
print("target balance:\n", train[target].value_counts(normalize=True))

features = [c for c in train.columns if c not in (id_col, target)]
cat_cols = [c for c in features if str(train[c].dtype) in ("object", "category")]
print("n_features:", len(features), "| n_categorical:", len(cat_cols))

# Label-encode categoricals on the union of train+test categories
for c in cat_cols:
    cats = pd.concat([train[c], test[c]], axis=0).astype("category").cat.categories
    train[c] = pd.Categorical(train[c], categories=cats).codes
    test[c] = pd.Categorical(test[c], categories=cats).codes

X = train[features]
y = train[target].astype(int).values
Xtest = test[features]

N_SPLITS = 3 if SMOKE else 10
SEEDS = [42] if SMOKE else [42, 1, 2025]
LGB_EST = 300 if SMOKE else 4000
XGB_EST = 300 if SMOKE else 4000
CAT_EST = 300 if SMOKE else 4000

def rank01(a):
    a = pd.Series(a)
    return (a.rank() / len(a)).values

oof = np.zeros(len(train))
pred = np.zeros(len(test))
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
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]

        # --- LightGBM ---
        m = lgb.LGBMClassifier(
            n_estimators=LGB_EST, learning_rate=0.02, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        p_va = rank01(m.predict_proba(Xva)[:, 1])
        p_te = rank01(m.predict_proba(Xtest)[:, 1])
        oof[va] += p_va; pred += p_te; n_oof[va] += 1; n_pred += 1

        # --- XGBoost ---
        if use_xgb:
            mx = xgb.XGBClassifier(
                n_estimators=XGB_EST, learning_rate=0.02, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="auc", random_state=seed, n_jobs=-1,
                tree_method="hist")
            mx.fit(Xtr, ytr)
            oof[va] += rank01(mx.predict_proba(Xva)[:, 1])
            pred += rank01(mx.predict_proba(Xtest)[:, 1])
            n_oof[va] += 1; n_pred += 1

        # --- CatBoost ---
        if use_cat:
            mc = CatBoostClassifier(
                iterations=CAT_EST, learning_rate=0.02, depth=6,
                l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                eval_metric="AUC")
            mc.fit(Xtr, ytr)
            oof[va] += rank01(mc.predict_proba(Xva)[:, 1])
            pred += rank01(mc.predict_proba(Xtest)[:, 1])
            n_oof[va] += 1; n_pred += 1

        print(f"seed={{seed}} fold={{fold}} done ({{time.time()-t0:.0f}}s)")

oof /= np.maximum(n_oof, 1)
pred /= max(n_pred, 1)
print("\\nOOF AUC:", round(roc_auc_score(y, oof), 5))

sub[target] = pred
sub.to_csv("submission.csv", index=False)
print("saved submission.csv", sub.shape)
print(sub.head())
'''

code = CODE.replace("{smoke}", "True" if SMOKE else "False")

title = (
    "# Playground Series S6E6 — GBDT Ensemble (AUC)\n\n"
    "Tabular binary classification. LightGBM"
    + ("" if SMOKE else " + XGBoost + CatBoost")
    + ", StratifiedKFold, rank-averaged ensemble.\n\n"
    "Target/id auto-detected from `sample_submission.csv`. "
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

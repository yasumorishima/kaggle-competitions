"""Generate S6E6 baseline notebook (.ipynb).

Playground Series S6E6 = 3-class classification (GALAXY / QSO / STAR,
SDSS-style stellar object classification). Submission is the predicted
class LABEL ([id, class]). LB metric is macro-F1 (proven 2026-06-11:
OOF macro-F1 0.95494 vs LB 0.95466), so the pipeline optimizes macro-F1
directly: balanced class weights during training + per-class probability
weights tuned on OOF by coordinate ascent before argmax.

Feature engineering: pairwise differences of all numeric columns
(generalized color indices, e.g. u-g / g-r for SDSS photometry — GBT
axis-aligned splits cannot represent x-y directly).

SMOKE=True runs an A/B test (LightGBM only, 3-fold, early stopping):
base features vs base+diff on identical folds, printing OOF macro-F1
for both, to validate the diff features before the full ensemble.
SMOKE=False runs the full LGB/XGB/CatBoost ensemble (1 seed x 5 folds,
early stopping, cap 4000 trees) on base+diff.
"""

import json

SMOKE = False

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

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

features = [c for c in train.columns if c not in (id_col, target)]
cat_cols = [c for c in features if str(train[c].dtype) in ("object", "category")]
num_cols = [c for c in features if c not in cat_cols]
print("numeric features:", num_cols)
print("categorical features:", cat_cols)

for c in cat_cols:
    cats = pd.concat([train[c], test[c]], axis=0).astype("category").cat.categories
    train[c] = pd.Categorical(train[c], categories=cats).codes
    test[c] = pd.Categorical(test[c], categories=cats).codes

# Pairwise differences of numeric columns (generalized color indices).
diff_cols = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        a, b = num_cols[i], num_cols[j]
        name = "d_" + a + "_" + b
        train[name] = train[a] - train[b]
        test[name] = test[a] - test[b]
        diff_cols.append(name)
print("n_diff_features:", len(diff_cols))

y = train[target].astype(str).map(cls2idx).values
cls_w = compute_class_weight("balanced", classes=np.arange(n_class), y=y)
print("balanced class weights:", dict(zip(classes, np.round(cls_w, 3))))

N_SPLITS = 3 if SMOKE else 5
SEED = 42
MAX_EST = 1000 if SMOKE else 4000
ES_ROUNDS = 100 if SMOKE else 200

import lightgbm as lgb

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

def run_lgb(cols, tag):
    X = train[cols]
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
    f1_arg = f1_score(y, oof.argmax(1), average="macro")
    w, f1_w = tune_weights(oof, y)
    print(tag, "| OOF macro-F1 argmax:", round(f1_arg, 5),
          "| weighted:", round(f1_w, 5),
          "| weights:", dict(zip(classes, np.round(w, 3))))
    return f1_w

if SMOKE:
    # A/B: identical folds, base vs base+diff.
    f_base = run_lgb(features, "BASE")
    f_diff = run_lgb(features + diff_cols, "BASE+DIFF")
    print("A/B result: base", round(f_base, 5), "vs base+diff", round(f_diff, 5),
          "| delta", round(f_diff - f_base, 5))
    raise SystemExit(0)

# ---- Full ensemble (base+diff) ----
FEATS = features + diff_cols
X = train[FEATS]
Xtest = test[FEATS]

oof = np.zeros((len(train), n_class))
pred = np.zeros((len(test), n_class))
n_oof = np.zeros(len(train))
n_pred = 0

import xgboost as xgb
from catboost import CatBoostClassifier

t0 = time.time()
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]
    wtr = compute_sample_weight("balanced", ytr)

    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=n_class,
        n_estimators=MAX_EST, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    oof[va] += m.predict_proba(Xva)
    pred += m.predict_proba(Xtest)
    n_oof[va] += 1; n_pred += 1
    print("  lgb best_iter:", m.best_iteration_)

    mx = xgb.XGBClassifier(
        objective="multi:softprob", num_class=n_class,
        n_estimators=MAX_EST, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        early_stopping_rounds=ES_ROUNDS,
        random_state=SEED, n_jobs=-1, tree_method="hist")
    mx.fit(Xtr, ytr, sample_weight=wtr,
           eval_set=[(Xva, yva)], verbose=False)
    oof[va] += mx.predict_proba(Xva)
    pred += mx.predict_proba(Xtest)
    n_oof[va] += 1; n_pred += 1
    print("  xgb best_iter:", mx.best_iteration)

    mc = CatBoostClassifier(
        loss_function="MultiClass", classes_count=n_class,
        iterations=MAX_EST, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, class_weights=list(cls_w),
        random_seed=SEED, verbose=0)
    mc.fit(Xtr, ytr, eval_set=(Xva, yva),
           early_stopping_rounds=ES_ROUNDS, use_best_model=True)
    oof[va] += mc.predict_proba(Xva)
    pred += mc.predict_proba(Xtest)
    n_oof[va] += 1; n_pred += 1
    print("  cat best_iter:", mc.get_best_iteration())

    print("fold", fold, "done", round(time.time() - t0), "s")

oof /= np.maximum(n_oof, 1)[:, None]
pred /= max(n_pred, 1)
oof_pred = oof.argmax(1)
print("OOF accuracy:", round(accuracy_score(y, oof_pred), 5))
print("OOF macro-F1 (argmax):", round(f1_score(y, oof_pred, average="macro"), 5))

best_w, best_f1 = tune_weights(oof, y)
print("best per-class weights:", dict(zip(classes, np.round(best_w, 3))))
print("OOF macro-F1 (weighted):", round(best_f1, 5))

test_pred = (pred * best_w[None, :]).argmax(1)
sub[target] = [idx2cls[i] for i in test_pred]
sub.to_csv("submission.csv", index=False)
print("saved submission.csv", sub.shape)
print(sub[target].value_counts(normalize=True))
print(sub.head())
'''

code = CODE.replace("__SMOKE__", "True" if SMOKE else "False")

title = (
    "# Playground Series S6E6 — Stellar Classification (GALAXY / QSO / STAR)\n\n"
    "3-class classification, label submission, macro-F1 optimized "
    "(balanced class weights + per-class probability weights tuned on OOF).\n"
    "Feature engineering: pairwise numeric differences (generalized color indices).\n\n"
    + ("**SMOKE A/B run** (LGB-only, 3-fold): base vs base+diff features."
       if SMOKE else
       "Full ensemble: LGB + XGB + CatBoost, 1 seed x 5 folds, early stopping, base+diff features.")
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

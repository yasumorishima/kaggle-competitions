"""Generate S6E6 baseline notebook (.ipynb).

Playground Series S6E6 = 3-class classification (GALAXY / QSO / STAR,
SDSS stellar object classification: alpha, delta, u/g/r/i/z, redshift
+ 2 synthetic categoricals). Submission is the predicted class LABEL
([id, class]). LB metric is macro-F1 (proven 2026-06-11: OOF macro-F1
matches LB within ~0.001), so the pipeline optimizes macro-F1 directly:
balanced class weights during training + per-class probability weights
tuned on OOF by coordinate ascent before argmax.

Feature engineering: pairwise differences of all numeric columns
(generalized color indices; A/B validated +0.00093 OOF macro-F1).

External-data lever REJECTED (2026-06-12): appending original SDSS17
rows to fold-train raised OOF (+0.00022; near-duplicates of validation
rows leak) but dropped LB 0.95884 -> 0.95741. OOF deltas only predict
LB deltas for same-train-distribution changes.

Current lever: pseudo-label distillation. Stage 1 trains on train only
and predicts test; test rows whose max averaged probability >=
PSEUDO_CONF are added to the TRAINING part of each fold only (never to
validation) in stage 2. Pseudo rows come from the test distribution
itself, so the distribution-shift failure mode of the external-data
lever does not apply; the honest check is still the LB.

SMOKE=True runs an A/B test (LightGBM only, 3-fold, early stopping)
on identical folds: DIFF vs DIFF+PSEUDO (stage 1 = the A run itself,
reusing its averaged test predictions).
SMOKE=False runs the two-stage full LGB/XGB/CatBoost ensemble
(1 seed x 5 folds, early stopping, cap 4000 trees); submission comes
from stage 2.

Round-1 result (2026-06-12): two-stage pseudo-label = LB 0.95944, new
best (prior 0.95884; STAGE2 OOF +0.00025 understated the +0.0006 LB
gain - test-distribution alignment does not show in OOF). This version
additionally saves OOF and averaged test probabilities as .npy kernel
outputs so downstream kernels (pseudo round 2 / stage blending / NN
diversity) can mount them via kernel_sources instead of retraining
stage 1+2 (6.7h) inside a single 12h kernel.
"""

import json

SMOKE = False

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

SMOKE = __SMOKE__
PSEUDO_CONF = 0.995

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
print("train:", train.shape, "| test:", test.shape, "| classes:", classes)

features = [c for c in train.columns if c not in (id_col, target)]
cat_cols = [c for c in features if str(train[c].dtype) in ("object", "category")]
num_cols = [c for c in features if c not in cat_cols]

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

FEATS = features + diff_cols
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

def build_pseudo(pred):
    """Confident test rows (raw averaged probs) -> pseudo training rows."""
    conf = pred.max(1)
    keep = conf >= PSEUDO_CONF
    Xp = test.loc[keep, FEATS].reset_index(drop=True)
    yp = pred[keep].argmax(1)
    uniq, cnt = np.unique(yp, return_counts=True)
    print("pseudo rows:", int(keep.sum()), "/", len(test),
          "| class counts:", {idx2cls[int(u)]: int(c) for u, c in zip(uniq, cnt)})
    return Xp, yp

def fold_train_data(tr, pseudo):
    """Training matrix for one fold; pseudo rows go to train side only."""
    Xtr, ytr = train[FEATS].iloc[tr], y[tr]
    if pseudo is not None:
        Xp, yp = pseudo
        Xtr = pd.concat([Xtr, Xp], axis=0, ignore_index=True)
        ytr = np.concatenate([ytr, yp])
    return Xtr, ytr

def run_lgb(tag, pseudo, predict_test=False):
    X = train[FEATS]
    Xtest = test[FEATS]
    oof = np.zeros((len(train), n_class))
    pred = np.zeros((len(test), n_class)) if predict_test else None
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr = fold_train_data(tr, pseudo)
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_class,
            n_estimators=MAX_EST, learning_rate=0.04117702665865346,
            num_leaves=255, max_depth=8, min_child_samples=120,
            subsample=0.987935611121138, colsample_bytree=0.8790103068923594,
            reg_lambda=0.9201056783506876, reg_alpha=0.03178900344437503,
            class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])
        if predict_test:
            pred += m.predict_proba(Xtest)
        print(" ", tag, "fold", fold, "best_iter", m.best_iteration_,
              "elapsed", round(time.time() - t0), "s")
    if predict_test:
        pred /= N_SPLITS
    f1_arg = f1_score(y, oof.argmax(1), average="macro")
    w, f1_w = tune_weights(oof, y)
    print(tag, "| OOF macro-F1 argmax:", round(f1_arg, 5),
          "| weighted:", round(f1_w, 5),
          "| weights:", dict(zip(classes, np.round(w, 3))))
    return f1_w, pred

if SMOKE:
    f_a, pred_a = run_lgb("DIFF", None, predict_test=True)
    pseudo = build_pseudo(pred_a)
    f_b, _ = run_lgb("DIFF+PSEUDO", pseudo)
    print("A/B result: diff", round(f_a, 5), "vs diff+pseudo", round(f_b, 5),
          "| delta", round(f_b - f_a, 5))
    raise SystemExit(0)

# ---- Full two-stage ensemble (LGB + XGB + CatBoost) ----
import xgboost as xgb
from catboost import CatBoostClassifier

X = train[FEATS]
Xtest = test[FEATS]

def run_ensemble(stage, pseudo):
    oof = np.zeros((len(train), n_class))
    pred = np.zeros((len(test), n_class))
    n_oof = np.zeros(len(train))
    n_pred = 0
    t0 = time.time()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr = fold_train_data(tr, pseudo)
        Xva, yva = X.iloc[va], y[va]
        wtr = compute_sample_weight("balanced", ytr)

        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_class,
            n_estimators=MAX_EST, learning_rate=0.04117702665865346,
            num_leaves=255, max_depth=8, min_child_samples=120,
            subsample=0.987935611121138, colsample_bytree=0.8790103068923594,
            reg_lambda=0.9201056783506876, reg_alpha=0.03178900344437503,
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

        print(stage, "fold", fold, "done", round(time.time() - t0), "s")

    oof /= np.maximum(n_oof, 1)[:, None]
    pred /= max(n_pred, 1)
    f1_arg = f1_score(y, oof.argmax(1), average="macro")
    w, f1_w = tune_weights(oof, y)
    print(stage, "| OOF macro-F1 argmax:", round(f1_arg, 5),
          "| weighted:", round(f1_w, 5),
          "| weights:", dict(zip(classes, np.round(w, 3))))
    return oof, pred, w, f1_w

oof1, pred1, w1, f1w1 = run_ensemble("STAGE1", None)
pseudo = build_pseudo(pred1)
oof2, pred2, w2, f1w2 = run_ensemble("STAGE2", pseudo)
print("stage1 weighted:", round(f1w1, 5), "| stage2 weighted:", round(f1w2, 5),
      "| delta", round(f1w2 - f1w1, 5))

# Persist artifacts for cross-kernel chaining (pseudo round 2, stage
# blending, NN diversity): mounted by downstream kernels via
# kernel_sources at /kaggle/input/<kernel-slug>/.
np.save("oof_stage1.npy", oof1)
np.save("test_probs_stage1.npy", pred1)
np.save("oof_stage2.npy", oof2)
np.save("test_probs_stage2.npy", pred2)
np.save("y_train.npy", y)
print("saved npy artifacts: oof/test_probs stage1+stage2, y_train")

test_pred = (pred2 * w2[None, :]).argmax(1)
sub[target] = [idx2cls[i] for i in test_pred]
sub.to_csv("submission.csv", index=False)
print("saved submission.csv", sub.shape)
print(sub[target].value_counts(normalize=True))
print(sub.head())
'''

code = CODE.replace("__SMOKE__", "True" if SMOKE else "False")

title = (
    "# Playground Series S6E6 - Stellar Classification (GALAXY / QSO / STAR)\n\n"
    "3-class classification, label submission, macro-F1 optimized "
    "(balanced class weights + per-class probability weights tuned on OOF).\n"
    "Features: pairwise numeric differences (generalized color indices). "
    "Lever: pseudo-label distillation (confident test rows join fold-train).\n\n"
    + ("**SMOKE A/B run** (LGB-only, 3-fold): diff vs diff+pseudo-labels."
       if SMOKE else
       "Full two-stage ensemble: LGB + XGB + CatBoost, 1 seed x 5 folds, "
       "early stopping; stage 2 adds confident pseudo-labeled test rows.")
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
"""Generate S6E6 stage-3 notebook (.ipynb): pseudo round 2 + stage blend.

Chained kernel: mounts the baseline kernel output (kernel_sources =
yasunorim/s6e6-gbdt-ensemble-baseline) which persists oof_stage2.npy /
test_probs_stage2.npy / y_train.npy, so stage 1+2 (6.7h) are not
retrained here.

Fact base (2026-06-12): two-stage pseudo-label = LB 0.95944 (best;
STAGE1 OOF weighted 0.95771 / STAGE2 0.95796). OOF deltas predict LB
deltas only for same-train-distribution changes; the stage2 / stage3 /
blend comparisons here are all on the same train rows and folds, so
the OOF signal is valid for selection.

SMOKE=True: LGB-only 3-fold plumbing check (artifact mount, pseudo
round 2, fold_train_data) then exit - no submission decision.
SMOKE=False: full STAGE3 (LGB+XGB+Cat, 1 seed x 5 folds, ES, cap
4000) trained with pseudo round 2 from STAGE2 test probabilities,
then a stage-blend alpha search on OOF; submission = best of
{STAGE2, STAGE3, BLEND} by OOF weighted macro-F1 (if STAGE2 wins the
file equals the already-submitted LB 0.95944 - do not resubmit).
"""

import json

SMOKE = False

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

SMOKE = __SMOKE__
PSEUDO_CONF = 0.995

IN = None
ART = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "sample_submission.csv" in files:
        IN = root
    if "test_probs_stage2.npy" in files:
        ART = root
if IN is None:
    raise SystemExit("train.csv not found under /kaggle/input")
if ART is None:
    raise SystemExit("stage2 artifacts not found - check kernel_sources mount")
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
print("train:", train.shape, "| test:", test.shape, "| classes:", classes)

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
print("n_diff_features:", len(diff_cols))

FEATS = features + diff_cols
y = train[target].astype(str).map(cls2idx).values
cls_w = compute_class_weight("balanced", classes=np.arange(n_class), y=y)

y_art = np.load(ART + "/y_train.npy")
assert y_art.shape[0] == y.shape[0], "y length mismatch vs artifacts"
assert (y_art == y).all(), "y content mismatch vs artifacts"
oof2 = np.load(ART + "/oof_stage2.npy")
pred2 = np.load(ART + "/test_probs_stage2.npy")
assert oof2.shape == (len(train), n_class), oof2.shape
assert pred2.shape == (len(test), n_class), pred2.shape
print("artifacts loaded:", oof2.shape, pred2.shape)

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

w2, f1w2 = tune_weights(oof2, y)
print("STAGE2 (loaded) | OOF weighted:", round(f1w2, 5), "| expect ~0.95796")

pseudo2 = build_pseudo(pred2)

if SMOKE:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), n_class))
    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(train[FEATS], y)):
        Xtr, ytr = fold_train_data(tr, pseudo2)
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_class,
            n_estimators=MAX_EST, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(train[FEATS].iloc[va], y[va])],
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        oof[va] = m.predict_proba(train[FEATS].iloc[va])
        print("  smoke fold", fold, "best_iter", m.best_iteration_,
              "elapsed", round(time.time() - t0), "s")
    wq, fq = tune_weights(oof, y)
    print("SMOKE pseudo-round2 LGB 3fold | OOF weighted:", round(fq, 5))
    print("PLUMBING OK")
    raise SystemExit(0)

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

        print(stage, "fold", fold, "done", round(time.time() - t0), "s")

    oof /= np.maximum(n_oof, 1)[:, None]
    pred /= max(n_pred, 1)
    f1_arg = f1_score(y, oof.argmax(1), average="macro")
    w, f1_w = tune_weights(oof, y)
    print(stage, "| OOF macro-F1 argmax:", round(f1_arg, 5),
          "| weighted:", round(f1_w, 5),
          "| weights:", dict(zip(classes, np.round(w, 3))))
    return oof, pred, w, f1_w

oof3, pred3, w3, f1w3 = run_ensemble("STAGE3", pseudo2)
np.save("oof_stage3.npy", oof3)
np.save("test_probs_stage3.npy", pred3)
print("stage2:", round(f1w2, 5), "| stage3:", round(f1w3, 5),
      "| delta", round(f1w3 - f1w2, 5))

best_a, best_wb, best_fb = 0.5, None, -1.0
for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    bo = a * oof2 + (1.0 - a) * oof3
    wb, fb = tune_weights(bo, y)
    print("blend a=%.1f | OOF weighted: %.5f" % (a, fb))
    if fb > best_fb:
        best_a, best_wb, best_fb = a, wb, fb
print("best blend: a=%.1f | OOF weighted: %.5f" % (best_a, best_fb))

cands = [
    ("STAGE2", pred2, w2, f1w2),
    ("STAGE3", pred3, w3, f1w3),
    ("BLEND%.1f" % best_a, best_a * pred2 + (1.0 - best_a) * pred3, best_wb, best_fb),
]
tag, predf, wf, f1f = max(cands, key=lambda t: t[3])
print("submission from:", tag, "| OOF weighted:", round(f1f, 5))
if tag == "STAGE2":
    print("SKIP_SUBMIT: best candidate equals already-submitted STAGE2 (LB 0.95944)")
test_pred = (predf * wf[None, :]).argmax(1)
sub[target] = [idx2cls[int(i)] for i in test_pred]
sub.to_csv("submission.csv", index=False)
print("saved submission.csv", sub.shape)
print(sub[target].value_counts(normalize=True))
'''

code = CODE.replace("__SMOKE__", "True" if SMOKE else "False")

title = (
    "# S6E6 Stage3 - Pseudo Round 2 + Stage Blend\n\n"
    "Chained kernel: loads STAGE2 OOF / test probabilities from the baseline "
    "kernel output (kernel_sources), builds pseudo round 2 (conf 0.995-plus), "
    "trains STAGE3, then picks best of STAGE2 / STAGE3 / blend by OOF "
    "weighted macro-F1.\n\n"
    + ("**SMOKE plumbing check** (LGB-only, 3-fold)."
       if SMOKE else
       "Full STAGE3: LGB + XGB + CatBoost, 1 seed x 5 folds, early stopping.")
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

with open("s6e6-stage3-pseudo-round2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Generated s6e6-stage3-pseudo-round2.ipynb (SMOKE=%s)" % SMOKE)
"""Generate S6E6 Optuna HPO proxy notebook (.ipynb).

Champion = two-stage pseudo-label GBT ensemble (LB 0.95944, STAGE2 OOF
weighted 0.95796). Closed levers (this session): external SDSS17 data,
pseudo round 2, NN-diversity MLP blend, physics interaction FE - all
rejected on LB/OOF. Remaining credible lever: hyperparameter search.

This kernel runs Optuna (TPE) over LightGBM on the proven DIFF feature
set, 3-fold early stopping, optimizing OOF macro-F1 with per-class
probability weights (the proven metric). Default-LGB 3-fold baseline =
0.95684 (reproduced twice). If the best trial clears it by a margin,
the winning LGB params are ported into the baseline champion generator's
LGB component and the full two-stage ensemble is re-run.

SMOKE=True: 2 trials, small tree cap (loop + objective sanity).
SMOKE=False: N_TRIALS trials at the proxy tree cap.
"""

import json

SMOKE = False

CODE = r'''
import os, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import optuna
import json

SMOKE = __SMOKE__
SEED = 42
N_SPLITS = 3
N_TRIALS = 2 if SMOKE else 25
MAX_EST = 600 if SMOKE else 1500
ES_ROUNDS = 100
BASELINE = 0.95684  # default-LGB 3fold DIFF, reproduced 2026-06-11 & -13

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
FEATS = features + diff_cols
y = train[target].astype(str).map(cls2idx).values
print("train:", train.shape, "| n_feats:", len(FEATS), "| trials:", N_TRIALS)

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
    return bf

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(skf.split(train[FEATS], y))
X = train[FEATS]

def objective(trial):
    params = dict(
        objective="multiclass", num_class=n_class,
        n_estimators=MAX_EST,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 120),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros((len(train), n_class))
    t0 = time.time()
    for tr, va in splits:
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])
    fw = tune_weights(oof, y)
    print("trial", trial.number, "| OOF weighted:", round(fw, 5),
          "| elapsed", round(time.time() - t0), "s")
    return fw

sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS)

print("=== best ===")
print("best OOF weighted:", round(study.best_value, 5),
      "| vs baseline", BASELINE,
      "| delta", round(study.best_value - BASELINE, 5))
print("best params:", json.dumps(study.best_params))
df = study.trials_dataframe()
df.to_csv("optuna_trials.csv", index=False)
print("decision: PORT HPO" if study.best_value - BASELINE >= 0.0005 else "decision: REJECT HPO (gain < +0.0005)")
'''

code = CODE.replace("__SMOKE__", "True" if SMOKE else "False")

title = (
    "# S6E6 Optuna HPO proxy - LightGBM over DIFF features\n\n"
    "TPE search over LGB (3-fold, early stopping) optimizing OOF macro-F1 "
    "with per-class weights. Baseline default-LGB = 0.95684. Winning "
    "params port into the champion ensemble if the gain clears +0.0005.\n"
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

with open("s6e6-hpo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Generated s6e6-hpo.ipynb (SMOKE=%s)" % SMOKE)
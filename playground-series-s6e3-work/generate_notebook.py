"""Generate S6E3 Customer Churn EDA + Full Pipeline notebook.

Strategy (no shortcuts):
- Heavy feature engineering: target encoding (CV-aware), interaction/polynomial/ratio features
- Multi-model: LightGBM + XGBoost + CatBoost, each with Optuna hyperparameter tuning
- Multi-seed ensemble: 5 seeds x 3 models x 10 folds = 150 models
- Stacking: OOF predictions as meta-features -> LogisticRegression + Ridge
- Rank averaging as final ensemble
- W&B offline tracking
"""

import json

cells = []
cell_counter = 0


def add_md(source):
    global cell_counter
    cell_counter += 1
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "id": f"cell-{cell_counter:03d}", "metadata": {}, "source": src})


def add_code(source):
    global cell_counter
    cell_counter += 1
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append(
        {
            "cell_type": "code",
            "id": f"cell-{cell_counter:03d}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        }
    )


# ── Title ──
add_md(
    """# S6E3 Predict Customer Churn — Optuna-Tuned Multi-Model Stacking

**Approach**: Optuna HPO per model → 5 seeds × 3 GBDT (LGB/XGB/Cat) × 10-fold CV → Stacking + Rank Avg

**Tools** (self-made PyPI):
- [`kaggle-notebook-deploy`](https://pypi.org/project/kaggle-notebook-deploy/)
- [`kaggle-wandb-sync`](https://pypi.org/project/kaggle-wandb-sync/)"""
)

# ── Setup ──
add_code(
    """import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'kaggle-s6e3-churn'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import wandb
import warnings
warnings.filterwarnings('ignore')
import time
import gc

plt.style.use('seaborn-v0_8-whitegrid')
print('All libraries loaded.')"""
)

# ── W&B init ──
add_code(
    """run = wandb.init(
    project='kaggle-s6e3-churn',
    name='optuna-stacking-v1',
    tags=['optuna', 'stacking', 'multi-seed', '10fold', 'gpu'],
    config={
        'n_seeds': 5, 'n_splits': 10, 'models': ['lgb', 'xgb', 'cat'],
        'optuna_trials': 60, 'strategy': 'optuna_hpo + multi_seed_stacking',
    },
)
print(f'W&B run: {run.name}')"""
)

# ── Load data ──
add_code(
    """import glob as _glob

_slug = 'playground-series-s6e3'
_matches = _glob.glob(f'/kaggle/input/**/{_slug}', recursive=True)
DATA_DIR = _matches[0] if _matches else f'/kaggle/input/{_slug}'
print(f'DATA_DIR: {DATA_DIR}')

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
submission = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

print(f'Train: {train.shape}, Test: {test.shape}')
print(f'Columns: {train.columns.tolist()}')
print(f'\\nDtypes:\\n{train.dtypes}')
print(f'\\nTarget column (last): {train.columns[-1]}')
print(f'Target distribution:\\n{train.iloc[:, -1].value_counts()}')
print(f'\\nMissing values:\\n{train.isnull().sum()[train.isnull().sum() > 0]}')
print(f'\\nHead:\\n{train.head()}')"""
)

# ── EDA: Target + Distributions ──
add_code(
    """# Identify target and ID columns
TARGET = submission.columns[-1]  # last column of submission is target
ID = submission.columns[0]       # first column is id
print(f'TARGET: {TARGET}, ID: {ID}')

# Identify categorical vs numerical
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in [ID, TARGET]]
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [ID, TARGET]]
print(f'Categorical ({len(cat_cols)}): {cat_cols}')
print(f'Numerical ({len(num_cols)}): {num_cols}')

# Target encoding for analysis
if train[TARGET].dtype == 'object':
    target_le = LabelEncoder()
    y_analysis = target_le.fit_transform(train[TARGET])
    print(f'Target classes: {list(target_le.classes_)}')
else:
    y_analysis = train[TARGET].values

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = train[TARGET].value_counts()
axes[0].bar(range(len(counts)), counts.values, tick_label=counts.index.astype(str))
axes[0].set_title('Target Count', fontsize=16, fontweight='bold')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + v * 0.01, str(v), ha='center', fontweight='bold', fontsize=14)
axes[0].tick_params(labelsize=14)

props = counts / counts.sum()
axes[1].pie(props.values, labels=props.index.astype(str), autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 14})
axes[1].set_title('Target Proportion', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()"""
)

# ── EDA: Feature distributions by target ──
add_code(
    """# Numerical features by target
if len(num_cols) > 0:
    n = len(num_cols)
    ncols_plot = 4
    nrows_plot = (n + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(18, 4 * nrows_plot))
    axes = axes.flatten() if nrows_plot > 1 else [axes] if ncols_plot == 1 else axes.flatten()
    for i, col in enumerate(num_cols):
        for label in sorted(train[TARGET].unique()):
            mask = train[TARGET] == label
            axes[i].hist(train.loc[mask, col].dropna(), alpha=0.6, label=str(label), bins=40, edgecolor='white')
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].tick_params(labelsize=10)
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Numerical Features by Target', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

# Categorical features
if len(cat_cols) > 0:
    n = len(cat_cols)
    ncols_plot = min(3, n)
    nrows_plot = (n + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(6 * ncols_plot, 4 * nrows_plot))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(train[col], train[TARGET], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].tick_params(labelsize=10)
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Categorical Features by Target', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()"""
)

# ── EDA: Correlation ──
add_code(
    """# Correlation with target
train_corr = train.copy()
if train_corr[TARGET].dtype == 'object':
    train_corr['_target_num'] = y_analysis
else:
    train_corr['_target_num'] = train_corr[TARGET]

# Encode categoricals for correlation
for col in cat_cols:
    train_corr[col] = LabelEncoder().fit_transform(train_corr[col].astype(str))

corr_cols = num_cols + cat_cols + ['_target_num']
corr = train_corr[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=len(corr_cols) < 25, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

target_corr = corr['_target_num'].drop('_target_num').abs().sort_values(ascending=False)
print('Top correlations with target:')
for feat, val in target_corr.items():
    d = '+' if corr.loc[feat, '_target_num'] > 0 else '-'
    print(f'  {feat:35s} {d}{val:.4f}')"""
)

# ── Feature Engineering ──
add_code(
    """# ========================================
# Feature Engineering — go heavy
# ========================================
train_len = len(train)
df = pd.concat([train, test], axis=0, ignore_index=True)

# --- 1. Label encode categoricals ---
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- 2. Target encode categoricals (CV-aware, train only, mean for test) ---
for col in cat_cols:
    te_col = f'{col}_te'
    df[te_col] = np.nan
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_target = y_analysis
    for tr_idx, va_idx in skf_te.split(df.iloc[:train_len], train_target):
        mean_map = df.iloc[tr_idx].groupby(col)[TARGET if df[TARGET].dtype != 'object' else col].mean()
        # Use label-encoded target for mean
        tr_df = df.iloc[:train_len].copy()
        tr_df['_y'] = train_target
        mean_map = tr_df.iloc[tr_idx].groupby(col)['_y'].mean()
        df.loc[df.index[va_idx], te_col] = df.iloc[va_idx][col].map(mean_map)
    # Test: use full train mean
    tr_df = df.iloc[:train_len].copy()
    tr_df['_y'] = train_target
    full_mean = tr_df.groupby(col)['_y'].mean()
    test_mask = df.index >= train_len
    df.loc[test_mask, te_col] = df.loc[test_mask, col].map(full_mean)
    # Fill NaN with global mean
    df[te_col] = df[te_col].fillna(train_target.mean())

print(f'After target encoding: {df.shape}')

# --- 3. Numerical interactions (top correlated pairs) ---
if len(num_cols) >= 2:
    # Generate all pairwise interactions for top features
    top_num = target_corr.head(min(8, len(num_cols))).index.tolist()
    top_num = [c for c in top_num if c in num_cols]
    interaction_count = 0
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
            df[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + 1e-8)
            df[f'{c1}_plus_{c2}'] = df[c1] + df[c2]
            df[f'{c1}_minus_{c2}'] = df[c1] - df[c2]
            interaction_count += 4
    print(f'Interaction features added: {interaction_count}')

# --- 4. Polynomial features for top correlated ---
if len(num_cols) >= 1:
    top3 = target_corr.head(min(3, len(num_cols))).index.tolist()
    top3 = [c for c in top3 if c in num_cols]
    for col in top3:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    print(f'Polynomial features for: {top3}')

# --- 5. Frequency encoding ---
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)

# --- 6. Aggregate features (mean/std per categorical group) ---
for cat_col in cat_cols[:3]:  # top 3 categoricals only
    for num_col in num_cols[:5]:  # top 5 numericals
        grp = df.groupby(cat_col)[num_col]
        df[f'{num_col}_mean_by_{cat_col}'] = grp.transform('mean')
        df[f'{num_col}_std_by_{cat_col}'] = grp.transform('std').fillna(0)

print(f'Final feature count: {df.shape[1]}')

# --- Build X, y, X_test ---
drop_cols = [ID, TARGET]
if '_y' in df.columns:
    drop_cols.append('_y')
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df.iloc[:train_len][feature_cols].values.astype(np.float32)
y = y_analysis.astype(int)
X_test = df.iloc[train_len:][feature_cols].values.astype(np.float32)

print(f'X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}')
print(f'y distribution: {np.bincount(y)}')
wandb.config.update({'n_features': len(feature_cols), 'feature_names': feature_cols[:50]})"""
)

# ── Optuna HPO for LightGBM ──
add_code(
    """# ========================================
# Optuna Hyperparameter Optimization
# ========================================
N_OPTUNA_TRIALS = 60
N_HPO_FOLDS = 5
HPO_SEED = 42

def lgb_objective(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'device': 'gpu',
    }
    skf = StratifiedKFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=HPO_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing LightGBM...')
t0 = time.time()
lgb_study = optuna.create_study(direction='maximize', study_name='lgb')
lgb_study.optimize(lgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
lgb_best = lgb_study.best_params
lgb_best_score = lgb_study.best_value
print(f'LGB best AUC: {lgb_best_score:.5f} ({time.time()-t0:.0f}s)')
print(f'Best params: {lgb_best}')
wandb.log({'lgb_optuna_best_auc': lgb_best_score, 'lgb_optuna_params': lgb_best})"""
)

# ── Optuna HPO for XGBoost ──
add_code(
    """def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'n_estimators': 2000, 'early_stopping_rounds': 50,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'verbosity': 0,
        'tree_method': 'hist', 'device': 'cuda',
    }
    skf = StratifiedKFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=HPO_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing XGBoost...')
t0 = time.time()
xgb_study = optuna.create_study(direction='maximize', study_name='xgb')
xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
xgb_best = xgb_study.best_params
xgb_best_score = xgb_study.best_value
print(f'XGB best AUC: {xgb_best_score:.5f} ({time.time()-t0:.0f}s)')
print(f'Best params: {xgb_best}')
wandb.log({'xgb_optuna_best_auc': xgb_best_score, 'xgb_optuna_params': xgb_best})"""
)

# ── Optuna HPO for CatBoost ──
add_code(
    """def cat_objective(trial):
    params = {
        'iterations': 2000, 'eval_metric': 'AUC', 'verbose': 0,
        'early_stopping_rounds': 50, 'task_type': 'GPU',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
    }
    skf = StratifiedKFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=HPO_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]))
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing CatBoost...')
t0 = time.time()
cat_study = optuna.create_study(direction='maximize', study_name='cat')
cat_study.optimize(cat_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
cat_best = cat_study.best_params
cat_best_score = cat_study.best_value
print(f'CAT best AUC: {cat_best_score:.5f} ({time.time()-t0:.0f}s)')
print(f'Best params: {cat_best}')
wandb.log({'cat_optuna_best_auc': cat_best_score, 'cat_optuna_params': cat_best})

print('\\n=== Optuna Summary ===')
print(f'LGB: {lgb_best_score:.5f}')
print(f'XGB: {xgb_best_score:.5f}')
print(f'CAT: {cat_best_score:.5f}')"""
)

# ── Multi-seed training with tuned params ──
add_code(
    """# ========================================
# Multi-Seed Training with Optuna-Tuned Params
# ========================================
SEEDS = [42, 123, 2024, 7, 999]
N_SPLITS = 10

def train_model_multiseed(model_type, best_params, seeds, n_splits):
    all_oof = []
    all_preds = []
    all_importances = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        preds = np.zeros(len(X_test))
        imp = np.zeros(X.shape[1])

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            if model_type == 'lgb':
                params = {
                    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                    'n_estimators': 3000, 'device': 'gpu', 'random_state': seed,
                    **best_params,
                }
                model = lgb.LGBMClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                imp += model.feature_importances_ / n_splits

            elif model_type == 'xgb':
                params = {
                    'objective': 'binary:logistic', 'eval_metric': 'auc',
                    'n_estimators': 3000, 'early_stopping_rounds': 100,
                    'verbosity': 0, 'tree_method': 'hist', 'device': 'cuda',
                    'random_state': seed, **best_params,
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
                imp += model.feature_importances_ / n_splits

            elif model_type == 'cat':
                params = {
                    'iterations': 3000, 'eval_metric': 'AUC', 'verbose': 0,
                    'early_stopping_rounds': 100, 'task_type': 'GPU',
                    'random_seed': seed, **best_params,
                }
                model = CatBoostClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]))
                imp += model.feature_importances_ / n_splits

            oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
            preds += model.predict_proba(X_test)[:, 1] / n_splits

        seed_auc = roc_auc_score(y, oof)
        print(f'  {model_type.upper()} seed={seed}: CV AUC = {seed_auc:.5f}')
        wandb.log({f'{model_type}_seed_{seed}_auc': seed_auc})
        all_oof.append(oof)
        all_preds.append(preds)
        all_importances.append(imp)
        gc.collect()

    mean_oof = np.mean(all_oof, axis=0)
    mean_preds = np.mean(all_preds, axis=0)
    mean_imp = np.mean(all_importances, axis=0)
    mean_auc = roc_auc_score(y, mean_oof)
    print(f'  >>> {model_type.upper()} Multi-Seed CV AUC: {mean_auc:.5f}')
    return mean_oof, mean_preds, mean_imp, mean_auc"""
)

# ── Run training ──
add_code(
    """print('=' * 60)
print('  LightGBM — Optuna-Tuned Multi-Seed')
print('=' * 60)
lgb_oof, lgb_preds, lgb_imp, lgb_auc = train_model_multiseed('lgb', lgb_best, SEEDS, N_SPLITS)
wandb.log({'lgb_final_auc': lgb_auc})

print()
print('=' * 60)
print('  XGBoost — Optuna-Tuned Multi-Seed')
print('=' * 60)
xgb_oof, xgb_preds, xgb_imp, xgb_auc = train_model_multiseed('xgb', xgb_best, SEEDS, N_SPLITS)
wandb.log({'xgb_final_auc': xgb_auc})

print()
print('=' * 60)
print('  CatBoost — Optuna-Tuned Multi-Seed')
print('=' * 60)
cat_oof, cat_preds, cat_imp, cat_auc = train_model_multiseed('cat', cat_best, SEEDS, N_SPLITS)
wandb.log({'cat_final_auc': cat_auc})"""
)

# ── Ensemble ──
add_code(
    """# ========================================
# Ensemble Methods
# ========================================
print('\\n' + '=' * 60)
print('  Ensemble')
print('=' * 60)

# --- Simple Average ---
avg_oof = (lgb_oof + xgb_oof + cat_oof) / 3
avg_preds = (lgb_preds + xgb_preds + cat_preds) / 3
avg_auc = roc_auc_score(y, avg_oof)
print(f'Simple Average:   AUC = {avg_auc:.5f}')

# --- Rank Average ---
def rank_average(*preds_list):
    ranks = [rankdata(p) for p in preds_list]
    return np.mean(ranks, axis=0)

rank_oof = rank_average(lgb_oof, xgb_oof, cat_oof)
rank_preds = rank_average(lgb_preds, xgb_preds, cat_preds)
rank_auc = roc_auc_score(y, rank_oof)
print(f'Rank Average:     AUC = {rank_auc:.5f}')

# --- Weighted Average (based on individual AUC) ---
w_lgb = lgb_auc ** 4
w_xgb = xgb_auc ** 4
w_cat = cat_auc ** 4
w_sum = w_lgb + w_xgb + w_cat
weighted_oof = (w_lgb * lgb_oof + w_xgb * xgb_oof + w_cat * cat_oof) / w_sum
weighted_preds = (w_lgb * lgb_preds + w_xgb * xgb_preds + w_cat * cat_preds) / w_sum
weighted_auc = roc_auc_score(y, weighted_oof)
print(f'Weighted Average: AUC = {weighted_auc:.5f} (weights: LGB={w_lgb/w_sum:.3f}, XGB={w_xgb/w_sum:.3f}, CAT={w_cat/w_sum:.3f})')

# --- Stacking: LR + Ridge meta-learners ---
stack_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
stack_test = np.column_stack([lgb_preds, xgb_preds, cat_preds])

# LR stacking
lr_stack_oof = np.zeros(len(y))
lr_stack_preds = np.zeros(len(X_test))
skf_meta = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for tr_idx, va_idx in skf_meta.split(stack_train, y):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(stack_train[tr_idx], y[tr_idx])
    lr_stack_oof[va_idx] = meta.predict_proba(stack_train[va_idx])[:, 1]
    lr_stack_preds += meta.predict_proba(stack_test)[:, 1] / 10
lr_stack_auc = roc_auc_score(y, lr_stack_oof)
print(f'Stacking (LR):    AUC = {lr_stack_auc:.5f}')

# --- Rank Average of Stacking + Individual ---
final_oof = rank_average(lgb_oof, xgb_oof, cat_oof, lr_stack_oof)
final_preds = rank_average(lgb_preds, xgb_preds, cat_preds, lr_stack_preds)
final_auc = roc_auc_score(y, final_oof)
print(f'Rank Avg (all 4): AUC = {final_auc:.5f}')

# --- Pick best ---
results = {
    'LightGBM': (lgb_auc, lgb_preds),
    'XGBoost': (xgb_auc, xgb_preds),
    'CatBoost': (cat_auc, cat_preds),
    'Simple Avg': (avg_auc, avg_preds),
    'Rank Avg (3)': (rank_auc, rank_preds),
    'Weighted Avg': (weighted_auc, weighted_preds),
    'Stacking (LR)': (lr_stack_auc, lr_stack_preds),
    'Rank Avg (all 4)': (final_auc, final_preds),
}
best_name = max(results, key=lambda k: results[k][0])
best_auc, best_preds = results[best_name]
print(f'\\n*** Best: {best_name} (CV AUC = {best_auc:.5f}) ***')

for name, (auc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    wandb.log({f'ensemble_{name}_auc': auc})
wandb.log({'best_method': best_name, 'best_auc': best_auc})"""
)

# ── ROC + Feature Importance plots ──
add_code(
    """fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#e67e22', '#9b59b6', '#1abc9c', '#3498db', '#e74c3c', '#95a5a6', '#2c3e50', '#f39c12']
plot_data = [
    ('LightGBM', lgb_oof), ('XGBoost', xgb_oof), ('CatBoost', cat_oof),
    ('Rank Avg', rank_oof), ('Stacking', lr_stack_oof), ('Rank All4', final_oof),
]
for (name, oof_data), color in zip(plot_data, colors):
    fpr, tpr, _ = roc_curve(y, oof_data)
    auc_val = roc_auc_score(y, oof_data)
    lw = 3 if name == best_name or 'All4' in name else 1.5
    axes[0].plot(fpr, tpr, label=f'{name} ({auc_val:.4f})', linewidth=lw, color=color)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('FPR', fontsize=14)
axes[0].set_ylabel('TPR', fontsize=14)
axes[0].set_title('ROC Curves', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].tick_params(labelsize=12)

# Feature importance (LGB)
imp_df = pd.Series(lgb_imp, index=feature_cols).sort_values(ascending=True).tail(25)
imp_df.plot(kind='barh', ax=axes[1], color='#3498db')
axes[1].set_title('Top 25 Feature Importance (LGB)', fontsize=16, fontweight='bold')
axes[1].tick_params(labelsize=10)
plt.tight_layout()
plt.show()"""
)

# ── Submission ──
add_code(
    """# Prediction distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(best_preds, bins=60, alpha=0.7, color='#3498db', edgecolor='white')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Predicted Probability', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'Test Predictions — {best_name}', fontsize=16, fontweight='bold')
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()

print(f'Range: [{best_preds.min():.5f}, {best_preds.max():.5f}]')
print(f'Mean:  {best_preds.mean():.5f}')"""
)

add_code(
    """submission[TARGET] = best_preds
submission.to_csv('submission.csv', index=False)
print(f'Submission saved: {submission.shape}')
print(f'Method: {best_name} (CV AUC = {best_auc:.5f})')
print(f'\\nFinal Results:')
for name, (auc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = ' <<<' if name == best_name else ''
    print(f'  {name:25s} AUC: {auc:.5f}{marker}')

submission.head(10)"""
)

# ── W&B finish ──
add_code(
    """wandb.log({'submission_method': best_name, 'submission_auc': best_auc})
wandb.finish()
print('Done. Sync: kaggle-wandb-sync run . --kernel-id yasunorim/s6e3-churn-optuna-stacking-work')"""
)


# ── Build notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "s6e3-churn-optuna-stacking-work.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path} ({len(cells)} cells)")

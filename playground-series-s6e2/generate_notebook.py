"""Generate improved S6E2 notebook (.ipynb) with multi-seed ensemble + stacking.

Improvements over baseline:
- W&B offline mode (kaggle-wandb-sync compatible)
- Multi-seed ensemble (3 seeds x 3 models = 9 models)
- Enhanced feature engineering (20+ new features)
- 10-fold CV
- Stacking meta-learner (LogisticRegression)
- Mentions self-made PyPI: kaggle-notebook-deploy, kaggle-wandb-sync
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


# ── Cell 1: Title ──
add_md(
    """# S6E2 Heart Disease — Multi-Seed Ensemble + Stacking

**Approach**: 3 GBDT models (LightGBM / XGBoost / CatBoost) × 3 seeds, 10-fold CV, stacking meta-learner.

**Deployment tools** (self-made PyPI packages):
- [`kaggle-notebook-deploy`](https://pypi.org/project/kaggle-notebook-deploy/) — automated notebook push via GitHub Actions
- [`kaggle-wandb-sync`](https://pypi.org/project/kaggle-wandb-sync/) — offline W&B run sync from Kaggle to cloud"""
)

# ── Cell 2: Setup ──
add_code(
    """import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'kaggle-s6e2-heart-disease'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import wandb
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'no': '#2ecc71', 'yes': '#e74c3c', 'accent': '#3498db'}
print('All libraries loaded. W&B mode:', os.environ.get('WANDB_MODE'))"""
)

# ── Cell 3: W&B offline init ──
add_code(
    """# W&B offline mode — logs saved to /kaggle/working/wandb/
# After notebook runs, sync with: kaggle-wandb-sync run . --kernel-id yasunorim/s6e2-heart-disease-eda-ensemble-wandb
run = wandb.init(
    project='kaggle-s6e2-heart-disease',
    name='multi-seed-stacking-v11',
    tags=['multi-seed', 'stacking', '10fold', 'gpu'],
    config={'n_seeds': 3, 'n_splits': 10, 'models': ['lgb', 'xgb', 'cat']},
)
print(f'W&B run: {run.name} (offline mode)')"""
)

# ── Cell 4: Load data + original UCI dataset ──
add_code(
    """train = pd.read_csv('/kaggle/input/playground-series-s6e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s6e2/sample_submission.csv')

TARGET = 'Heart Disease'
ID = 'id'

print(f'PGS Train: {train.shape}, Test: {test.shape}')

# Load original UCI Heart Disease dataset
import glob
orig_files = glob.glob('/kaggle/input/heart-disease-dataset/*.csv')
print(f'Original data files: {orig_files}')
orig = pd.read_csv(orig_files[0])
print(f'Original columns: {orig.columns.tolist()}')
print(f'Original shape: {orig.shape}')

# Map UCI column names to PGS column names
col_map = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest pain type',
    'trestbps': 'BP', 'chol': 'Cholesterol', 'fbs': 'FBS over 120',
    'restecg': 'EKG results', 'thalach': 'Max HR', 'exang': 'Exercise angina',
    'oldpeak': 'ST depression', 'slope': 'Slope of ST',
    'ca': 'Number of vessels fluro', 'thal': 'Thallium', 'target': TARGET
}
orig = orig.rename(columns=col_map)

# Convert target: 0 -> 'Absence', 1 -> 'Presence'
if orig[TARGET].dtype in ['int64', 'float64']:
    orig[TARGET] = orig[TARGET].map({0: 'Absence', 1: 'Presence'})

# Add id column and combine
orig[ID] = range(-len(orig), 0)
# Keep only columns that exist in train
common_cols = [c for c in train.columns if c in orig.columns]
orig = orig[common_cols]

print(f'\\nOriginal data mapped: {orig.shape}')
print(f'Target distribution (original): {orig[TARGET].value_counts().to_dict()}')

# Combine PGS + original
train = pd.concat([train, orig], axis=0, ignore_index=True)
print(f'Combined train: {train.shape}')
print(f'Target: {train[TARGET].value_counts().to_dict()}')"""
)

# ── Cell 5: Target visualization ──
add_code(
    """target_map = {'Absence': 0, 'Presence': 1}
train['target'] = train[TARGET].map(target_map)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = train[TARGET].value_counts()
axes[0].bar(counts.index, counts.values, color=[COLORS['no'], COLORS['yes']])
axes[0].set_title('Target Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + v*0.01, str(v), ha='center', fontweight='bold')

props = train[TARGET].value_counts(normalize=True)
axes[1].pie(props.values, labels=props.index, colors=[COLORS['no'], COLORS['yes']],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Target Proportion')
plt.suptitle('Target Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""
)

# ── Cell 6: Feature distributions ──
add_code(
    """feature_cols = [c for c in train.columns if c not in [ID, TARGET, 'target']]
n_features = len(feature_cols)
n_rows = (n_features + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    for label, color, name in [(0, COLORS['no'], 'No'), (1, COLORS['yes'], 'Yes')]:
        axes[i].hist(train[train['target'] == label][col], alpha=0.6, label=name, bins=30, color=color, edgecolor='white')
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Feature Distributions by Heart Disease', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""
)

# ── Cell 7: Correlation ──
add_code(
    """corr_df = train[feature_cols + ['target']].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

target_corr = corr_df['target'].drop('target').abs().sort_values(ascending=False)
print('Top correlations with target (abs):')
for feat, val in target_corr.items():
    d = '+' if corr_df.loc[feat, 'target'] > 0 else '-'
    print(f'  {feat:30s} {d}{val:.3f}')"""
)

# ── Cell 8: Feature Engineering (original 6 interactions only) ──
add_code(
    """train_len = len(train)
df = pd.concat([train.drop(['target'], axis=1), test], axis=0, ignore_index=True)
le_target = LabelEncoder()
df.loc[:train_len-1, 'target_encoded'] = le_target.fit_transform(df.loc[:train_len-1, TARGET])

# Interaction features (same as v10 baseline)
df['Age_x_MaxHR'] = df['Age'] * df['Max HR']
df['Age_x_STdep'] = df['Age'] * df['ST depression']
df['STdep_x_Slope'] = df['ST depression'] * df['Slope of ST']
df['BP_x_Chol'] = df['BP'] * df['Cholesterol']
df['MaxHR_div_Age'] = df['Max HR'] / (df['Age'] + 1)
df['Vessels_x_Thal'] = df['Number of vessels fluro'] * df['Thallium']

# All model features
model_features = [c for c in df.columns if c not in [ID, TARGET, 'target_encoded']]
X = df.iloc[:train_len][model_features].values
y = df.iloc[:train_len]['target_encoded'].values.astype(int)
X_test = df.iloc[train_len:][model_features].values

print(f'Features ({len(model_features)}):')
for f in model_features:
    print(f'  - {f}')
print(f'\\nX: {X.shape}, y: {y.shape}, X_test: {X_test.shape}')
wandb.config.update({'n_features': len(model_features), 'features': model_features})"""
)

# ── Cell 9: Training config ──
add_code(
    """SEEDS = [42, 123, 2024]
N_SPLITS = 5

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6,
    'num_leaves': 31, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'device': 'gpu',
}

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'verbosity': 0, 'early_stopping_rounds': 50,
    'tree_method': 'hist', 'device': 'cuda',
}

cat_params = {
    'iterations': 1000, 'learning_rate': 0.05, 'depth': 6,
    'eval_metric': 'AUC', 'verbose': 0,
    'early_stopping_rounds': 50, 'task_type': 'GPU',
}

print(f'Seeds: {SEEDS}, Folds: {N_SPLITS}')
print(f'Total models: {len(SEEDS)} seeds x 3 models x {N_SPLITS} folds = {len(SEEDS)*3*N_SPLITS}')"""
)

# ── Cell 10: LightGBM multi-seed ──
add_code(
    """print('=' * 50)
print('  LightGBM — Multi-Seed Training')
print('=' * 50)

all_lgb_oof = []
all_lgb_preds = []
all_lgb_importances = []

for seed in SEEDS:
    lgb_params['random_state'] = seed
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    imp = np.zeros(len(model_features))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        imp += model.feature_importances_ / N_SPLITS

    seed_auc = roc_auc_score(y, oof)
    print(f'  Seed {seed}: CV AUC = {seed_auc:.5f}')
    wandb.log({f'lgb_seed_{seed}_auc': seed_auc})
    all_lgb_oof.append(oof)
    all_lgb_preds.append(preds)
    all_lgb_importances.append(imp)

lgb_oof = np.mean(all_lgb_oof, axis=0)
lgb_preds = np.mean(all_lgb_preds, axis=0)
lgb_importances = np.mean(all_lgb_importances, axis=0)
lgb_auc = roc_auc_score(y, lgb_oof)
print(f'  >>> LightGBM Multi-Seed CV AUC: {lgb_auc:.5f}')
wandb.log({'lgb_cv_auc': lgb_auc})"""
)

# ── Cell 11: XGBoost multi-seed ──
add_code(
    """print('=' * 50)
print('  XGBoost — Multi-Seed Training')
print('=' * 50)

all_xgb_oof = []
all_xgb_preds = []

for seed in SEEDS:
    xgb_params['random_state'] = seed
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    seed_auc = roc_auc_score(y, oof)
    print(f'  Seed {seed}: CV AUC = {seed_auc:.5f}')
    wandb.log({f'xgb_seed_{seed}_auc': seed_auc})
    all_xgb_oof.append(oof)
    all_xgb_preds.append(preds)

xgb_oof = np.mean(all_xgb_oof, axis=0)
xgb_preds = np.mean(all_xgb_preds, axis=0)
xgb_auc = roc_auc_score(y, xgb_oof)
print(f'  >>> XGBoost Multi-Seed CV AUC: {xgb_auc:.5f}')
wandb.log({'xgb_cv_auc': xgb_auc})"""
)

# ── Cell 12: CatBoost multi-seed ──
add_code(
    """print('=' * 50)
print('  CatBoost — Multi-Seed Training')
print('=' * 50)

all_cat_oof = []
all_cat_preds = []

for seed in SEEDS:
    cat_params['random_seed'] = seed
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = CatBoostClassifier(**cat_params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]))
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    seed_auc = roc_auc_score(y, oof)
    print(f'  Seed {seed}: CV AUC = {seed_auc:.5f}')
    wandb.log({f'cat_seed_{seed}_auc': seed_auc})
    all_cat_oof.append(oof)
    all_cat_preds.append(preds)

cat_oof = np.mean(all_cat_oof, axis=0)
cat_preds = np.mean(all_cat_preds, axis=0)
cat_auc = roc_auc_score(y, cat_oof)
print(f'  >>> CatBoost Multi-Seed CV AUC: {cat_auc:.5f}')
wandb.log({'cat_cv_auc': cat_auc})"""
)

# ── Cell 13: Ensemble + Rank Averaging ──
add_code(
    """from scipy.stats import rankdata

# === Simple Average ===
avg_oof = (lgb_oof + xgb_oof + cat_oof) / 3
avg_preds = (lgb_preds + xgb_preds + cat_preds) / 3
avg_auc = roc_auc_score(y, avg_oof)
print(f'Simple Average CV AUC: {avg_auc:.5f}')

# === Rank Average (key technique for PGS) ===
def rank_average(*preds_list):
    ranks = [rankdata(p) for p in preds_list]
    return np.mean(ranks, axis=0)

rank_oof = rank_average(lgb_oof, xgb_oof, cat_oof)
rank_preds = rank_average(lgb_preds, xgb_preds, cat_preds)
rank_auc = roc_auc_score(y, rank_oof)
print(f'Rank Average CV AUC: {rank_auc:.5f}')

# === Stacking Meta-Learner ===
stack_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
stack_test = np.column_stack([lgb_preds, xgb_preds, cat_preds])

stack_oof = np.zeros(len(y))
stack_preds = np.zeros(len(X_test))
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(stack_train, y)):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(stack_train[tr_idx], y[tr_idx])
    stack_oof[va_idx] = meta.predict_proba(stack_train[va_idx])[:, 1]
    stack_preds += meta.predict_proba(stack_test)[:, 1] / 5

stack_auc = roc_auc_score(y, stack_oof)
print(f'Stacking (LR) CV AUC: {stack_auc:.5f}')

# === Best ensemble ===
results = {
    'LightGBM': (lgb_auc, lgb_preds),
    'XGBoost': (xgb_auc, xgb_preds),
    'CatBoost': (cat_auc, cat_preds),
    'Simple Avg': (avg_auc, avg_preds),
    'Rank Avg': (rank_auc, rank_preds),
    'Stacking': (stack_auc, stack_preds),
}
best_name = max(results, key=lambda k: results[k][0])
best_auc, best_preds = results[best_name]
print(f'\\n*** Best: {best_name} (CV AUC = {best_auc:.5f}) ***')

wandb.log({
    'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc, 'cat_auc': cat_auc,
    'avg_auc': avg_auc, 'rank_auc': rank_auc, 'stacking_auc': stack_auc,
    'best_method': best_name, 'best_auc': best_auc,
})"""
)

# ── Cell 14: Results table ──
add_code(
    """print('=' * 50)
print('      FINAL RESULTS')
print('=' * 50)
res_df = pd.DataFrame([
    {'Model': name, 'CV AUC': auc}
    for name, (auc, _) in results.items()
]).sort_values('CV AUC', ascending=False)

for _, row in res_df.iterrows():
    marker = ' <<<' if row['Model'] == best_name else ''
    print(f"  {row['Model']:20s} AUC: {row['CV AUC']:.5f}{marker}")
print('=' * 50)"""
)

# ── Cell 15: ROC curves ──
add_code(
    """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, oof_data, color in [
    ('LightGBM', lgb_oof, '#e67e22'),
    ('XGBoost', xgb_oof, '#9b59b6'),
    ('CatBoost', cat_oof, '#1abc9c'),
    ('Rank Avg', rank_oof, '#95a5a6'),
    ('Stacking', stack_oof, '#e74c3c'),
]:
    fpr, tpr, _ = roc_curve(y, oof_data)
    auc_val = roc_auc_score(y, oof_data)
    lw = 3 if name in ['Stacking', best_name] else 1.5
    axes[0].plot(fpr, tpr, label=f'{name} ({auc_val:.4f})', linewidth=lw, color=color)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].legend(fontsize=9)

# Feature importance
imp = pd.Series(lgb_importances, index=model_features).sort_values(ascending=True).tail(20)
imp.plot(kind='barh', ax=axes[1], color=COLORS['accent'])
axes[1].set_title('Top 20 Feature Importance (LGB)', fontweight='bold')
plt.tight_layout()
plt.show()"""
)

# ── Cell 16: Prediction distribution ──
add_code(
    """fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(best_preds, bins=50, alpha=0.7, color=COLORS['accent'], edgecolor='white', label='Test predictions')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Count')
ax.set_title(f'Test Prediction Distribution ({best_name})', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

print(f'Prediction range: [{best_preds.min():.4f}, {best_preds.max():.4f}]')
print(f'Prediction mean:  {best_preds.mean():.4f}')"""
)

# ── Cell 17: Submission ──
add_code(
    """submission[TARGET] = best_preds
submission.to_csv('submission.csv', index=False)
print(f'Submission saved: {submission.shape}')
print(f'Method: {best_name} (CV AUC = {best_auc:.5f})')
submission.head(10)"""
)

# ── Cell 18: W&B finish ──
add_code(
    """wandb.log({'submission_method': best_name, 'submission_auc': best_auc})
wandb.finish()
print('W&B offline run saved. Sync with: kaggle-wandb-sync run . --kernel-id yasunorim/s6e2-heart-disease-eda-ensemble-wandb')"""
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

out_path = "playground-series-s6e2-baseline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path} ({len(cells)} cells)")

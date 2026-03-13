"""Generate S6E3 Customer Churn v2 notebook.

V2 improvements over V1 (LB 0.91400):
- Original data concatenation (blastchar/telco-customer-churn)
- Adversarial Validation with sample weighting
- Pseudo Labeling (high-confidence test predictions)
- Enhanced Feature Engineering: WoE, category cross features, domain features, binning, permutation importance
- 5 models: LGB + XGB + Cat + ExtraTrees + HistGradientBoosting
- Optuna 100 trials (LGB/XGB/Cat), 50 trials (ET/HGBC)
- Level-2 stacking: LR + Ridge + LGB meta-learner + Optuna weight search
- Rank averaging of all ensemble methods
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
    """# S6E3 Predict Customer Churn v2 — Original Data + Adversarial + Pseudo Label + 5-Model Stacking

**Approach**: Original data concat + Adversarial Validation weighting + Pseudo Labeling + WoE/cross features + Optuna HPO (5 models) + Multi-seed 10-fold + Stacking + Rank Avg

**Models**: LightGBM / XGBoost / CatBoost (GPU) + ExtraTrees / HistGradientBoosting (CPU)

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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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

# Auto-detect GPU availability
import subprocess
try:
    subprocess.run(['nvidia-smi'], capture_output=True, check=True)
    DEVICE = 'gpu'
    CATBOOST_TASK = 'GPU'
except Exception:
    DEVICE = 'cpu'
    CATBOOST_TASK = 'CPU'
print(f'Device: {DEVICE}')
print('All libraries loaded.')"""
)

# ── W&B init ──
add_code(
    """run = wandb.init(
    project='kaggle-s6e3-churn',
    name='optuna-stacking-v2',
    tags=['optuna', 'stacking', 'multi-seed', '10fold', 'gpu', 'v2',
          'original-data', 'adversarial', 'pseudo-label', '5-model'],
    config={
        'n_seeds': 5, 'n_splits': 10,
        'models': ['lgb', 'xgb', 'cat', 'et', 'hgbc'],
        'optuna_trials_gbdt': 100, 'optuna_trials_sklearn': 50,
        'strategy': 'original_data + adversarial_val + pseudo_label + 5model_stacking',
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

# Identify target and ID columns
TARGET = submission.columns[-1]
ID = submission.columns[0]
print(f'\\nTARGET: {TARGET}, ID: {ID}')
print(f'Target distribution:\\n{train[TARGET].value_counts()}')
print(f'\\nMissing values:\\n{train.isnull().sum()[train.isnull().sum() > 0]}')"""
)

# ── Load and merge original data ──
add_code(
    """# ========================================
# Load Original Data (blastchar/telco-customer-churn)
# ========================================
_orig_slug = 'telco-customer-churn'
_orig_matches = _glob.glob(f'/kaggle/input/**/{_orig_slug}', recursive=True)
ORIG_DIR = _orig_matches[0] if _orig_matches else f'/kaggle/input/{_orig_slug}'
print(f'ORIG_DIR: {ORIG_DIR}')

# Find the CSV file in original dataset
import os as _os
orig_files = [f for f in _os.listdir(ORIG_DIR) if f.endswith('.csv')]
print(f'Original dataset files: {orig_files}')

# Load original data
orig = pd.read_csv(f'{ORIG_DIR}/{orig_files[0]}')
print(f'Original data shape: {orig.shape}')
print(f'Original columns: {orig.columns.tolist()}')
print(f'Original dtypes:\\n{orig.dtypes}')

# Align column names between original and competition data
# The original dataset may have different column names or casing
orig_cols_lower = {c.lower().replace(' ', ''): c for c in orig.columns}
train_cols_lower = {c.lower().replace(' ', ''): c for c in train.columns}

print(f'\\nColumn mapping check:')
for tc_lower, tc in train_cols_lower.items():
    if tc_lower in orig_cols_lower:
        oc = orig_cols_lower[tc_lower]
        print(f'  {tc} <-> {oc}')
    else:
        print(f'  {tc} -> NOT FOUND in original')

# Rename original columns to match competition columns
rename_map = {}
for tc_lower, tc in train_cols_lower.items():
    if tc_lower in orig_cols_lower:
        oc = orig_cols_lower[tc_lower]
        if oc != tc:
            rename_map[oc] = tc
orig = orig.rename(columns=rename_map)
print(f'\\nRenamed columns: {rename_map}')

# Keep only columns present in both datasets
common_cols = [c for c in train.columns if c in orig.columns]
missing_in_orig = [c for c in train.columns if c not in orig.columns]
print(f'Common columns: {len(common_cols)}')
print(f'Missing in original: {missing_in_orig}')

# Fix target column if needed (original may have Yes/No, competition may have 1/0)
if TARGET in orig.columns:
    print(f'\\nOriginal target unique: {orig[TARGET].unique()[:10]}')
    print(f'Train target unique: {train[TARGET].unique()[:10]}')

    # Map target values to match competition format
    if orig[TARGET].dtype == 'object' and train[TARGET].dtype != 'object':
        target_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        orig[TARGET] = orig[TARGET].map(target_map)
        print(f'Mapped original target: {orig[TARGET].value_counts().to_dict()}')
    elif train[TARGET].dtype == 'object' and orig[TARGET].dtype != 'object':
        pass  # will handle below

# Handle TotalCharges (common issue: blank strings in original)
if 'TotalCharges' in orig.columns:
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    n_null = orig['TotalCharges'].isnull().sum()
    if n_null > 0:
        orig['TotalCharges'] = orig['TotalCharges'].fillna(orig['TotalCharges'].median())
        print(f'Fixed {n_null} null TotalCharges in original data')

# Ensure dtypes match
for col in common_cols:
    if col in [ID, TARGET]:
        continue
    if train[col].dtype != orig[col].dtype:
        try:
            if train[col].dtype == 'object':
                orig[col] = orig[col].astype(str)
            else:
                orig[col] = pd.to_numeric(orig[col], errors='coerce')
        except Exception as e:
            print(f'  dtype mismatch for {col}: {e}')

# Add ID column to original if missing
if ID not in orig.columns:
    orig[ID] = range(len(train) + len(test) + 1, len(train) + len(test) + 1 + len(orig))

# Only keep common columns
orig = orig[[c for c in common_cols if c in orig.columns]]

# Drop rows with NaN target
orig = orig.dropna(subset=[TARGET])

print(f'\\nOriginal data after alignment: {orig.shape}')
print(f'Original target distribution: {orig[TARGET].value_counts().to_dict()}')

# Concatenate original data to training data
train_len_before = len(train)
train = pd.concat([train, orig], axis=0, ignore_index=True)
print(f'\\nTrain BEFORE concat: {train_len_before}')
print(f'Train AFTER concat: {len(train)}')
print(f'Added {len(orig)} rows from original dataset')
print(f'Combined target distribution:\\n{train[TARGET].value_counts()}')
wandb.log({'train_size_original': train_len_before, 'train_size_with_orig': len(train),
           'original_data_added': len(orig)})"""
)

# ── EDA ──
add_code(
    """# Identify categorical vs numerical
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in [ID, TARGET]]
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [ID, TARGET]]
print(f'Categorical ({len(cat_cols)}): {cat_cols}')
print(f'Numerical ({len(num_cols)}): {num_cols}')

# Target encoding for analysis
if train[TARGET].dtype == 'object':
    target_le = LabelEncoder()
    y_all = target_le.fit_transform(train[TARGET])
    print(f'Target classes: {list(target_le.classes_)}')
else:
    y_all = train[TARGET].values.astype(int)

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = train[TARGET].value_counts()
axes[0].bar(range(len(counts)), counts.values, tick_label=counts.index.astype(str))
axes[0].set_title('Target Count (Train + Original)', fontsize=16, fontweight='bold')
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

# ── EDA distributions ──
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

# ── Correlation ──
add_code(
    """# Correlation with target
train_corr = train.copy()
if train_corr[TARGET].dtype == 'object':
    train_corr['_target_num'] = y_all
else:
    train_corr['_target_num'] = train_corr[TARGET]

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
# Feature Engineering v2 — go heavy
# ========================================
train_len = len(train)
df = pd.concat([train, test], axis=0, ignore_index=True)

# --- 1. Label encode categoricals ---
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- 2. WoE (Weight of Evidence) encoding for categoricals ---
# WoE = ln(Distribution of Events / Distribution of Non-Events)
for col in cat_cols:
    woe_col = f'{col}_woe'
    df[woe_col] = 0.0
    train_df_woe = df.iloc[:train_len].copy()
    train_df_woe['_y'] = y_all

    skf_woe = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf_woe.split(train_df_woe, y_all):
        tr_part = train_df_woe.iloc[tr_idx]
        total_events = tr_part['_y'].sum()
        total_non_events = len(tr_part) - total_events
        if total_events == 0 or total_non_events == 0:
            continue

        for cat_val in tr_part[col].unique():
            mask_cat = tr_part[col] == cat_val
            events = tr_part.loc[mask_cat, '_y'].sum()
            non_events = mask_cat.sum() - events
            # Laplace smoothing
            dist_events = (events + 0.5) / (total_events + 1)
            dist_non_events = (non_events + 0.5) / (total_non_events + 1)
            woe_val = np.log(dist_events / dist_non_events)
            df.loc[df.index[va_idx][train_df_woe.iloc[va_idx][col] == cat_val], woe_col] = woe_val

    # Test: use full train WoE
    total_events = y_all.sum()
    total_non_events = len(y_all) - total_events
    for cat_val in train_df_woe[col].unique():
        mask_cat = train_df_woe[col] == cat_val
        events = train_df_woe.loc[mask_cat, '_y'].sum()
        non_events = mask_cat.sum() - events
        dist_events = (events + 0.5) / (total_events + 1)
        dist_non_events = (non_events + 0.5) / (total_non_events + 1)
        woe_val = np.log(dist_events / dist_non_events)
        test_mask = (df.index >= train_len) & (df[col] == cat_val)
        df.loc[test_mask, woe_col] = woe_val

print(f'After WoE encoding: {df.shape}')

# --- 3. Target encode categoricals (CV-aware, train only, mean for test) ---
for col in cat_cols:
    te_col = f'{col}_te'
    df[te_col] = np.nan
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_target = y_all
    for tr_idx, va_idx in skf_te.split(df.iloc[:train_len], train_target):
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
    df[te_col] = df[te_col].fillna(train_target.mean())

print(f'After target encoding: {df.shape}')

# --- 4. Category x Category cross features (all pairs) ---
cross_count = 0
for i in range(len(cat_cols)):
    for j in range(i + 1, len(cat_cols)):
        c1, c2 = cat_cols[i], cat_cols[j]
        cross_col = f'{c1}_x_{c2}'
        df[cross_col] = df[c1].astype(str) + '_' + df[c2].astype(str)
        le_cross = LabelEncoder()
        df[cross_col] = le_cross.fit_transform(df[cross_col])
        # Frequency encoding for cross feature
        freq = df[cross_col].value_counts(normalize=True)
        df[f'{cross_col}_freq'] = df[cross_col].map(freq)
        cross_count += 2
print(f'Category cross features added: {cross_count}')

# --- 5. Domain-specific features ---
domain_count = 0
# Detect column names (case-insensitive matching)
col_map = {c.lower(): c for c in df.columns}

tenure_col = col_map.get('tenure')
mc_col = col_map.get('monthlycharges')
tc_col = col_map.get('totalcharges')
contract_col = col_map.get('contract')

if tenure_col and mc_col:
    df['tenure_x_monthly'] = df[tenure_col] * df[mc_col]
    domain_count += 1
if tc_col and tenure_col:
    df['avg_monthly_from_total'] = df[tc_col] / (df[tenure_col] + 1)
    domain_count += 1
if mc_col and tc_col:
    df['monthly_to_total_ratio'] = df[mc_col] / (df[tc_col] + 1e-8)
    domain_count += 1
if tenure_col:
    df['tenure_sq'] = df[tenure_col] ** 2
    df['tenure_log'] = np.log1p(df[tenure_col])
    df['is_new_customer'] = (df[tenure_col] <= 6).astype(int)
    df['is_loyal_customer'] = (df[tenure_col] >= 48).astype(int)
    domain_count += 4
if mc_col:
    df['monthly_charges_sq'] = df[mc_col] ** 2
    df['monthly_charges_log'] = np.log1p(df[mc_col])
    domain_count += 2
if tenure_col and contract_col:
    df['tenure_x_contract'] = df[tenure_col] * df[contract_col]
    domain_count += 1
if mc_col and contract_col:
    df['monthly_x_contract'] = df[mc_col] * df[contract_col]
    domain_count += 1
print(f'Domain-specific features added: {domain_count}')

# --- 6. Numerical interactions (top correlated pairs) ---
if len(num_cols) >= 2:
    top_num = target_corr.head(min(8, len(num_cols))).index.tolist()
    top_num = [c for c in top_num if c in num_cols]
    interaction_count = 0
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
            df[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + 1e-8)
            df[f'{c1}_plus_{c2}'] = df[c1] + df[c2]
            interaction_count += 3
    print(f'Interaction features added: {interaction_count}')

# --- 7. Polynomial features for top correlated ---
if len(num_cols) >= 1:
    top3 = target_corr.head(min(3, len(num_cols))).index.tolist()
    top3 = [c for c in top3 if c in num_cols]
    for col in top3:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log1p'] = np.log1p(np.abs(df[col]))
    print(f'Polynomial features for: {top3}')

# --- 8. Frequency encoding ---
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)

# --- 9. Aggregate features (mean/std per categorical group) ---
for cat_col in cat_cols[:4]:
    for num_col in num_cols[:5]:
        grp = df.groupby(cat_col)[num_col]
        df[f'{num_col}_mean_by_{cat_col}'] = grp.transform('mean')
        df[f'{num_col}_std_by_{cat_col}'] = grp.transform('std').fillna(0)

# --- 10. Binning of numerical features ---
for col in num_cols:
    try:
        df[f'{col}_bin10'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
    except Exception:
        df[f'{col}_bin10'] = pd.cut(df[col], bins=10, labels=False)

print(f'\\nTotal feature count: {df.shape[1]}')

# --- Build X, y, X_test ---
drop_cols = [ID, TARGET]
if '_y' in df.columns:
    drop_cols.append('_y')
feature_cols = [c for c in df.columns if c not in drop_cols]
# Convert all to float32, handle any remaining non-numeric
for col in feature_cols:
    if df[col].dtype == 'object':
        le_tmp = LabelEncoder()
        df[col] = le_tmp.fit_transform(df[col].astype(str))

X = df.iloc[:train_len][feature_cols].values.astype(np.float32)
y = y_all.astype(int)
X_test = df.iloc[train_len:][feature_cols].values.astype(np.float32)

# Handle NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f'X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}')
print(f'y distribution: {np.bincount(y)}')
wandb.config.update({'n_features': len(feature_cols), 'feature_names': feature_cols[:50]})"""
)

# ── Adversarial Validation ──
add_code(
    """# ========================================
# Adversarial Validation
# ========================================
# Train a classifier to distinguish train vs test samples
# Samples that look like test get higher weight, unlike-test get lower weight
print('Running Adversarial Validation...')
t0 = time.time()

# Create adversarial labels: 0=train, 1=test
adv_X = np.vstack([X, X_test])
adv_y = np.concatenate([np.zeros(len(X)), np.ones(len(X_test))])

adv_oof = np.zeros(len(adv_X))
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf_adv.split(adv_X, adv_y)):
    adv_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1, device=DEVICE,
    )
    adv_model.fit(
        adv_X[tr_idx], adv_y[tr_idx],
        eval_set=[(adv_X[va_idx], adv_y[va_idx])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    adv_oof[va_idx] = adv_model.predict_proba(adv_X[va_idx])[:, 1]

adv_auc = roc_auc_score(adv_y, adv_oof)
print(f'Adversarial Validation AUC: {adv_auc:.5f} ({time.time()-t0:.0f}s)')
print(f'  (0.5 = no distribution shift, 1.0 = completely separable)')
wandb.log({'adversarial_auc': adv_auc})

# Compute sample weights for training data
# Higher weight = sample looks more like test (adv_oof closer to 1)
train_adv_scores = adv_oof[:len(X)]

# Weight formula: scale scores so similar-to-test samples get higher weight
# Use rank-based approach to avoid extreme weights
train_ranks = rankdata(train_adv_scores)
sample_weights = train_ranks / train_ranks.max()  # 0 to 1
# Shift to range [0.5, 1.5] to avoid zeroing out any sample
sample_weights = 0.5 + sample_weights

print(f'Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}')

# Show top features distinguishing train vs test
adv_imp = pd.Series(adv_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f'\\nTop 10 features distinguishing train vs test:')
for feat, imp in adv_imp.head(10).items():
    print(f'  {feat:40s} {imp}')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(train_adv_scores, bins=50, alpha=0.7, label='Train adv scores', edgecolor='white')
ax.axvline(0.5, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('P(test)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'Adversarial Validation Scores (AUC={adv_auc:.4f})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()"""
)

# ── Optuna HPO for LightGBM ──
add_code(
    """# ========================================
# Optuna Hyperparameter Optimization
# ========================================
N_OPTUNA_TRIALS_GBDT = 100
N_OPTUNA_TRIALS_SKLEARN = 50
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
        'device': DEVICE,
    }
    skf = StratifiedKFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=HPO_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                  sample_weight=sample_weights[tr_idx],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing LightGBM (100 trials)...')
t0 = time.time()
lgb_study = optuna.create_study(direction='maximize', study_name='lgb')
lgb_study.optimize(lgb_objective, n_trials=N_OPTUNA_TRIALS_GBDT, show_progress_bar=False)
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
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                  sample_weight=sample_weights[tr_idx], verbose=False)
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing XGBoost (100 trials)...')
t0 = time.time()
xgb_study = optuna.create_study(direction='maximize', study_name='xgb')
xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS_GBDT, show_progress_bar=False)
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
        'iterations': 2000, 'eval_metric': 'Logloss', 'verbose': 0,
        'early_stopping_rounds': 50, 'task_type': CATBOOST_TASK,
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
        model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]),
                  sample_weight=sample_weights[tr_idx])
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing CatBoost (100 trials)...')
t0 = time.time()
cat_study = optuna.create_study(direction='maximize', study_name='cat')
cat_study.optimize(cat_objective, n_trials=N_OPTUNA_TRIALS_GBDT, show_progress_bar=False)
cat_best = cat_study.best_params
cat_best_score = cat_study.best_value
print(f'CAT best AUC: {cat_best_score:.5f} ({time.time()-t0:.0f}s)')
print(f'Best params: {cat_best}')
wandb.log({'cat_optuna_best_auc': cat_best_score, 'cat_optuna_params': cat_best})"""
)

# ── Optuna HPO for ExtraTrees ──
add_code(
    """def et_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'n_jobs': -1, 'random_state': HPO_SEED,
    }
    skf = StratifiedKFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=HPO_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = ExtraTreesClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing ExtraTrees (50 trials)...')
t0 = time.time()
et_study = optuna.create_study(direction='maximize', study_name='et')
et_study.optimize(et_objective, n_trials=N_OPTUNA_TRIALS_SKLEARN, show_progress_bar=False)
et_best = et_study.best_params
et_best_score = et_study.best_value
print(f'ET best AUC: {et_best_score:.5f} ({time.time()-t0:.0f}s)')
print(f'Best params: {et_best}')
wandb.log({'et_optuna_best_auc': et_best_score, 'et_optuna_params': et_best})"""
)

# ── Optuna HPO for HistGradientBoosting ──
add_code(
    """def hgbc_objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 127),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 64, 255),
        'early_stopping': True, 'n_iter_no_change': 50,
        'validation_fraction': 0.15,
        'random_state': HPO_SEED,
    }
    skf = StratifiedKFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=HPO_SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = HistGradientBoostingClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
        pred = model.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.mean(scores)

print('Optimizing HistGradientBoosting (50 trials)...')
t0 = time.time()
hgbc_study = optuna.create_study(direction='maximize', study_name='hgbc')
hgbc_study.optimize(hgbc_objective, n_trials=N_OPTUNA_TRIALS_SKLEARN, show_progress_bar=False)
hgbc_best = hgbc_study.best_params
hgbc_best_score = hgbc_study.best_value
print(f'HGBC best AUC: {hgbc_best_score:.5f} ({time.time()-t0:.0f}s)')
print(f'Best params: {hgbc_best}')
wandb.log({'hgbc_optuna_best_auc': hgbc_best_score, 'hgbc_optuna_params': hgbc_best})

print('\\n=== Optuna Summary ===')
print(f'LGB:  {lgb_best_score:.5f}')
print(f'XGB:  {xgb_best_score:.5f}')
print(f'CAT:  {cat_best_score:.5f}')
print(f'ET:   {et_best_score:.5f}')
print(f'HGBC: {hgbc_best_score:.5f}')"""
)

# ── Multi-seed training with tuned params ──
add_code(
    """# ========================================
# Multi-Seed Training with Optuna-Tuned Params
# ========================================
SEEDS = [42, 123, 2024, 7, 999]
N_SPLITS = 10

def train_model_multiseed(model_type, best_params, seeds, n_splits, use_weights=True):
    all_oof = []
    all_preds = []
    all_importances = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        preds = np.zeros(len(X_test))
        imp = np.zeros(X.shape[1])

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            w = sample_weights[tr_idx] if use_weights else None

            if model_type == 'lgb':
                params = {
                    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                    'n_estimators': 3000, 'device': DEVICE, 'random_state': seed,
                    **best_params,
                }
                model = lgb.LGBMClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                          sample_weight=w,
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
                model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                          sample_weight=w, verbose=False)
                imp += model.feature_importances_ / n_splits

            elif model_type == 'cat':
                params = {
                    'iterations': 3000, 'eval_metric': 'Logloss', 'verbose': 0,
                    'early_stopping_rounds': 100, 'task_type': CATBOOST_TASK,
                    'random_seed': seed, **best_params,
                }
                model = CatBoostClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]),
                          sample_weight=w)
                imp += model.feature_importances_ / n_splits

            elif model_type == 'et':
                params = {
                    'n_jobs': -1, 'random_state': seed, **best_params,
                }
                model = ExtraTreesClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], sample_weight=w)
                imp += model.feature_importances_ / n_splits

            elif model_type == 'hgbc':
                params = {
                    'early_stopping': True, 'n_iter_no_change': 50,
                    'validation_fraction': 0.15,
                    'random_state': seed, **best_params,
                }
                model = HistGradientBoostingClassifier(**params)
                model.fit(X[tr_idx], y[tr_idx], sample_weight=w)
                # HistGB doesn't have feature_importances_ in same format, compute manually
                try:
                    imp += np.zeros(X.shape[1])  # placeholder
                except Exception:
                    pass

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

# ── Run training for all 5 models ──
add_code(
    """print('=' * 60)
print('  LightGBM')
print('=' * 60)
lgb_oof, lgb_preds, lgb_imp, lgb_auc = train_model_multiseed('lgb', lgb_best, SEEDS, N_SPLITS)
wandb.log({'lgb_final_auc': lgb_auc})

print()
print('=' * 60)
print('  XGBoost')
print('=' * 60)
xgb_oof, xgb_preds, xgb_imp, xgb_auc = train_model_multiseed('xgb', xgb_best, SEEDS, N_SPLITS)
wandb.log({'xgb_final_auc': xgb_auc})

print()
print('=' * 60)
print('  CatBoost')
print('=' * 60)
cat_oof, cat_preds, cat_imp, cat_auc = train_model_multiseed('cat', cat_best, SEEDS, N_SPLITS)
wandb.log({'cat_final_auc': cat_auc})

print()
print('=' * 60)
print('  ExtraTrees')
print('=' * 60)
et_oof, et_preds, et_imp, et_auc = train_model_multiseed('et', et_best, SEEDS, N_SPLITS)
wandb.log({'et_final_auc': et_auc})

print()
print('=' * 60)
print('  HistGradientBoosting')
print('=' * 60)
hgbc_oof, hgbc_preds, hgbc_imp, hgbc_auc = train_model_multiseed('hgbc', hgbc_best, SEEDS, N_SPLITS)
wandb.log({'hgbc_final_auc': hgbc_auc})

print()
print('=' * 60)
print('  Initial Model Summary')
print('=' * 60)
for name, auc in [('LGB', lgb_auc), ('XGB', xgb_auc), ('CAT', cat_auc), ('ET', et_auc), ('HGBC', hgbc_auc)]:
    print(f'  {name:6s} CV AUC: {auc:.5f}')"""
)

# ── Pseudo Labeling ──
add_code(
    """# ========================================
# Pseudo Labeling
# ========================================
# Use high-confidence predictions from initial models as pseudo labels
# Then retrain top 3 GBDT models with augmented data
print('\\n' + '=' * 60)
print('  Pseudo Labeling')
print('=' * 60)

# Average predictions from all 5 models for pseudo label selection
avg_test_pred = (lgb_preds + xgb_preds + cat_preds + et_preds + hgbc_preds) / 5

# Select high-confidence samples
CONF_HIGH = 0.95
CONF_LOW = 0.05
high_conf_pos = avg_test_pred >= CONF_HIGH
high_conf_neg = avg_test_pred <= CONF_LOW
n_pseudo_pos = high_conf_pos.sum()
n_pseudo_neg = high_conf_neg.sum()
n_pseudo = n_pseudo_pos + n_pseudo_neg
print(f'High confidence positive (>={CONF_HIGH}): {n_pseudo_pos}')
print(f'High confidence negative (<={CONF_LOW}): {n_pseudo_neg}')
print(f'Total pseudo labels: {n_pseudo} / {len(X_test)} test samples ({n_pseudo/len(X_test)*100:.1f}%)')
wandb.log({'pseudo_label_count': n_pseudo, 'pseudo_pos': int(n_pseudo_pos), 'pseudo_neg': int(n_pseudo_neg)})

if n_pseudo >= 50:
    # Create pseudo-labeled dataset
    pseudo_mask = high_conf_pos | high_conf_neg
    X_pseudo = X_test[pseudo_mask]
    y_pseudo = (avg_test_pred[pseudo_mask] >= 0.5).astype(int)

    # Augment training data
    X_aug = np.vstack([X, X_pseudo])
    y_aug = np.concatenate([y, y_pseudo])
    # Weights: original samples keep adversarial weights, pseudo labels get lower weight (0.5)
    w_aug = np.concatenate([sample_weights, np.full(len(X_pseudo), 0.5)])

    print(f'Augmented train: X={X_aug.shape}, y distribution={np.bincount(y_aug)}')

    # Retrain top 3 GBDT models with pseudo labels
    def train_model_pseudo(model_type, best_params, seeds, n_splits, X_tr, y_tr, w_tr):
        all_oof = []
        all_preds = []

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            # OOF only for original train samples
            oof = np.zeros(len(X))
            preds = np.zeros(len(X_test))

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr)):
                w = w_tr[tr_idx]

                if model_type == 'lgb':
                    params = {
                        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                        'n_estimators': 3000, 'device': DEVICE, 'random_state': seed,
                        **best_params,
                    }
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_tr[tr_idx], y_tr[tr_idx],
                              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                              sample_weight=w,
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

                elif model_type == 'xgb':
                    params = {
                        'objective': 'binary:logistic', 'eval_metric': 'auc',
                        'n_estimators': 3000, 'early_stopping_rounds': 100,
                        'verbosity': 0, 'tree_method': 'hist', 'device': 'cuda',
                        'random_state': seed, **best_params,
                    }
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_tr[tr_idx], y_tr[tr_idx],
                              eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                              sample_weight=w, verbose=False)

                elif model_type == 'cat':
                    params = {
                        'iterations': 3000, 'eval_metric': 'Logloss', 'verbose': 0,
                        'early_stopping_rounds': 100, 'task_type': CATBOOST_TASK,
                        'random_seed': seed, **best_params,
                    }
                    model = CatBoostClassifier(**params)
                    model.fit(X_tr[tr_idx], y_tr[tr_idx],
                              eval_set=(X_tr[va_idx], y_tr[va_idx]),
                              sample_weight=w)

                # OOF for original train indices only
                va_orig_mask = va_idx < len(X)
                va_orig_idx = va_idx[va_orig_mask]
                if len(va_orig_idx) > 0:
                    oof[va_orig_idx] = model.predict_proba(X_tr[va_orig_idx])[:, 1]
                preds += model.predict_proba(X_test)[:, 1] / n_splits

            seed_auc = roc_auc_score(y, oof)
            print(f'  {model_type.upper()}_PL seed={seed}: CV AUC = {seed_auc:.5f}')
            all_oof.append(oof)
            all_preds.append(preds)
            gc.collect()

        mean_oof = np.mean(all_oof, axis=0)
        mean_preds = np.mean(all_preds, axis=0)
        mean_auc = roc_auc_score(y, mean_oof)
        print(f'  >>> {model_type.upper()}_PL Multi-Seed CV AUC: {mean_auc:.5f}')
        return mean_oof, mean_preds, mean_auc

    print('\\nRetraining with pseudo labels...')
    lgb_pl_oof, lgb_pl_preds, lgb_pl_auc = train_model_pseudo('lgb', lgb_best, SEEDS, N_SPLITS, X_aug, y_aug, w_aug)
    xgb_pl_oof, xgb_pl_preds, xgb_pl_auc = train_model_pseudo('xgb', xgb_best, SEEDS, N_SPLITS, X_aug, y_aug, w_aug)
    cat_pl_oof, cat_pl_preds, cat_pl_auc = train_model_pseudo('cat', cat_best, SEEDS, N_SPLITS, X_aug, y_aug, w_aug)

    wandb.log({'lgb_pl_auc': lgb_pl_auc, 'xgb_pl_auc': xgb_pl_auc, 'cat_pl_auc': cat_pl_auc})
    has_pseudo = True
    print(f'\\nPseudo Label improvement:')
    print(f'  LGB: {lgb_auc:.5f} -> {lgb_pl_auc:.5f} ({"+" if lgb_pl_auc > lgb_auc else ""}{lgb_pl_auc - lgb_auc:.5f})')
    print(f'  XGB: {xgb_auc:.5f} -> {xgb_pl_auc:.5f} ({"+" if xgb_pl_auc > xgb_auc else ""}{xgb_pl_auc - xgb_auc:.5f})')
    print(f'  CAT: {cat_auc:.5f} -> {cat_pl_auc:.5f} ({"+" if cat_pl_auc > cat_auc else ""}{cat_pl_auc - cat_auc:.5f})')
else:
    print('Not enough high-confidence predictions for pseudo labeling. Skipping.')
    has_pseudo = False"""
)

# ── Permutation Importance Feature Selection ──
add_code(
    """# ========================================
# Permutation Importance Feature Selection
# ========================================
# Use LGB model for feature importance check
# Drop features with negative or zero permutation importance
print('\\nRunning Permutation Importance analysis...')
t0 = time.time()

# Train a quick LGB model for permutation importance
quick_lgb = lgb.LGBMClassifier(
    objective='binary', metric='auc', verbosity=-1,
    n_estimators=500, learning_rate=0.05, device=DEVICE,
    **{k: v for k, v in lgb_best.items() if k != 'learning_rate'},
)
skf_pi = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pi_scores = np.zeros(X.shape[1])
pi_count = 0
for tr_idx, va_idx in skf_pi.split(X, y):
    quick_lgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    result = permutation_importance(quick_lgb, X[va_idx], y[va_idx],
                                     n_repeats=3, random_state=42, scoring='roc_auc',
                                     n_jobs=-1)
    pi_scores += result.importances_mean
    pi_count += 1

pi_scores /= pi_count
pi_df = pd.DataFrame({'feature': feature_cols, 'importance': pi_scores}).sort_values('importance', ascending=False)

# Identify noise features (zero or negative importance)
noise_features = pi_df[pi_df['importance'] <= 0]['feature'].tolist()
print(f'Permutation Importance analysis done ({time.time()-t0:.0f}s)')
print(f'Noise features (importance <= 0): {len(noise_features)}')
if noise_features:
    print(f'Dropping: {noise_features[:20]}{"..." if len(noise_features) > 20 else ""}')

# Report top features
print(f'\\nTop 20 features by permutation importance:')
for _, row in pi_df.head(20).iterrows():
    print(f'  {row["feature"]:40s} {row["importance"]:.6f}')

wandb.log({'noise_features_count': len(noise_features), 'total_features_before_selection': len(feature_cols)})

# Note: We keep all features for now since GBDT models handle noise well.
# Only drop if there are many noise features (>30% of total)
if len(noise_features) > len(feature_cols) * 0.3:
    print(f'\\nToo many noise features ({len(noise_features)}/{len(feature_cols)}). Dropping them.')
    keep_cols = [c for c in feature_cols if c not in noise_features]
    keep_idx = [feature_cols.index(c) for c in keep_cols]
    X = X[:, keep_idx]
    X_test = X_test[:, keep_idx]
    feature_cols = keep_cols
    # Update sample weights size matches
    print(f'Features after selection: {len(feature_cols)}')
    wandb.config.update({'n_features_selected': len(feature_cols)})
else:
    print(f'\\nKeeping all features (noise ratio {len(noise_features)/len(feature_cols)*100:.1f}% < 30%)')"""
)

# ── Ensemble ──
add_code(
    """# ========================================
# Ensemble Methods
# ========================================
print('\\n' + '=' * 60)
print('  Ensemble')
print('=' * 60)

# Collect all OOF and predictions
oof_dict = {
    'lgb': lgb_oof, 'xgb': xgb_oof, 'cat': cat_oof, 'et': et_oof, 'hgbc': hgbc_oof,
}
pred_dict = {
    'lgb': lgb_preds, 'xgb': xgb_preds, 'cat': cat_preds, 'et': et_preds, 'hgbc': hgbc_preds,
}

if has_pseudo:
    oof_dict['lgb_pl'] = lgb_pl_oof
    oof_dict['xgb_pl'] = xgb_pl_oof
    oof_dict['cat_pl'] = cat_pl_oof
    pred_dict['lgb_pl'] = lgb_pl_preds
    pred_dict['xgb_pl'] = xgb_pl_preds
    pred_dict['cat_pl'] = cat_pl_preds

model_names = list(oof_dict.keys())
n_models = len(model_names)
print(f'Models for ensemble: {model_names}')

# --- Simple Average ---
avg_oof = np.mean([oof_dict[k] for k in model_names], axis=0)
avg_preds = np.mean([pred_dict[k] for k in model_names], axis=0)
avg_auc = roc_auc_score(y, avg_oof)
print(f'Simple Average ({n_models} models): AUC = {avg_auc:.5f}')

# --- Rank Average ---
def rank_average(*preds_list):
    ranks = [rankdata(p) for p in preds_list]
    return np.mean(ranks, axis=0)

rank_oof = rank_average(*[oof_dict[k] for k in model_names])
rank_preds = rank_average(*[pred_dict[k] for k in model_names])
rank_auc = roc_auc_score(y, rank_oof)
print(f'Rank Average ({n_models} models):   AUC = {rank_auc:.5f}')

# --- Optuna Weight Search ---
def weight_objective(trial):
    weights = []
    for name in model_names:
        w = trial.suggest_float(f'w_{name}', 0.0, 1.0)
        weights.append(w)
    weights = np.array(weights)
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return 0.0
    weights = weights / w_sum
    blend = sum(w * oof_dict[name] for w, name in zip(weights, model_names))
    return roc_auc_score(y, blend)

print('\\nOptuna weight search...')
t0 = time.time()
weight_study = optuna.create_study(direction='maximize', study_name='weights')
weight_study.optimize(weight_objective, n_trials=500, show_progress_bar=False)
opt_weights = np.array([weight_study.best_params[f'w_{name}'] for name in model_names])
opt_weights = opt_weights / opt_weights.sum()

optuna_blend_oof = sum(w * oof_dict[name] for w, name in zip(opt_weights, model_names))
optuna_blend_preds = sum(w * pred_dict[name] for w, name in zip(opt_weights, model_names))
optuna_blend_auc = roc_auc_score(y, optuna_blend_oof)

print(f'Optuna Weighted Blend: AUC = {optuna_blend_auc:.5f} ({time.time()-t0:.0f}s)')
print('Optimal weights:')
for name, w in zip(model_names, opt_weights):
    print(f'  {name:10s} {w:.4f}')
wandb.log({'optuna_blend_auc': optuna_blend_auc})"""
)

# ── Level-2 Stacking ──
add_code(
    """# ========================================
# Level-2 Stacking: LR + Ridge + LGB meta-learner
# ========================================
print('\\n' + '=' * 60)
print('  Level-2 Stacking')
print('=' * 60)

stack_train = np.column_stack([oof_dict[k] for k in model_names])
stack_test = np.column_stack([pred_dict[k] for k in model_names])
print(f'Stack features: {stack_train.shape[1]} models')

# --- LR stacking ---
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

# --- Ridge stacking ---
# RidgeClassifier doesn't have predict_proba, use decision_function
ridge_stack_oof = np.zeros(len(y))
ridge_stack_preds = np.zeros(len(X_test))
for tr_idx, va_idx in skf_meta.split(stack_train, y):
    meta = RidgeClassifier(alpha=1.0, random_state=42)
    meta.fit(stack_train[tr_idx], y[tr_idx])
    ridge_stack_oof[va_idx] = meta.decision_function(stack_train[va_idx])
    ridge_stack_preds += meta.decision_function(stack_test) / 10
ridge_stack_auc = roc_auc_score(y, ridge_stack_oof)
print(f'Stacking (Ridge): AUC = {ridge_stack_auc:.5f}')

# --- LGB meta-learner ---
lgb_stack_oof = np.zeros(len(y))
lgb_stack_preds = np.zeros(len(X_test))
for tr_idx, va_idx in skf_meta.split(stack_train, y):
    meta = lgb.LGBMClassifier(
        objective='binary', metric='auc', verbosity=-1,
        n_estimators=500, learning_rate=0.05,
        num_leaves=7, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        device=DEVICE,
    )
    meta.fit(stack_train[tr_idx], y[tr_idx],
             eval_set=[(stack_train[va_idx], y[va_idx])],
             callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    lgb_stack_oof[va_idx] = meta.predict_proba(stack_train[va_idx])[:, 1]
    lgb_stack_preds += meta.predict_proba(stack_test)[:, 1] / 10
lgb_stack_auc = roc_auc_score(y, lgb_stack_oof)
print(f'Stacking (LGB):   AUC = {lgb_stack_auc:.5f}')

wandb.log({'lr_stack_auc': lr_stack_auc, 'ridge_stack_auc': ridge_stack_auc, 'lgb_stack_auc': lgb_stack_auc})"""
)

# ── Final Rank Averaging of everything ──
add_code(
    """# ========================================
# Final Rank Averaging of All Methods
# ========================================
print('\\n' + '=' * 60)
print('  Final Rank Averaging')
print('=' * 60)

# Collect all methods
all_methods = {}

# Individual models
for name in model_names:
    auc = roc_auc_score(y, oof_dict[name])
    all_methods[name.upper()] = (auc, oof_dict[name], pred_dict[name])

# Ensemble methods
all_methods['Simple Avg'] = (avg_auc, avg_oof, avg_preds)
all_methods['Rank Avg'] = (rank_auc, rank_oof, rank_preds)
all_methods['Optuna Blend'] = (optuna_blend_auc, optuna_blend_oof, optuna_blend_preds)
all_methods['Stack (LR)'] = (lr_stack_auc, lr_stack_oof, lr_stack_preds)
all_methods['Stack (Ridge)'] = (ridge_stack_auc, ridge_stack_oof, ridge_stack_preds)
all_methods['Stack (LGB)'] = (lgb_stack_auc, lgb_stack_oof, lgb_stack_preds)

# Rank average of top methods
# Include: all stacking + optuna blend + rank avg
top_methods = ['Optuna Blend', 'Stack (LR)', 'Stack (Ridge)', 'Stack (LGB)', 'Rank Avg']
top_oof_list = [all_methods[m][1] for m in top_methods]
top_pred_list = [all_methods[m][2] for m in top_methods]

final_rank_oof = rank_average(*top_oof_list)
final_rank_preds = rank_average(*top_pred_list)
final_rank_auc = roc_auc_score(y, final_rank_oof)
all_methods['FINAL Rank Avg'] = (final_rank_auc, final_rank_oof, final_rank_preds)
print(f'Final Rank Avg of top methods: AUC = {final_rank_auc:.5f}')

# Also try rank avg of everything
all_oof_list = [v[1] for v in all_methods.values()]
all_pred_list = [v[2] for v in all_methods.values()]
mega_rank_oof = rank_average(*all_oof_list)
mega_rank_preds = rank_average(*all_pred_list)
mega_rank_auc = roc_auc_score(y, mega_rank_oof)
all_methods['MEGA Rank Avg'] = (mega_rank_auc, mega_rank_oof, mega_rank_preds)
print(f'Mega Rank Avg (all methods):   AUC = {mega_rank_auc:.5f}')

# --- Pick best ---
best_name = max(all_methods, key=lambda k: all_methods[k][0])
best_auc, best_oof, best_preds = all_methods[best_name]
print(f'\\n*** Best: {best_name} (CV AUC = {best_auc:.5f}) ***')

print(f'\\nAll results (sorted):')
for name, (auc, _, _) in sorted(all_methods.items(), key=lambda x: -x[1][0]):
    marker = ' <<<' if name == best_name else ''
    print(f'  {name:25s} AUC: {auc:.5f}{marker}')
    wandb.log({f'ensemble_{name}_auc': auc})
wandb.log({'best_method': best_name, 'best_auc': best_auc})"""
)

# ── ROC + Feature Importance plots ──
add_code(
    """fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#e67e22', '#9b59b6', '#1abc9c', '#3498db', '#e74c3c',
          '#95a5a6', '#2c3e50', '#f39c12', '#27ae60', '#8e44ad',
          '#c0392b', '#16a085', '#d35400']

# Plot ROC for key methods
plot_data = [
    ('LGB', lgb_oof), ('XGB', xgb_oof), ('CAT', cat_oof),
    ('ET', et_oof), ('HGBC', hgbc_oof),
    ('Optuna Blend', optuna_blend_oof),
    ('Stack (LR)', lr_stack_oof), ('Stack (LGB)', lgb_stack_oof),
    ('FINAL', best_oof),
]
for idx, (name, oof_data) in enumerate(plot_data):
    fpr, tpr, _ = roc_curve(y, oof_data)
    auc_val = roc_auc_score(y, oof_data)
    lw = 3 if name == 'FINAL' else 1.5
    axes[0].plot(fpr, tpr, label=f'{name} ({auc_val:.4f})', linewidth=lw,
                 color=colors[idx % len(colors)])
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('FPR', fontsize=14)
axes[0].set_ylabel('TPR', fontsize=14)
axes[0].set_title('ROC Curves', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=10)
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
ax.set_title(f'Test Predictions -- {best_name}', fontsize=16, fontweight='bold')
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
for name, (auc, _, _) in sorted(all_methods.items(), key=lambda x: -x[1][0]):
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

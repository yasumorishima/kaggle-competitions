!pip install -q --upgrade wandb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import wandb
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'no': '#2ecc71', 'yes': '#e74c3c', 'accent': '#3498db'}

print('All libraries loaded.')

# W&B login via Kaggle Secrets
# To use W&B: Add your API key in Kaggle > Add-ons > Secrets > Label: WANDB_API_KEY
USE_WANDB = False

# Step 1: Get API key from Kaggle Secrets
wandb_api_key = None
try:
    from kaggle_secrets import UserSecretsClient
    print('Step 1a: kaggle_secrets imported OK')
    secrets = UserSecretsClient()
    print('Step 1b: UserSecretsClient created OK')
    wandb_api_key = secrets.get_secret('WANDB_API_KEY')
    print(f'Step 1c: Secret retrieved OK (length={len(wandb_api_key) if wandb_api_key else 0})')
except Exception as e:
    print(f'Step 1 FAILED: {type(e).__name__}: {e}')

# Step 2: Login to W&B
if wandb_api_key:
    try:
        wandb.login(key=wandb_api_key)
        USE_WANDB = True
        print('Step 2: W&B logged in successfully!')
    except Exception as e:
        print(f'Step 2 FAILED: {type(e).__name__}: {e}')
else:
    print('No API key available. Running without W&B.')

print(f'\nUSE_WANDB = {USE_WANDB}')

train = pd.read_csv('/kaggle/input/playground-series-s6e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s6e2/sample_submission.csv')

print(f'Train shape: {train.shape}')
print(f'Test shape:  {test.shape}')
print(f'\nColumn names:')
for i, col in enumerate(train.columns):
    print(f'  {i:2d}. {col:30s} {train[col].dtype}')

# Identify target and feature columns
TARGET = 'Heart Disease'
ID = 'id'

print(f'Target column: "{TARGET}"')
print(f'Target unique values: {train[TARGET].unique()}')
print(f'Target value counts:')
print(train[TARGET].value_counts())

# Submission column check
print(f'\nSubmission columns: {submission.columns.tolist()}')
print(f'Submission head:\n{submission.head()}')

train.head(10)

# Missing values check
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
missing_df = pd.DataFrame({'train': missing_train, 'test': missing_test})
print('Missing values:')
print(missing_df[missing_df.sum(axis=1) > 0] if missing_df.sum().sum() > 0 else 'No missing values!')

train.describe().round(2)

# Encode target for analysis: Presence=1, Absence=0
target_map = {'Absence': 0, 'Presence': 1}
# Auto-detect target mapping
unique_vals = train[TARGET].unique()
print(f'Target unique values: {unique_vals}')

# Try to create binary target
if set(unique_vals) == {'Absence', 'Presence'}:
    train['target'] = train[TARGET].map(target_map)
elif train[TARGET].dtype in ['int64', 'float64']:
    train['target'] = train[TARGET]
else:
    le_target = LabelEncoder()
    train['target'] = le_target.fit_transform(train[TARGET])
    print(f'Label mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count
counts = train[TARGET].value_counts()
axes[0].bar(counts.index, counts.values, 
            color=[COLORS['no'], COLORS['yes']])
axes[0].set_title('Target Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + v*0.01, str(v), ha='center', fontweight='bold')

# Proportion
props = train[TARGET].value_counts(normalize=True)
axes[1].pie(props.values, labels=props.index, 
            colors=[COLORS['no'], COLORS['yes']],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Target Proportion')

plt.suptitle('Target Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

feature_cols = [c for c in train.columns if c not in [ID, TARGET, 'target']]
n_features = len(feature_cols)
n_rows = (n_features + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    for label, color, name in [(0, COLORS['no'], 'No Disease'), (1, COLORS['yes'], 'Disease')]:
        data = train[train['target'] == label][col]
        ax.hist(data, alpha=0.6, label=name, bins=30, color=color, edgecolor='white')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Feature Distributions by Heart Disease Status', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# Correlation with numeric target
corr_df = train[feature_cols + ['target']].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Top correlations with target
target_corr = corr_df['target'].drop('target').abs().sort_values(ascending=False)
print('\nTop correlations with Heart Disease (absolute):')
for feat, val in target_corr.items():
    direction = '+' if corr_df.loc[feat, 'target'] > 0 else '-'
    print(f'  {feat:30s} {direction}{val:.3f}')

# Mean values comparison
comparison = train.groupby('target')[feature_cols].mean().T
comparison.columns = ['No Disease (0)', 'Disease (1)']
comparison['Diff %'] = ((comparison['Disease (1)'] - comparison['No Disease (0)']) / comparison['No Disease (0)'] * 100).round(1)
comparison = comparison.round(2)
print('Feature means by target:')
comparison

# Combine train + test
train_len = len(train)
df = pd.concat([train.drop(['target'], axis=1), test], axis=0, ignore_index=True)

# Encode target
le_target = LabelEncoder()
df.loc[:train_len-1, 'target_encoded'] = le_target.fit_transform(df.loc[:train_len-1, TARGET])

# Interaction features
df['Age_x_MaxHR'] = df['Age'] * df['Max HR']
df['Age_x_STdep'] = df['Age'] * df['ST depression']
df['STdep_x_Slope'] = df['ST depression'] * df['Slope of ST']
df['BP_x_Chol'] = df['BP'] * df['Cholesterol']
df['MaxHR_div_Age'] = df['Max HR'] / (df['Age'] + 1)
df['Vessels_x_Thal'] = df['Number of vessels fluro'] * df['Thallium']

# All feature columns for model
model_features = [c for c in df.columns if c not in [ID, TARGET, 'target_encoded']]

# Split back
X = df.iloc[:train_len][model_features].values
y = df.iloc[:train_len]['target_encoded'].values.astype(int)
X_test = df.iloc[train_len:][model_features].values

print(f'Features ({len(model_features)}):')
for f in model_features:
    print(f'  - {f}')
print(f'\nX shape: {X.shape}, y shape: {y.shape}, X_test shape: {X_test.shape}')
print(f'Target: 0={le_target.classes_[0]}, 1={le_target.classes_[1]}')

N_SPLITS = 5
SEED = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

def wandb_init(name, tags, config):
    if USE_WANDB:
        return wandb.init(
            project='kaggle-s6e2-heart-disease',
            name=name, tags=tags, config=config, reinit=True
        )
    return None

def wandb_log(data):
    if USE_WANDB:
        wandb.log(data)

def wandb_end():
    if USE_WANDB:
        wandb.finish()

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'device': 'gpu',
}

wandb_init('lgb-baseline', ['lightgbm', 'baseline', 'gpu'], {'model': 'LightGBM', **lgb_params})

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))
lgb_importances = np.zeros(len(model_features))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgb_oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    lgb_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    lgb_importances += model.feature_importances_ / N_SPLITS
    
    fold_auc = roc_auc_score(y[val_idx], lgb_oof[val_idx])
    wandb_log({'fold': fold, 'fold_auc': fold_auc})
    print(f'  Fold {fold}: AUC = {fold_auc:.5f}')

lgb_auc = roc_auc_score(y, lgb_oof)
print(f'  >>> LightGBM CV AUC: {lgb_auc:.5f}')
wandb_log({'cv_auc': lgb_auc})
wandb_end()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'verbosity': 0,
    'early_stopping_rounds': 50,
    'tree_method': 'hist',
    'device': 'cuda',
}

wandb_init('xgb-baseline', ['xgboost', 'baseline', 'gpu'], {'model': 'XGBoost', **xgb_params})

xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False
    )
    xgb_oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    xgb_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    fold_auc = roc_auc_score(y[val_idx], xgb_oof[val_idx])
    wandb_log({'fold': fold, 'fold_auc': fold_auc})
    print(f'  Fold {fold}: AUC = {fold_auc:.5f}')

xgb_auc = roc_auc_score(y, xgb_oof)
print(f'  >>> XGBoost CV AUC: {xgb_auc:.5f}')
wandb_log({'cv_auc': xgb_auc})
wandb_end()

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': 0,
    'early_stopping_rounds': 50,
    'task_type': 'GPU',
}

wandb_init('cat-baseline', ['catboost', 'baseline', 'gpu'], {'model': 'CatBoost', **cat_params})

cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = CatBoostClassifier(**cat_params)
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=(X[val_idx], y[val_idx]),
    )
    cat_oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    cat_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    fold_auc = roc_auc_score(y[val_idx], cat_oof[val_idx])
    wandb_log({'fold': fold, 'fold_auc': fold_auc})
    print(f'  Fold {fold}: AUC = {fold_auc:.5f}')

cat_auc = roc_auc_score(y, cat_oof)
print(f'  >>> CatBoost CV AUC: {cat_auc:.5f}')
wandb_log({'cv_auc': cat_auc})
wandb_end()

# Ensemble
ens_oof = (lgb_oof + xgb_oof + cat_oof) / 3
ens_preds = (lgb_preds + xgb_preds + cat_preds) / 3
ens_auc = roc_auc_score(y, ens_oof)

# Log ensemble to W&B
wandb_init('ensemble-avg', ['ensemble'], {
    'method': 'simple_average',
    'models': ['LightGBM', 'XGBoost', 'CatBoost']
})
wandb_log({
    'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
    'cat_auc': cat_auc, 'ensemble_auc': ens_auc
})
wandb_end()

# Results table
results = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble (avg)'],
    'CV AUC': [lgb_auc, xgb_auc, cat_auc, ens_auc]
}).sort_values('CV AUC', ascending=False)

print('=' * 40)
print('      MODEL COMPARISON')
print('=' * 40)
for _, row in results.iterrows():
    marker = ' <<<' if row['Model'] == 'Ensemble (avg)' else ''
    print(f"  {row['Model']:20s} AUC: {row['CV AUC']:.5f}{marker}")
print('=' * 40)

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, oof, color in [
    ('LightGBM', lgb_oof, '#e67e22'),
    ('XGBoost', xgb_oof, '#9b59b6'),
    ('CatBoost', cat_oof, '#1abc9c'),
    ('Ensemble', ens_oof, '#e74c3c'),
]:
    fpr, tpr, _ = roc_curve(y, oof)
    auc_val = roc_auc_score(y, oof)
    lw = 3 if name == 'Ensemble' else 1.5
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', linewidth=lw, color=color)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].legend(fontsize=10)

# Feature importance (LightGBM)
imp = pd.Series(lgb_importances, index=model_features).sort_values(ascending=True)
imp.plot(kind='barh', ax=axes[1], color=COLORS['accent'])
axes[1].set_title('Feature Importance (LightGBM avg)', fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(ens_oof[y == 0], bins=50, alpha=0.6, label='No Disease (actual)', color=COLORS['no'], edgecolor='white')
ax.hist(ens_oof[y == 1], bins=50, alpha=0.6, label='Disease (actual)', color=COLORS['yes'], edgecolor='white')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Threshold=0.5')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Count')
ax.set_title('Ensemble OOF Prediction Distribution', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# Check submission format
print(f'Submission columns: {submission.columns.tolist()}')
print(f'Expected target column: "{TARGET}"')

submission[TARGET] = ens_preds
submission.to_csv('submission.csv', index=False)

print(f'\nSubmission shape: {submission.shape}')
print(f'Prediction range: [{ens_preds.min():.4f}, {ens_preds.max():.4f}]')
print(f'Prediction mean:  {ens_preds.mean():.4f}')
submission.head(10)

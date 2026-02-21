# Cell 1: W&B offline mode must be set BEFORE importing wandb
import os
os.environ['WANDB_MODE'] = 'offline'

import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print(f'wandb version: {wandb.__version__}')
print(f'WANDB_MODE: {os.environ.get("WANDB_MODE")}')

# Cell 2: Data path auto-detection
# competition_sources mounts at /kaggle/input/competitions/<slug>/ or /kaggle/input/<slug>/
SLUG = 'march-machine-learning-mania-2026'
CANDIDATES = [
    Path(f'/kaggle/input/competitions/{SLUG}'),
    Path(f'/kaggle/input/{SLUG}'),
]

DATA_DIR = None
for p in CANDIDATES:
    if (p / 'MTeams.csv').exists():
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError(f'MTeams.csv not found in: {CANDIDATES}')

print(f'DATA_DIR: {DATA_DIR}')

m_seeds   = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
m_tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
m_reg     = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')

print(f'Tourney games: {len(m_tourney)}')
print(f'Regular season games: {len(m_reg)}')

# Cell 3: Simple feature engineering (seed diff + win rate diff)
def parse_seed(s):
    return int(''.join(filter(str.isdigit, s)))

m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed)

def build_winrate(reg_df):
    w = reg_df[['Season', 'WTeamID']].assign(Win=1).rename(columns={'WTeamID': 'TeamID'})
    l = reg_df[['Season', 'LTeamID']].assign(Win=0).rename(columns={'LTeamID': 'TeamID'})
    df = pd.concat([w, l], ignore_index=True)
    return df.groupby(['Season', 'TeamID'])['Win'].mean().reset_index(name='WinRate')

winrate = build_winrate(m_reg)

tc = m_tourney.copy()
tc = tc.merge(
    m_seeds[['Season', 'TeamID', 'SeedNum']].rename(columns={'TeamID': 'WTeamID', 'SeedNum': 'WSeed'}),
    on=['Season', 'WTeamID'], how='left'
).merge(
    m_seeds[['Season', 'TeamID', 'SeedNum']].rename(columns={'TeamID': 'LTeamID', 'SeedNum': 'LSeed'}),
    on=['Season', 'LTeamID'], how='left'
).merge(
    winrate.rename(columns={'TeamID': 'WTeamID', 'WinRate': 'WWinRate'}),
    on=['Season', 'WTeamID'], how='left'
).merge(
    winrate.rename(columns={'TeamID': 'LTeamID', 'WinRate': 'LWinRate'}),
    on=['Season', 'LTeamID'], how='left'
)

tc['SeedDiff']    = tc['WSeed']    - tc['LSeed']
tc['WinRateDiff'] = tc['WWinRate'] - tc['LWinRate']

features = ['SeedDiff', 'WinRateDiff']
X = tc[features].fillna(0).values
y = np.ones(len(tc))

rng = np.random.default_rng(42)
flip = rng.integers(0, 2, len(X)).astype(bool)
X[flip] *= -1
y[flip] = 0

print(f'Training samples: {len(X)}, Features: {features}')

# Cell 4: W&B offline logging with LogisticRegression CV
run = wandb.init(
    project='march-mania-2026-test',
    name='offline-sync-test',
    config={
        'model': 'LogisticRegression',
        'features': features,
        'cv_folds': 5,
        'random_state': 42,
    }
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42, max_iter=1000)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_log_loss')
log_loss_mean = -cv_scores.mean()
log_loss_std  = cv_scores.std()

wandb.log({
    'cv_log_loss_mean': log_loss_mean,
    'cv_log_loss_std':  log_loss_std,
    'n_features': len(features),
    'n_samples': len(X),
})

print(f'CV Log Loss: {log_loss_mean:.4f} ± {log_loss_std:.4f}')
print(f'W&B run dir: {run.dir}')

wandb.finish()
print('wandb.finish() completed')

# Cell 5: Verify wandb offline files exist in /kaggle/working/wandb/
import subprocess
result = subprocess.run(
    ['find', '/kaggle/working/wandb', '-type', 'f'],
    capture_output=True, text=True
)
print('=== /kaggle/working/wandb/ contents ===')
print(result.stdout if result.stdout else '(empty)')
if result.stderr:
    print('STDERR:', result.stderr)

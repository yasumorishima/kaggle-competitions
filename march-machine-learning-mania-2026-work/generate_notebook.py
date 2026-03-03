"""March ML Mania 2026 — Elo + Multi-Massey + Recent Form (private working notebook)

Improvements over public baseline:
- Elo ratings (computed from all regular season games, with season carryover)
- Multiple Massey ordinals aggregated (all systems → AvgRank)
- Recent form (last 30 days win rate)
- XGBoost added (LGB + XGB rank average ensemble)
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
    cells.append({
        "cell_type": "code",
        "id": f"cell-{cell_counter:03d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })


# ── Cell 1: Title ──
add_md("""# March ML Mania 2026 — Elo + Multi-Massey + Recent Form

**Working notebook (private)** — experiment log

**vs public baseline:**
- Elo ratings from full game history (season carryover 0.5)
- Massey ordinals: all systems aggregated → AvgRank
- Recent form: last 30 days win rate
- XGBoost added → LGB + XGB rank average
- W&B offline tracking""")

# ── Cell 2: Setup ──
add_code("""import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'march-machine-learning-mania-2026'

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import wandb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/kaggle/input/competitions/march-machine-learning-mania-2026')
print('Libraries loaded. WANDB_MODE:', os.environ['WANDB_MODE'])""")

# ── Cell 3: W&B init ──
add_code("""run = wandb.init(
    project='march-machine-learning-mania-2026',
    name='elo-multimassey-v1',
    tags=['elo', 'multi-massey', 'recent-form', 'lgb+xgb'],
    config={
        'elo_k': 32,
        'elo_carryover': 0.5,
        'recent_days': 30,
        'n_seeds': 3,
        'n_splits': 5,
    }
)
print(f'W&B run: {run.name}')""")

# ── Cell 4: Load data ──
add_code("""# Diagnose what's mounted
import os
INPUT_ROOT = Path('/kaggle/input')
print('=== /kaggle/input/ ===')
if INPUT_ROOT.exists():
    for p in sorted(INPUT_ROOT.iterdir()):
        files = list(p.iterdir())[:5]
        print(f'  {p.name}/  ({len(list(p.iterdir()))} files)')
        for f in files:
            print(f'    {f.name}')
else:
    print('  (not found)')
print()

m_teams   = pd.read_csv(DATA_DIR / 'MTeams.csv')
m_seeds   = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
m_tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
m_reg     = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
massey    = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')
sub       = pd.read_csv(DATA_DIR / 'SampleSubmissionStage1.csv')

def parse_seed(s):
    return int(''.join(filter(str.isdigit, s)))

m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed)

print(f"M reg: {m_reg.shape}")
print(f"Massey systems: {massey['SystemName'].nunique()}")
print(f"Submission rows: {len(sub):,}")""")

# ── Cell 5: Elo computation ──
add_code("""def compute_elo(games_df, k=32, initial=1500, carryover=0.5):
    \"\"\"End-of-season Elo per (Season, TeamID). Seasons processed in order.\"\"\"
    elo = {}
    records = []

    for season in sorted(games_df['Season'].unique()):
        # Decay toward mean at season start
        for tid in elo:
            elo[tid] = initial + carryover * (elo[tid] - initial)

        season_games = games_df[games_df['Season'] == season].sort_values('DayNum')

        for _, row in season_games.iterrows():
            w, l = row['WTeamID'], row['LTeamID']
            elo.setdefault(w, initial)
            elo.setdefault(l, initial)

            exp_w = 1 / (1 + 10 ** (-(elo[w] - elo[l]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))

        # Record end-of-season Elo for all teams that played this season
        teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in teams:
            records.append({'Season': season, 'TeamID': tid, 'Elo': elo.get(tid, initial)})

    return pd.DataFrame(records)


m_elo = compute_elo(m_reg, k=32, initial=1500, carryover=0.5)

print(f"M Elo range: {m_elo['Elo'].min():.0f} - {m_elo['Elo'].max():.0f}")
m_elo.head()""")

# ── Cell 6: Multi-Massey aggregation ──
add_code("""# Use all systems — take last ranking day of each season, average across systems
massey_multi = (
    massey
    .sort_values(['Season', 'RankingDayNum'])
    .groupby(['Season', 'TeamID', 'SystemName']).last()
    .reset_index()
    .groupby(['Season', 'TeamID'])['OrdinalRank'].mean()
    .reset_index()
    .rename(columns={'OrdinalRank': 'AvgRank'})
)

# Also keep MOR alone for comparison
massey_mor = (
    massey[massey['SystemName'] == 'MOR']
    .sort_values(['Season', 'RankingDayNum'])
    .groupby(['Season', 'TeamID']).last()
    .reset_index()[['Season', 'TeamID', 'OrdinalRank']]
    .rename(columns={'OrdinalRank': 'MOR'})
)

print(f"Massey multi: {massey_multi.shape}")
print(f"AvgRank range: {massey_multi['AvgRank'].min():.1f} - {massey_multi['AvgRank'].max():.1f}")
massey_multi.head()""")

# ── Cell 7: Season stats + Recent form ──
add_code("""def build_season_stats(reg_df):
    w = reg_df[['Season','WTeamID','WScore','LScore']].copy()
    w.columns = ['Season','TeamID','ScoreFor','ScoreAgainst']
    w['Win'] = 1
    l = reg_df[['Season','LTeamID','LScore','WScore']].copy()
    l.columns = ['Season','TeamID','ScoreFor','ScoreAgainst']
    l['Win'] = 0
    df = pd.concat([w, l], ignore_index=True)
    df['Margin'] = df['ScoreFor'] - df['ScoreAgainst']
    stats = df.groupby(['Season','TeamID']).agg(
        WinRate=('Win','mean'),
        AvgMargin=('Margin','mean'),
        AvgScore=('ScoreFor','mean'),
    ).reset_index()
    return stats


def build_recent_form(reg_df, days_back=30):
    \"\"\"Win rate in the last `days_back` days of regular season per team.\"\"\"
    max_day = reg_df.groupby('Season')['DayNum'].max()
    rows = []
    for season, grp in reg_df.groupby('Season'):
        cutoff = max_day[season] - days_back
        recent = grp[grp['DayNum'] >= cutoff]
        for _, r in recent.iterrows():
            rows.append({'Season': season, 'TeamID': r['WTeamID'], 'Win': 1})
            rows.append({'Season': season, 'TeamID': r['LTeamID'], 'Win': 0})
    rf = pd.DataFrame(rows)
    return (
        rf.groupby(['Season','TeamID'])['Win'].mean()
        .reset_index().rename(columns={'Win': 'RecentWinRate'})
    )


m_stats  = build_season_stats(m_reg)
m_recent = build_recent_form(m_reg, days_back=30)

print("Season stats sample:")
print(m_stats.head())
print("\\nRecent form sample:")
print(m_recent.head())""")

# ── Cell 8: Merge all features per team ──
add_code("""def merge_team_features(stats, seeds, elo, recent, massey_multi, massey_mor):
    \"\"\"One row per (Season, TeamID) with all features.\"\"\"
    df = stats.copy()
    df = df.merge(seeds[['Season','TeamID','SeedNum']], on=['Season','TeamID'], how='left')
    df = df.merge(elo,    on=['Season','TeamID'], how='left')
    df = df.merge(recent, on=['Season','TeamID'], how='left')
    df = df.merge(massey_multi, on=['Season','TeamID'], how='left')
    df = df.merge(massey_mor,   on=['Season','TeamID'], how='left')
    return df


m_feats = merge_team_features(m_stats, m_seeds, m_elo, m_recent, massey_multi, massey_mor)

print(f"M features: {m_feats.shape}")
print(m_feats.head())""")

# ── Cell 9: Build training dataset ──
add_code("""FEAT_COLS = [
    # Diff features (T1 - T2)
    'SeedDiff', 'EloDiff', 'WinRateDiff', 'MarginDiff', 'ScoreDiff',
    'RecentWinRateDiff', 'AvgRankDiff', 'MORDiff',
    # Individual team features
    'T1_Seed', 'T2_Seed',
    'T1_Elo', 'T2_Elo',
    'T1_WinRate', 'T2_WinRate',
    'T1_AvgMargin', 'T2_AvgMargin',
    'T1_RecentWinRate', 'T2_RecentWinRate',
    'T1_AvgRank', 'T2_AvgRank',
]


def build_train_df(tourney, feats):
    rows = []
    feat_idx = feats.set_index(['Season','TeamID'])

    for _, r in tourney.iterrows():
        s = r['Season']
        t1, t2 = sorted([r['WTeamID'], r['LTeamID']])
        label = 1 if r['WTeamID'] == t1 else 0

        def get(tid, col):
            try:
                return feat_idx.loc[(s, tid), col]
            except Exception:
                return np.nan

        row = dict(Season=s, T1=t1, T2=t2, Label=label)
        for col in ['SeedNum','Elo','WinRate','AvgMargin','AvgScore','RecentWinRate','AvgRank','MOR']:
            row[f'T1_{col}'] = get(t1, col)
            row[f'T2_{col}'] = get(t2, col)

        rows.append(row)

    df = pd.DataFrame(rows)
    df['SeedDiff']          = df['T1_SeedNum']        - df['T2_SeedNum']
    df['EloDiff']           = df['T1_Elo']             - df['T2_Elo']
    df['WinRateDiff']       = df['T1_WinRate']         - df['T2_WinRate']
    df['MarginDiff']        = df['T1_AvgMargin']       - df['T2_AvgMargin']
    df['ScoreDiff']         = df['T1_AvgScore']        - df['T2_AvgScore']
    df['RecentWinRateDiff'] = df['T1_RecentWinRate']   - df['T2_RecentWinRate']
    df['AvgRankDiff']       = df['T1_AvgRank']         - df['T2_AvgRank']
    df['MORDiff']           = df['T1_MOR']             - df['T2_MOR']
    # Rename for FEAT_COLS compatibility
    df = df.rename(columns={'T1_SeedNum':'T1_Seed','T2_SeedNum':'T2_Seed'})
    return df


m_train = build_train_df(m_tourney, m_feats)

print(f"M train: {m_train.shape}, label balance: {m_train['Label'].mean():.3f}")
print(f"NaN ratio:\\n{m_train[FEAT_COLS].isna().mean().sort_values(ascending=False).head(10)}")""")

# ── Cell 10: CV + model training ──
add_code("""SEEDS   = [42, 123, 2024]
N_SPLITS = 5

lgb_params = dict(
    objective='binary', metric='binary_logloss', verbosity=-1,
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
)

xgb_params = dict(
    objective='binary:logistic', eval_metric='logloss',
    n_estimators=500, learning_rate=0.05,
    max_depth=5, subsample=0.8, colsample_bytree=0.8,
    verbosity=0, early_stopping_rounds=50,
)


def train_ensemble(train_df, label_str):
    df  = train_df.dropna(subset=['SeedDiff','EloDiff','WinRateDiff']).copy()
    med = df[FEAT_COLS].median()
    X   = df[FEAT_COLS].fillna(med).values
    y   = df['Label'].values

    all_lgb_oof, all_lgb_preds = [], []
    all_xgb_oof, all_xgb_preds = [], []

    for seed in SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        lgb_oof = np.zeros(len(X))
        xgb_oof = np.zeros(len(X))

        for tr_idx, va_idx in kf.split(X, y):
            # LGB
            m_lgb = lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})
            m_lgb.fit(X[tr_idx], y[tr_idx],
                      eval_set=[(X[va_idx], y[va_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            lgb_oof[va_idx] = m_lgb.predict_proba(X[va_idx])[:, 1]

            # XGB
            m_xgb = xgb.XGBClassifier(**{**xgb_params, 'random_state': seed})
            m_xgb.fit(X[tr_idx], y[tr_idx],
                      eval_set=[(X[va_idx], y[va_idx])], verbose=False)
            xgb_oof[va_idx] = m_xgb.predict_proba(X[va_idx])[:, 1]

        all_lgb_oof.append(lgb_oof)
        all_xgb_oof.append(xgb_oof)

    lgb_oof_mean = np.mean(all_lgb_oof, axis=0)
    xgb_oof_mean = np.mean(all_xgb_oof, axis=0)

    lgb_ll  = log_loss(y, lgb_oof_mean)
    xgb_ll  = log_loss(y, xgb_oof_mean)
    rank_avg = (rankdata(lgb_oof_mean) + rankdata(xgb_oof_mean)) / 2
    rank_ll  = log_loss(y, rank_avg / len(rank_avg))  # normalize for log_loss

    print(f"[{label_str}] LGB  CV LogLoss: {lgb_ll:.5f}")
    print(f"[{label_str}] XGB  CV LogLoss: {xgb_ll:.5f}")
    print(f"[{label_str}] Rank CV LogLoss: {rank_ll:.5f}")
    wandb.log({
        f'{label_str}_lgb_logloss':  lgb_ll,
        f'{label_str}_xgb_logloss':  xgb_ll,
        f'{label_str}_rank_logloss': rank_ll,
    })
    return med, lgb_ll, xgb_ll


m_med, m_lgb_ll, m_xgb_ll = train_ensemble(m_train, 'Men')""")

# ── Cell 11: Full training on all data + prediction ──
add_code("""def train_full_and_predict(train_df, sub_df, feats, med, label_str):
    \"\"\"Retrain on all tourney data and predict submission.\"\"\"
    df  = train_df.dropna(subset=['SeedDiff','EloDiff','WinRateDiff']).copy()
    X   = df[FEAT_COLS].fillna(med).values
    y   = df['Label'].values

    feat_idx  = feats.set_index(['Season','TeamID'])
    sub_df    = sub_df.copy()
    sub_df[['Season','T1','T2']] = sub_df['ID'].str.split('_', expand=True).astype(int)

    # Build sub features
    sub_rows = []
    for _, r in sub_df.iterrows():
        s, t1, t2 = r['Season'], r['T1'], r['T2']

        def get(tid, col):
            try: return feat_idx.loc[(s, tid), col]
            except: return np.nan

        row = {}
        for col in ['SeedNum','Elo','WinRate','AvgMargin','AvgScore','RecentWinRate','AvgRank','MOR']:
            row[f'T1_{col}'] = get(t1, col)
            row[f'T2_{col}'] = get(t2, col)

        row['SeedDiff']          = row['T1_SeedNum']      - row['T2_SeedNum']
        row['EloDiff']           = row['T1_Elo']           - row['T2_Elo']
        row['WinRateDiff']       = row['T1_WinRate']       - row['T2_WinRate']
        row['MarginDiff']        = row['T1_AvgMargin']     - row['T2_AvgMargin']
        row['ScoreDiff']         = row['T1_AvgScore']      - row['T2_AvgScore']
        row['RecentWinRateDiff'] = row['T1_RecentWinRate'] - row['T2_RecentWinRate']
        row['AvgRankDiff']       = row['T1_AvgRank']       - row['T2_AvgRank']
        row['MORDiff']           = row['T1_MOR']           - row['T2_MOR']
        row['T1_Seed']           = row.pop('T1_SeedNum')
        row['T2_Seed']           = row.pop('T2_SeedNum')
        sub_rows.append(row)

    X_sub = pd.DataFrame(sub_rows)[FEAT_COLS].fillna(med).values

    all_lgb_preds, all_xgb_preds = [], []
    for seed in SEEDS:
        m_lgb = lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})
        m_lgb.fit(X, y)
        all_lgb_preds.append(m_lgb.predict_proba(X_sub)[:, 1])

        m_xgb = xgb.XGBClassifier(**{**xgb_params, 'random_state': seed, 'early_stopping_rounds': None})
        m_xgb.fit(X, y)
        all_xgb_preds.append(m_xgb.predict_proba(X_sub)[:, 1])

    lgb_pred = np.mean(all_lgb_preds, axis=0)
    xgb_pred = np.mean(all_xgb_preds, axis=0)
    rank_pred = (rankdata(lgb_pred) + rankdata(xgb_pred)) / 2
    # Normalize rank to [0,1]
    rank_pred = (rank_pred - rank_pred.min()) / (rank_pred.max() - rank_pred.min())
    pred = np.clip(rank_pred, 0.025, 0.975)
    print(f"[{label_str}] pred range: {pred.min():.3f} - {pred.max():.3f}, mean: {pred.mean():.3f}")
    return sub_df, pred


sub_out, m_pred = train_full_and_predict(m_train, sub, m_feats, m_med, 'Men')""")

# ── Cell 12: Save submission ──
add_code("""sub_out['Pred'] = m_pred
submission = sub_out[['ID','Pred']].sort_values('ID').reset_index(drop=True)
submission.to_csv('submission.csv', index=False)

print(f"submission.csv saved: {len(submission):,} rows")
print(submission.head(10))

wandb.log({
    'submission_rows': len(submission),
    'pred_mean': submission['Pred'].mean(),
    'pred_min': submission['Pred'].min(),
    'pred_max': submission['Pred'].max(),
})
wandb.finish()
print('Done. W&B offline run saved.')""")


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

out_path = "march-mania-2026-elo-work.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path} ({len(cells)} cells)")

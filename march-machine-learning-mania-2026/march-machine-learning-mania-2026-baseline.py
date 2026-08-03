import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

DATA_DIR = Path('/kaggle/input/march-machine-learning-mania-2026')

# Load Data
m_teams   = pd.read_csv(DATA_DIR / 'MTeams.csv')
m_seeds   = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
m_tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
m_reg     = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
massey    = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')
w_teams   = pd.read_csv(DATA_DIR / 'WTeams.csv')
w_seeds   = pd.read_csv(DATA_DIR / 'WNCAATourneySeeds.csv')
w_tourney = pd.read_csv(DATA_DIR / 'WNCAATourneyCompactResults.csv')
w_reg     = pd.read_csv(DATA_DIR / 'WRegularSeasonCompactResults.csv')
sub_stage1 = pd.read_csv(DATA_DIR / 'SampleSubmissionStage1.csv')


def parse_seed(s):
    return int(''.join(filter(str.isdigit, s)))


m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed)
w_seeds['SeedNum'] = w_seeds['Seed'].apply(parse_seed)


def attach_seeds(tourney, seeds):
    tc = tourney.copy()
    tc = tc.merge(seeds[['Season','TeamID','SeedNum']].rename(columns={'TeamID':'WTeamID','SeedNum':'WSeed'}), on=['Season','WTeamID'])
    tc = tc.merge(seeds[['Season','TeamID','SeedNum']].rename(columns={'TeamID':'LTeamID','SeedNum':'LSeed'}), on=['Season','LTeamID'])
    tc['SeedDiff'] = tc['WSeed'] - tc['LSeed']
    return tc


def build_season_stats(reg_df):
    w = reg_df[['Season','WTeamID','WScore','LScore']].copy()
    w.columns = ['Season','TeamID','ScoreFor','ScoreAgainst']
    w['Win'] = 1
    l = reg_df[['Season','LTeamID','LScore','WScore']].copy()
    l.columns = ['Season','TeamID','ScoreFor','ScoreAgainst']
    l['Win'] = 0
    df = pd.concat([w, l], ignore_index=True)
    df['Margin'] = df['ScoreFor'] - df['ScoreAgainst']
    stats = df.groupby(['Season','TeamID']).agg(
        Games=('Win','count'), Wins=('Win','sum'),
        AvgScore=('ScoreFor','mean'), AvgMargin=('Margin','mean'),
    ).reset_index()
    stats['WinRate'] = stats['Wins'] / stats['Games']
    return stats


m_stats = build_season_stats(m_reg)
w_stats = build_season_stats(w_reg)

massey_agg = (
    massey[massey['SystemName'] == 'MOR']
    .sort_values(['Season','RankingDayNum'])
    .groupby(['Season','TeamID']).last().reset_index()
    [['Season','TeamID','OrdinalRank']].rename(columns={'OrdinalRank':'MOR'})
)

FEAT_COLS = [
    'SeedDiff','WinRateDiff','MarginDiff','ScoreDiff','MORDiff',
    'T1_Seed','T2_Seed','T1_WinRate','T2_WinRate',
    'T1_Margin','T2_Margin','T1_MOR','T2_MOR'
]


def get_feats(season, team_id, stats, seed_df, mor_df=None):
    s_row = stats[(stats.Season == season) & (stats.TeamID == team_id)]
    e_row = seed_df[(seed_df.Season == season) & (seed_df.TeamID == team_id)]
    win_rate   = s_row['WinRate'].values[0]   if len(s_row) else np.nan
    avg_margin = s_row['AvgMargin'].values[0] if len(s_row) else np.nan
    avg_score  = s_row['AvgScore'].values[0]  if len(s_row) else np.nan
    seed_num   = e_row['SeedNum'].values[0]   if len(e_row) else np.nan
    mor = np.nan
    if mor_df is not None:
        m_row = mor_df[(mor_df.Season == season) & (mor_df.TeamID == team_id)]
        mor = m_row['MOR'].values[0] if len(m_row) else np.nan
    return win_rate, avg_margin, avg_score, seed_num, mor


def build_train_df(tourney, stats, seed_df, mor_df=None):
    rows = []
    for _, r in tourney.iterrows():
        s = r['Season']
        t1, t2 = sorted([r['WTeamID'], r['LTeamID']])
        label = 1 if r['WTeamID'] == t1 else 0
        f1 = get_feats(s, t1, stats, seed_df, mor_df)
        f2 = get_feats(s, t2, stats, seed_df, mor_df)
        rows.append(dict(
            Season=s, T1=t1, T2=t2,
            T1_WinRate=f1[0], T2_WinRate=f2[0],
            T1_Margin=f1[1],  T2_Margin=f2[1],
            T1_Score=f1[2],   T2_Score=f2[2],
            T1_Seed=f1[3],    T2_Seed=f2[3],
            T1_MOR=f1[4],     T2_MOR=f2[4],
            Label=label
        ))
    df = pd.DataFrame(rows)
    df['SeedDiff']    = df['T1_Seed']    - df['T2_Seed']
    df['WinRateDiff'] = df['T1_WinRate'] - df['T2_WinRate']
    df['MarginDiff']  = df['T1_Margin']  - df['T2_Margin']
    df['ScoreDiff']   = df['T1_Score']   - df['T2_Score']
    df['MORDiff']     = df['T1_MOR']     - df['T2_MOR']
    return df


def train_models(train_df, label='Men'):
    df = train_df.dropna(subset=['SeedDiff','WinRateDiff','MarginDiff','Label'])
    X = df[FEAT_COLS].fillna(df[FEAT_COLS].median()).values
    y = df['Label'].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    cv_lr = cross_val_score(lr, X_sc, y, cv=5, scoring='neg_log_loss')
    print(f'[{label}] LR  CV LogLoss: {-cv_lr.mean():.4f} ± {cv_lr.std():.4f}')
    lr.fit(X_sc, y)
    lgb_m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    cv_lgb = cross_val_score(lgb_m, X, y, cv=5, scoring='neg_log_loss')
    print(f'[{label}] LGB CV LogLoss: {-cv_lgb.mean():.4f} ± {cv_lgb.std():.4f}')
    lgb_m.fit(X, y)
    med = df[FEAT_COLS].median()
    return lr, lgb_m, scaler, med


m_train = build_train_df(m_tourney, m_stats, m_seeds, massey_agg)
w_train = build_train_df(w_tourney, w_stats, w_seeds, mor_df=None)
m_lr, m_lgb, m_scaler, m_med = train_models(m_train, 'Men')
w_lr, w_lgb, w_scaler, w_med = train_models(w_train, 'Women')


def build_sub_features(sub_df, stats, seed_df, mor_df, med):
    rows = []
    for _, r in sub_df.iterrows():
        s, t1, t2 = int(r['Season']), int(r['T1']), int(r['T2'])
        f1 = get_feats(s, t1, stats, seed_df, mor_df)
        f2 = get_feats(s, t2, stats, seed_df, mor_df)
        rows.append(dict(
            T1_WinRate=f1[0], T2_WinRate=f2[0],
            T1_Margin=f1[1],  T2_Margin=f2[1],
            T1_Score=f1[2],   T2_Score=f2[2],
            T1_Seed=f1[3],    T2_Seed=f2[3],
            T1_MOR=f1[4],     T2_MOR=f2[4],
            SeedDiff=   (f1[3] if not np.isnan(f1[3]) else np.nan) - (f2[3] if not np.isnan(f2[3]) else np.nan),
            WinRateDiff=(f1[0] if not np.isnan(f1[0]) else np.nan) - (f2[0] if not np.isnan(f2[0]) else np.nan),
            MarginDiff= (f1[1] if not np.isnan(f1[1]) else np.nan) - (f2[1] if not np.isnan(f2[1]) else np.nan),
            ScoreDiff=  (f1[2] if not np.isnan(f1[2]) else np.nan) - (f2[2] if not np.isnan(f2[2]) else np.nan),
            MORDiff=    (f1[4] if not np.isnan(f1[4]) else np.nan) - (f2[4] if not np.isnan(f2[4]) else np.nan),
        ))
    return pd.DataFrame(rows)[FEAT_COLS].fillna(med)


sub_stage1[['Season','T1','T2']] = sub_stage1['ID'].str.split('_', expand=True).astype(int)
m_team_ids = set(m_teams['TeamID'])
sub_stage1['Gender'] = sub_stage1['T1'].apply(lambda x: 'M' if x in m_team_ids else 'W')

m_sub = sub_stage1[sub_stage1['Gender'] == 'M'].copy()
X_m = build_sub_features(m_sub, m_stats, m_seeds, massey_agg, m_med)
m_sub['Pred'] = np.clip((m_lgb.predict_proba(X_m)[:, 1] + m_lr.predict_proba(m_scaler.transform(X_m))[:, 1]) / 2, 0.025, 0.975)

w_sub = sub_stage1[sub_stage1['Gender'] == 'W'].copy()
X_w = build_sub_features(w_sub, w_stats, w_seeds, None, w_med)
w_sub['Pred'] = np.clip((w_lgb.predict_proba(X_w)[:, 1] + w_lr.predict_proba(w_scaler.transform(X_w))[:, 1]) / 2, 0.025, 0.975)

submission = pd.concat([m_sub[['ID','Pred']], w_sub[['ID','Pred']]], ignore_index=True)
submission = submission.sort_values('ID').reset_index(drop=True)
submission.to_csv('submission.csv', index=False)
print(f'Saved submission.csv ({len(submission):,} rows)')
print(submission.head(10))

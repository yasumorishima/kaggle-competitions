"""March ML Mania 2026 v2 — Full Ensemble (LGB + XGB + CatBoost + LogReg)

Competition-grade notebook with:
- Unified Men + Women pipeline
- Elo ratings with season carryover
- Glicko-2 ratings (deviation + volatility)
- Bradley-Terry model via logistic regression
- Multi-Massey ordinals aggregation (Men only; Women NaN-filled)
- DetailedResults advanced stats (Four Factors, efficiency, etc.)
- Recent form (14-day and 30-day)
- Strength of schedule
- Conference strength
- Home/Away/Neutral splits
- Expanding Window CV (no future leakage)
- Multi-seed ensemble (5 seeds)
- Rank average + probability average blending
- LightGBM (GPU) + XGBoost (GPU) + CatBoost (GPU) + LogisticRegression
- W&B offline tracking
- Stage1 (historical) + Stage2 (2026) predictions
"""

import json

cells = []
cell_counter = 0


def add_md(source):
    global cell_counter
    cell_counter += 1
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "markdown",
        "id": f"cell-{cell_counter:03d}",
        "metadata": {},
        "source": src,
    })


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


# =============================================================================
# Cell: Title
# =============================================================================
add_md("""# March ML Mania 2026 v2 — Full Ensemble

**Private competition notebook** — LGB + XGB + CatBoost + LogReg

## Features
- Elo ratings (season carryover, tuned K-factor)
- Glicko-2 ratings (deviation + volatility)
- Bradley-Terry model (logistic regression on game outcomes)
- Multi-Massey ordinals (all ranking systems aggregated, Men only)
- DetailedResults: Four Factors, offensive/defensive efficiency, rebound rates
- Recent form (14-day, 30-day)
- Strength of schedule, Conference strength
- Home/Away/Neutral splits
- Expanding Window CV (train on seasons < S, validate on S)
- Multi-seed ensemble (5 seeds) with rank + probability blending
- GPU-accelerated: LightGBM, XGBoost, CatBoost""")

# =============================================================================
# Cell: Setup & Imports
# =============================================================================
add_code("""import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'march-machine-learning-mania-2026'

import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import wandb
import warnings
warnings.filterwarnings('ignore')

import glob as _glob
_slug = 'march-machine-learning-mania-2026'
_matches = _glob.glob(f'/kaggle/input/**/{_slug}', recursive=True)
DATA_DIR = Path(_matches[0]) if _matches else Path(f'/kaggle/input/{_slug}')
print(f'DATA_DIR: {DATA_DIR}')

# Verify data files
csv_files = sorted(DATA_DIR.glob('*.csv'))
print(f'CSV files found: {len(csv_files)}')
for f in csv_files:
    print(f'  {f.name}')

print('\\nLibraries loaded. GPU available for training.')""")

# =============================================================================
# Cell: W&B init
# =============================================================================
add_code("""run = wandb.init(
    project='march-machine-learning-mania-2026',
    name='v2-full-ensemble',
    tags=['elo', 'glicko2', 'bradley-terry', 'four-factors', 'catboost',
          'lgb', 'xgb', 'logreg', 'multi-seed', 'expanding-window'],
    config={
        'elo_k': 32,
        'elo_carryover': 0.5,
        'glicko2_tau': 0.5,
        'recent_days_short': 14,
        'recent_days_long': 30,
        'seeds': [42, 123, 2024, 777, 999],
        'clip_min': 0.025,
        'clip_max': 0.975,
        'models': ['lgb', 'xgb', 'catboost', 'logreg'],
    }
)
print(f'W&B run: {run.name}')""")

# =============================================================================
# Cell: Load all data
# =============================================================================
add_code("""# ── Men's data ──
m_teams   = pd.read_csv(DATA_DIR / 'MTeams.csv')
m_seeds   = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')
m_tourney = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
m_reg     = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
m_reg_det = pd.read_csv(DATA_DIR / 'MRegularSeasonDetailedResults.csv')
m_trn_det = pd.read_csv(DATA_DIR / 'MNCAATourneyDetailedResults.csv')
massey    = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')
m_conf    = pd.read_csv(DATA_DIR / 'MConferences.csv') if (DATA_DIR / 'MConferences.csv').exists() else None

# ── Women's data ──
w_teams   = pd.read_csv(DATA_DIR / 'WTeams.csv')
w_seeds   = pd.read_csv(DATA_DIR / 'WNCAATourneySeeds.csv')
w_tourney = pd.read_csv(DATA_DIR / 'WNCAATourneyCompactResults.csv')
w_reg     = pd.read_csv(DATA_DIR / 'WRegularSeasonCompactResults.csv')
w_reg_det = pd.read_csv(DATA_DIR / 'WRegularSeasonDetailedResults.csv')
w_trn_det = pd.read_csv(DATA_DIR / 'WNCAATourneyDetailedResults.csv')

# ── Submissions ──
sub_s1 = pd.read_csv(DATA_DIR / 'SampleSubmissionStage1.csv')
sub_s2_path = DATA_DIR / 'SampleSubmissionStage2.csv'
sub_s2 = pd.read_csv(sub_s2_path) if sub_s2_path.exists() else None

# ── Parse seeds ──
def parse_seed(s):
    return int(''.join(filter(str.isdigit, s)))

m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed)
w_seeds['SeedNum'] = w_seeds['Seed'].apply(parse_seed)

print(f"Men reg compact: {m_reg.shape}, detailed: {m_reg_det.shape}")
print(f"Women reg compact: {w_reg.shape}, detailed: {w_reg_det.shape}")
print(f"Massey systems: {massey['SystemName'].nunique()}")
print(f"Stage1 submission: {len(sub_s1):,} rows")
if sub_s2 is not None:
    print(f"Stage2 submission: {len(sub_s2):,} rows")
else:
    print("Stage2 submission: not yet available")""")

# =============================================================================
# Cell: Elo rating system
# =============================================================================
add_code("""def compute_elo(games_df, k=32, initial=1500, carryover=0.5, home_adv=0):
    \"\"\"Compute end-of-season Elo ratings with season carryover.

    Returns DataFrame with (Season, TeamID, Elo) — one row per team-season.
    Also returns the full elo dict for pre-tourney ratings.
    \"\"\"
    elo = {}
    records = []

    for season in sorted(games_df['Season'].unique()):
        # Season start: decay all ratings toward mean
        for tid in elo:
            elo[tid] = initial + carryover * (elo[tid] - initial)

        season_games = games_df[games_df['Season'] == season].sort_values('DayNum')

        for _, row in season_games.iterrows():
            w, l = row['WTeamID'], row['LTeamID']
            elo.setdefault(w, initial)
            elo.setdefault(l, initial)

            # Home advantage adjustment
            ha = 0
            if 'WLoc' in row and row['WLoc'] == 'H':
                ha = home_adv
            elif 'WLoc' in row and row['WLoc'] == 'A':
                ha = -home_adv

            exp_w = 1 / (1 + 10 ** (-(elo[w] - elo[l] + ha) / 400))
            update = k * (1 - exp_w)

            # Margin of victory multiplier (capped)
            if 'WScore' in row.index and 'LScore' in row.index:
                margin = row['WScore'] - row['LScore']
                mov_mult = math.log(max(margin, 1) + 1) * (2.2 / ((elo[w] - elo[l]) * 0.001 + 2.2))
                update *= mov_mult

            elo[w] += update
            elo[l] -= update

        # Record end-of-season Elo
        teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in teams:
            records.append({'Season': season, 'TeamID': tid, 'Elo': elo.get(tid, initial)})

    return pd.DataFrame(records)


m_elo = compute_elo(m_reg, k=32, initial=1500, carryover=0.5, home_adv=100)
w_elo = compute_elo(w_reg, k=32, initial=1500, carryover=0.5, home_adv=100)

print(f"Men Elo range: {m_elo['Elo'].min():.0f} - {m_elo['Elo'].max():.0f}")
print(f"Women Elo range: {w_elo['Elo'].min():.0f} - {w_elo['Elo'].max():.0f}")""")

# =============================================================================
# Cell: Glicko-2 rating system
# =============================================================================
add_code("""class Glicko2:
    \"\"\"Glicko-2 rating system implementation.

    Tracks rating (mu), rating deviation (phi), and volatility (sigma).
    Better than Elo because it accounts for uncertainty in ratings.
    \"\"\"
    def __init__(self, mu=1500, phi=350, sigma=0.06, tau=0.5):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.tau = tau
        self.SCALE = 173.7178

    def _g(self, phi):
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def _E(self, mu, mu_j, phi_j):
        return 1 / (1 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def rate(self, rating, opponents, outcomes):
        \"\"\"Update rating based on a batch of games.

        rating: (mu, phi, sigma) tuple
        opponents: list of (mu, phi) tuples
        outcomes: list of 1 (win) or 0 (loss)
        \"\"\"
        mu, phi, sigma = rating
        # Convert to Glicko-2 scale
        mu_g2 = (mu - 1500) / self.SCALE
        phi_g2 = phi / self.SCALE

        if len(opponents) == 0:
            # No games: increase uncertainty
            phi_new = math.sqrt(phi_g2**2 + sigma**2)
            return (mu, phi_new * self.SCALE, sigma)

        opp_g2 = [((m - 1500) / self.SCALE, p / self.SCALE) for m, p in opponents]

        # Step 3: Compute v (estimated variance)
        v_inv = 0
        delta_sum = 0
        for (mu_j, phi_j), s in zip(opp_g2, outcomes):
            g_j = self._g(phi_j)
            E_j = self._E(mu_g2, mu_j, phi_j)
            v_inv += g_j**2 * E_j * (1 - E_j)
            delta_sum += g_j * (s - E_j)

        v = 1 / v_inv if v_inv > 0 else 1e6
        delta = v * delta_sum

        # Step 4: Compute new volatility (simplified Illinois algorithm)
        a = math.log(sigma**2)
        tau2 = self.tau**2

        def f(x):
            ex = math.exp(x)
            d2 = delta**2
            p2 = phi_g2**2
            num1 = ex * (d2 - p2 - v - ex)
            den1 = 2 * (p2 + v + ex)**2
            return num1 / den1 - (x - a) / tau2

        # Bracket the root
        A = a
        if delta**2 > phi_g2**2 + v:
            B = math.log(delta**2 - phi_g2**2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        # Illinois method
        fA, fB = f(A), f(B)
        for _ in range(50):
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC
            if abs(B - A) < 1e-6:
                break

        sigma_new = math.exp(B / 2)

        # Step 5-6: Update rating and deviation
        phi_star = math.sqrt(phi_g2**2 + sigma_new**2)
        phi_new = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
        mu_new = mu_g2 + phi_new**2 * delta_sum

        # Convert back
        return (mu_new * self.SCALE + 1500, phi_new * self.SCALE, sigma_new)


def compute_glicko2(games_df, tau=0.5, carryover=0.5):
    \"\"\"Compute Glicko-2 ratings per team-season.\"\"\"
    g2 = Glicko2(tau=tau)
    # (mu, phi, sigma) per team
    ratings = {}
    records = []
    INIT = (1500, 350, 0.06)

    for season in sorted(games_df['Season'].unique()):
        # Season start: decay mu toward mean, reset phi partially
        for tid in ratings:
            mu, phi, sigma = ratings[tid]
            mu_new = 1500 + carryover * (mu - 1500)
            phi_new = min(350, math.sqrt(phi**2 + 50**2))  # increase uncertainty between seasons
            ratings[tid] = (mu_new, phi_new, sigma)

        # Collect all games per team for batch update
        season_games = games_df[games_df['Season'] == season].sort_values('DayNum')

        # Process in day-batches for Glicko-2 (batch per rating period)
        day_groups = season_games.groupby('DayNum')

        for day, day_games in day_groups:
            # Collect opponents and outcomes for each team
            team_opponents = defaultdict(lambda: ([], []))  # (opponents, outcomes)

            for _, row in day_games.iterrows():
                w, l = row['WTeamID'], row['LTeamID']
                ratings.setdefault(w, INIT)
                ratings.setdefault(l, INIT)

                mu_w, phi_w, _ = ratings[w]
                mu_l, phi_l, _ = ratings[l]

                team_opponents[w][0].append((mu_l, phi_l))
                team_opponents[w][1].append(1)
                team_opponents[l][0].append((mu_w, phi_w))
                team_opponents[l][1].append(0)

            # Update all teams that played
            for tid, (opps, outs) in team_opponents.items():
                ratings[tid] = g2.rate(ratings[tid], opps, outs)

        # Record end-of-season ratings
        teams = set(season_games['WTeamID']) | set(season_games['LTeamID'])
        for tid in teams:
            mu, phi, sigma = ratings.get(tid, INIT)
            records.append({
                'Season': season, 'TeamID': tid,
                'Glicko2Mu': mu, 'Glicko2Phi': phi, 'Glicko2Sigma': sigma
            })

    return pd.DataFrame(records)


m_glicko = compute_glicko2(m_reg, tau=0.5, carryover=0.5)
w_glicko = compute_glicko2(w_reg, tau=0.5, carryover=0.5)

print(f"Men Glicko-2 mu range: {m_glicko['Glicko2Mu'].min():.0f} - {m_glicko['Glicko2Mu'].max():.0f}")
print(f"Women Glicko-2 mu range: {w_glicko['Glicko2Mu'].min():.0f} - {w_glicko['Glicko2Mu'].max():.0f}")""")

# =============================================================================
# Cell: Bradley-Terry model
# =============================================================================
add_code("""def compute_bradley_terry(games_df, max_iter=200, tol=1e-8):
    \"\"\"Compute Bradley-Terry strength via MM algorithm (Hunter 2004).

    For each season, estimate team strength parameters π_i such that
    P(i beats j) = π_i / (π_i + π_j). Uses the iterative Minorization-
    Maximization algorithm which is guaranteed to converge.

    Returns log-strength (log π_i, zero-centered) per (Season, TeamID).
    \"\"\"
    records = []

    for season in sorted(games_df['Season'].unique()):
        season_games = games_df[games_df['Season'] == season]
        teams = sorted(set(season_games['WTeamID']) | set(season_games['LTeamID']))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        if n_teams < 5:
            continue

        # Build win counts and matchup counts
        # wins[i] = total wins for team i
        # matchups[i][j] = number of games between i and j
        wins = np.zeros(n_teams)
        matchups = np.zeros((n_teams, n_teams))

        for _, row in season_games.iterrows():
            w_idx = team_to_idx[row['WTeamID']]
            l_idx = team_to_idx[row['LTeamID']]
            wins[w_idx] += 1
            matchups[w_idx][l_idx] += 1
            matchups[l_idx][w_idx] += 1

        # MM algorithm (vectorized): π_i = w_i / Σ_j (n_ij / (π_i + π_j))
        pi = np.ones(n_teams)

        for iteration in range(max_iter):
            pi_old = pi.copy()

            # Broadcast: pi_sum[i,j] = pi[i] + pi[j]
            pi_sum = pi[:, None] + pi[None, :]
            # denom[i] = Σ_j matchups[i,j] / (pi[i] + pi[j])
            denom = np.sum(matchups / np.where(pi_sum > 0, pi_sum, 1.0), axis=1)

            # Update: teams with 0 wins get minimum strength
            pi = np.where((denom > 0) & (wins > 0), wins / denom, 1e-10)

            # Normalize so geometric mean = 1
            pi = pi / np.exp(np.mean(np.log(np.maximum(pi, 1e-20))))

            # Check convergence
            if np.max(np.abs(np.log(np.maximum(pi, 1e-20)) - np.log(np.maximum(pi_old, 1e-20)))) < tol:
                break

        # Convert to log-strength (zero-centered)
        log_strength = np.log(np.maximum(pi, 1e-20))
        log_strength -= log_strength.mean()

        for tid, idx in team_to_idx.items():
            records.append({
                'Season': season, 'TeamID': tid,
                'BradleyTerry': log_strength[idx]
            })

    return pd.DataFrame(records)


m_bt = compute_bradley_terry(m_reg)
w_bt = compute_bradley_terry(w_reg)

print(f"Men BT range: {m_bt['BradleyTerry'].min():.3f} - {m_bt['BradleyTerry'].max():.3f}")
print(f"Women BT range: {w_bt['BradleyTerry'].min():.3f} - {w_bt['BradleyTerry'].max():.3f}")""")

# =============================================================================
# Cell: Multi-Massey ordinals aggregation
# =============================================================================
add_code("""def aggregate_massey(massey_df):
    \"\"\"Aggregate all Massey ranking systems — use last available day per season.

    Returns per (Season, TeamID): AvgRank, MedianRank, BestRank, NumSystems
    \"\"\"
    # For each season-system, use the last ranking day
    last_day = (
        massey_df
        .sort_values(['Season', 'RankingDayNum'])
        .groupby(['Season', 'SystemName', 'TeamID']).last()
        .reset_index()
    )

    agg = (
        last_day
        .groupby(['Season', 'TeamID'])
        .agg(
            MasseyAvgRank=('OrdinalRank', 'mean'),
            MasseyMedianRank=('OrdinalRank', 'median'),
            MasseyBestRank=('OrdinalRank', 'min'),
            MasseyWorstRank=('OrdinalRank', 'max'),
            MasseyNumSystems=('OrdinalRank', 'count'),
            MasseyStdRank=('OrdinalRank', 'std'),
        )
        .reset_index()
    )
    agg['MasseyStdRank'] = agg['MasseyStdRank'].fillna(0)
    return agg


massey_agg = aggregate_massey(massey)

print(f"Massey aggregated: {massey_agg.shape}")
print(f"AvgRank range: {massey_agg['MasseyAvgRank'].min():.1f} - {massey_agg['MasseyAvgRank'].max():.1f}")
print(f"Systems per team: {massey_agg['MasseyNumSystems'].describe()}")""")

# =============================================================================
# Cell: DetailedResults feature engineering
# =============================================================================
add_code("""def compute_detailed_stats(detailed_df):
    \"\"\"Compute per-team-season advanced stats from DetailedResults.

    Features: shooting percentages, efficiency, Four Factors, rebound/assist/turnover rates.
    \"\"\"
    rows = []

    # Winner stats
    w = detailed_df.copy()
    w_stats = pd.DataFrame({
        'Season': w['Season'],
        'TeamID': w['WTeamID'],
        'Score': w['WScore'],
        'OppScore': w['LScore'],
        'FGM': w['WFGM'], 'FGA': w['WFGA'],
        'FGM3': w['WFGM3'], 'FGA3': w['WFGA3'],
        'FTM': w['WFTM'], 'FTA': w['WFTA'],
        'OR': w['WOR'], 'DR': w['WDR'],
        'Ast': w['WAst'], 'TO': w['WTO'],
        'Stl': w['WStl'], 'Blk': w['WBlk'],
        'PF': w['WPF'],
        # Opponent stats for defensive metrics
        'OppFGM': w['LFGM'], 'OppFGA': w['LFGA'],
        'OppFGM3': w['LFGM3'], 'OppFGA3': w['LFGA3'],
        'OppFTM': w['LFTM'], 'OppFTA': w['LFTA'],
        'OppOR': w['LOR'], 'OppDR': w['LDR'],
        'OppTO': w['LTO'],
    })

    # Loser stats
    l_stats = pd.DataFrame({
        'Season': w['Season'],
        'TeamID': w['LTeamID'],
        'Score': w['LScore'],
        'OppScore': w['WScore'],
        'FGM': w['LFGM'], 'FGA': w['LFGA'],
        'FGM3': w['LFGM3'], 'FGA3': w['LFGA3'],
        'FTM': w['LFTM'], 'FTA': w['LFTA'],
        'OR': w['LOR'], 'DR': w['LDR'],
        'Ast': w['LAst'], 'TO': w['LTO'],
        'Stl': w['LStl'], 'Blk': w['LBlk'],
        'PF': w['LPF'],
        'OppFGM': w['WFGM'], 'OppFGA': w['WFGA'],
        'OppFGM3': w['WFGM3'], 'OppFGA3': w['WFGA3'],
        'OppFTM': w['WFTM'], 'OppFTA': w['WFTA'],
        'OppOR': w['WOR'], 'OppDR': w['WDR'],
        'OppTO': w['WTO'],
    })

    all_games = pd.concat([w_stats, l_stats], ignore_index=True)

    # Possession estimation: FGA - OR + TO + 0.475 * FTA
    all_games['Poss'] = all_games['FGA'] - all_games['OR'] + all_games['TO'] + 0.475 * all_games['FTA']
    all_games['OppPoss'] = all_games['OppFGA'] - all_games['OppOR'] + all_games['OppTO'] + 0.475 * all_games['OppFTA']

    # Per-game stats
    all_games['FGPct'] = all_games['FGM'] / all_games['FGA'].clip(1)
    all_games['FG3Pct'] = all_games['FGM3'] / all_games['FGA3'].clip(1)
    all_games['FTPct'] = all_games['FTM'] / all_games['FTA'].clip(1)

    # Four Factors (Offense)
    all_games['eFGPct'] = (all_games['FGM'] + 0.5 * all_games['FGM3']) / all_games['FGA'].clip(1)
    all_games['TORate'] = all_games['TO'] / all_games['Poss'].clip(1)
    all_games['ORBRate'] = all_games['OR'] / (all_games['OR'] + all_games['OppDR']).clip(1)
    all_games['FTRate'] = all_games['FTM'] / all_games['FGA'].clip(1)

    # Four Factors (Defense)
    all_games['OppeFGPct'] = (all_games['OppFGM'] + 0.5 * all_games['OppFGM3']) / all_games['OppFGA'].clip(1)
    all_games['OppTORate'] = all_games['OppTO'] / all_games['OppPoss'].clip(1)
    all_games['DRBRate'] = all_games['DR'] / (all_games['DR'] + all_games['OppOR']).clip(1)
    all_games['OppFTRate'] = all_games['OppFTM'] / all_games['OppFGA'].clip(1)

    # Efficiency
    all_games['OffEff'] = all_games['Score'] / all_games['Poss'].clip(1) * 100
    all_games['DefEff'] = all_games['OppScore'] / all_games['OppPoss'].clip(1) * 100
    all_games['NetEff'] = all_games['OffEff'] - all_games['DefEff']

    # Rate stats
    all_games['AstRate'] = all_games['Ast'] / all_games['FGM'].clip(1)
    all_games['StlRate'] = all_games['Stl'] / all_games['OppPoss'].clip(1)
    all_games['BlkRate'] = all_games['Blk'] / (all_games['OppFGA'] - all_games['OppFGA3']).clip(1)

    # Aggregate per team-season
    agg_cols = [
        'FGPct', 'FG3Pct', 'FTPct',
        'eFGPct', 'TORate', 'ORBRate', 'FTRate',
        'OppeFGPct', 'OppTORate', 'DRBRate', 'OppFTRate',
        'OffEff', 'DefEff', 'NetEff',
        'AstRate', 'StlRate', 'BlkRate',
        'Poss', 'Score', 'OppScore',
    ]

    result = all_games.groupby(['Season', 'TeamID'])[agg_cols].mean().reset_index()

    # Rename with prefix for clarity
    rename_map = {c: f'Det_{c}' for c in agg_cols}
    result = result.rename(columns=rename_map)

    return result


m_det_stats = compute_detailed_stats(m_reg_det)
w_det_stats = compute_detailed_stats(w_reg_det)

print(f"Men detailed stats: {m_det_stats.shape}")
print(f"Women detailed stats: {w_det_stats.shape}")
print(f"Columns: {list(m_det_stats.columns)}")""")

# =============================================================================
# Cell: Recent form (14 and 30 days)
# =============================================================================
add_code("""def build_recent_form(reg_df, reg_det_df, days_back, suffix):
    \"\"\"Win rate and scoring metrics in the last N days of regular season.\"\"\"
    max_day = reg_df.groupby('Season')['DayNum'].max()
    rows = []

    for season, grp in reg_df.groupby('Season'):
        cutoff = max_day[season] - days_back
        recent = grp[grp['DayNum'] >= cutoff]
        for _, r in recent.iterrows():
            rows.append({
                'Season': season, 'TeamID': r['WTeamID'],
                'Win': 1, 'Score': r['WScore'], 'OppScore': r['LScore'],
                'Margin': r['WScore'] - r['LScore'],
            })
            rows.append({
                'Season': season, 'TeamID': r['LTeamID'],
                'Win': 0, 'Score': r['LScore'], 'OppScore': r['WScore'],
                'Margin': r['LScore'] - r['WScore'],
            })

    rf = pd.DataFrame(rows)
    if len(rf) == 0:
        return pd.DataFrame(columns=['Season', 'TeamID'])

    result = (
        rf.groupby(['Season', 'TeamID']).agg(
            WinRate=('Win', 'mean'),
            AvgMargin=('Margin', 'mean'),
            AvgScore=('Score', 'mean'),
            Games=('Win', 'count'),
        ).reset_index()
    )

    rename_map = {
        'WinRate': f'Recent{suffix}_WinRate',
        'AvgMargin': f'Recent{suffix}_AvgMargin',
        'AvgScore': f'Recent{suffix}_AvgScore',
        'Games': f'Recent{suffix}_Games',
    }
    result = result.rename(columns=rename_map)
    return result


# Short form (14 days) and Long form (30 days)
m_recent14 = build_recent_form(m_reg, m_reg_det, 14, '14')
m_recent30 = build_recent_form(m_reg, m_reg_det, 30, '30')
w_recent14 = build_recent_form(w_reg, w_reg_det, 14, '14')
w_recent30 = build_recent_form(w_reg, w_reg_det, 30, '30')

print(f"Men recent14: {m_recent14.shape}, recent30: {m_recent30.shape}")
print(f"Women recent14: {w_recent14.shape}, recent30: {w_recent30.shape}")""")

# =============================================================================
# Cell: Strength of schedule
# =============================================================================
add_code("""def compute_strength_of_schedule(reg_df):
    \"\"\"Average opponent win rate per team-season.\"\"\"
    # First, compute win rate per team-season
    w = reg_df[['Season', 'WTeamID']].copy().rename(columns={'WTeamID': 'TeamID'})
    w['Win'] = 1
    l = reg_df[['Season', 'LTeamID']].copy().rename(columns={'LTeamID': 'TeamID'})
    l['Win'] = 0
    all_results = pd.concat([w, l], ignore_index=True)
    win_rates = all_results.groupby(['Season', 'TeamID'])['Win'].mean().reset_index()
    win_rates.columns = ['Season', 'TeamID', 'WinRate']

    # For each game, record opponent
    opp_rows = []
    for _, row in reg_df.iterrows():
        opp_rows.append({'Season': row['Season'], 'TeamID': row['WTeamID'], 'OppID': row['LTeamID']})
        opp_rows.append({'Season': row['Season'], 'TeamID': row['LTeamID'], 'OppID': row['WTeamID']})

    opp_df = pd.DataFrame(opp_rows)
    opp_df = opp_df.merge(
        win_rates.rename(columns={'TeamID': 'OppID', 'WinRate': 'OppWinRate'}),
        on=['Season', 'OppID'], how='left'
    )

    sos = opp_df.groupby(['Season', 'TeamID'])['OppWinRate'].mean().reset_index()
    sos.columns = ['Season', 'TeamID', 'SOS']

    return sos


m_sos = compute_strength_of_schedule(m_reg)
w_sos = compute_strength_of_schedule(w_reg)

print(f"Men SOS range: {m_sos['SOS'].min():.3f} - {m_sos['SOS'].max():.3f}")
print(f"Women SOS range: {w_sos['SOS'].min():.3f} - {w_sos['SOS'].max():.3f}")""")

# =============================================================================
# Cell: Conference strength
# =============================================================================
add_code("""def compute_conference_strength(reg_df, conf_df, elo_df):
    \"\"\"Average Elo per conference-season. If no conference data, return empty.\"\"\"
    if conf_df is None:
        return pd.DataFrame(columns=['Season', 'TeamID', 'ConfStrength'])

    # Merge team-conference mapping with elo
    merged = conf_df.merge(elo_df, on=['Season', 'TeamID'], how='inner')

    conf_avg = merged.groupby(['Season', 'ConfAbbrev'])['Elo'].mean().reset_index()
    conf_avg.columns = ['Season', 'ConfAbbrev', 'ConfStrength']

    result = conf_df.merge(conf_avg, on=['Season', 'ConfAbbrev'], how='left')
    return result[['Season', 'TeamID', 'ConfStrength']]


m_conf_str = compute_conference_strength(m_reg, m_conf, m_elo)
# Women don't have conference data in standard format
w_conf_str = pd.DataFrame(columns=['Season', 'TeamID', 'ConfStrength'])

if len(m_conf_str) > 0:
    print(f"Men ConfStrength range: {m_conf_str['ConfStrength'].min():.0f} - {m_conf_str['ConfStrength'].max():.0f}")
else:
    print("Men ConfStrength: no data")
print(f"Women ConfStrength: no data (expected)")""")

# =============================================================================
# Cell: Home/Away/Neutral splits
# =============================================================================
add_code("""def compute_location_splits(reg_df):
    \"\"\"Win rate at Home / Away / Neutral per team-season.\"\"\"
    if 'WLoc' not in reg_df.columns:
        return pd.DataFrame(columns=['Season', 'TeamID'])

    rows = []
    for _, r in reg_df.iterrows():
        wloc = r.get('WLoc', 'N')
        rows.append({'Season': r['Season'], 'TeamID': r['WTeamID'], 'Win': 1, 'Loc': wloc})
        # Loser location is the mirror
        l_loc = 'A' if wloc == 'H' else ('H' if wloc == 'A' else 'N')
        rows.append({'Season': r['Season'], 'TeamID': r['LTeamID'], 'Win': 0, 'Loc': l_loc})

    df = pd.DataFrame(rows)

    # Win rate per location
    pivoted = df.groupby(['Season', 'TeamID', 'Loc'])['Win'].mean().unstack(fill_value=np.nan)
    pivoted.columns = [f'WinRate_{c}' for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Ensure all columns exist
    for col in ['WinRate_H', 'WinRate_A', 'WinRate_N']:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    return pivoted[['Season', 'TeamID', 'WinRate_H', 'WinRate_A', 'WinRate_N']]


m_loc = compute_location_splits(m_reg)
w_loc = compute_location_splits(w_reg)

print(f"Men location splits: {m_loc.shape}")
if len(m_loc) > 0:
    print(m_loc[['WinRate_H', 'WinRate_A', 'WinRate_N']].describe().round(3))""")

# =============================================================================
# Cell: Season-level basic stats
# =============================================================================
add_code("""def build_season_stats(reg_df):
    \"\"\"Basic season stats: win rate, average margin, average score.\"\"\"
    w = reg_df[['Season', 'WTeamID', 'WScore', 'LScore']].copy()
    w.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    w['Win'] = 1
    l = reg_df[['Season', 'LTeamID', 'LScore', 'WScore']].copy()
    l.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst']
    l['Win'] = 0
    df = pd.concat([w, l], ignore_index=True)
    df['Margin'] = df['ScoreFor'] - df['ScoreAgainst']

    stats = df.groupby(['Season', 'TeamID']).agg(
        WinRate=('Win', 'mean'),
        AvgMargin=('Margin', 'mean'),
        AvgScore=('ScoreFor', 'mean'),
        AvgOppScore=('ScoreAgainst', 'mean'),
        Games=('Win', 'count'),
        StdMargin=('Margin', 'std'),
    ).reset_index()
    stats['StdMargin'] = stats['StdMargin'].fillna(0)
    return stats


m_stats = build_season_stats(m_reg)
w_stats = build_season_stats(w_reg)

print(f"Men season stats: {m_stats.shape}")
print(f"Women season stats: {w_stats.shape}")""")

# =============================================================================
# Cell: Merge all team features
# =============================================================================
add_code("""def merge_all_team_features(stats, seeds, elo, glicko, bt, det_stats,
                              recent14, recent30, sos, conf_str, loc,
                              massey_data=None):
    \"\"\"Merge all features into one DataFrame per (Season, TeamID).\"\"\"
    df = stats.copy()

    # Core
    df = df.merge(seeds[['Season', 'TeamID', 'SeedNum']], on=['Season', 'TeamID'], how='left')
    df = df.merge(elo, on=['Season', 'TeamID'], how='left')
    df = df.merge(glicko, on=['Season', 'TeamID'], how='left')
    df = df.merge(bt, on=['Season', 'TeamID'], how='left')

    # Detailed stats
    df = df.merge(det_stats, on=['Season', 'TeamID'], how='left')

    # Recent form
    df = df.merge(recent14, on=['Season', 'TeamID'], how='left')
    df = df.merge(recent30, on=['Season', 'TeamID'], how='left')

    # SOS & Conference
    df = df.merge(sos, on=['Season', 'TeamID'], how='left')
    if len(conf_str) > 0:
        df = df.merge(conf_str, on=['Season', 'TeamID'], how='left')
    else:
        df['ConfStrength'] = np.nan

    # Location splits
    df = df.merge(loc, on=['Season', 'TeamID'], how='left')

    # Massey (Men only)
    if massey_data is not None:
        df = df.merge(massey_data, on=['Season', 'TeamID'], how='left')

    return df


m_feats = merge_all_team_features(
    m_stats, m_seeds, m_elo, m_glicko, m_bt, m_det_stats,
    m_recent14, m_recent30, m_sos, m_conf_str, m_loc,
    massey_data=massey_agg
)
w_feats = merge_all_team_features(
    w_stats, w_seeds, w_elo, w_glicko, w_bt, w_det_stats,
    w_recent14, w_recent30, w_sos, w_conf_str, w_loc,
    massey_data=None  # Women don't have Massey
)

print(f"Men features: {m_feats.shape}, columns: {len(m_feats.columns)}")
print(f"Women features: {w_feats.shape}, columns: {len(w_feats.columns)}")
print(f"\\nMen feature columns:")
for c in sorted(m_feats.columns):
    print(f"  {c}: NaN={m_feats[c].isna().mean():.1%}")""")

# =============================================================================
# Cell: Define feature columns for matchup
# =============================================================================
add_code("""# All per-team feature columns (these get T1_ and T2_ prefixes in matchup data)
TEAM_FEAT_COLS = [
    'SeedNum', 'Elo', 'Glicko2Mu', 'Glicko2Phi', 'BradleyTerry',
    'WinRate', 'AvgMargin', 'AvgScore', 'AvgOppScore', 'Games', 'StdMargin',
    'Det_FGPct', 'Det_FG3Pct', 'Det_FTPct',
    'Det_eFGPct', 'Det_TORate', 'Det_ORBRate', 'Det_FTRate',
    'Det_OppeFGPct', 'Det_OppTORate', 'Det_DRBRate', 'Det_OppFTRate',
    'Det_OffEff', 'Det_DefEff', 'Det_NetEff',
    'Det_AstRate', 'Det_StlRate', 'Det_BlkRate',
    'Det_Poss',
    'Recent14_WinRate', 'Recent14_AvgMargin',
    'Recent30_WinRate', 'Recent30_AvgMargin',
    'SOS', 'ConfStrength',
    'WinRate_H', 'WinRate_A', 'WinRate_N',
    'MasseyAvgRank', 'MasseyMedianRank', 'MasseyBestRank',
    'MasseyWorstRank', 'MasseyNumSystems', 'MasseyStdRank',
]

# Difference features (T1 - T2)
DIFF_FEAT_NAMES = [
    'SeedDiff', 'EloDiff', 'Glicko2MuDiff', 'Glicko2PhiDiff', 'BTDiff',
    'WinRateDiff', 'MarginDiff', 'ScoreDiff', 'OppScoreDiff',
    'eFGDiff', 'TODiff', 'ORBDiff', 'FTRateDiff',
    'OppeFGDiff', 'OppTODiff', 'DRBDiff', 'OppFTRateDiff',
    'OffEffDiff', 'DefEffDiff', 'NetEffDiff',
    'AstRateDiff', 'StlRateDiff', 'BlkRateDiff',
    'PossDiff',
    'Recent14WRDiff', 'Recent14MarginDiff',
    'Recent30WRDiff', 'Recent30MarginDiff',
    'SOSDiff', 'ConfStrDiff',
    'MasseyAvgDiff', 'MasseyMedianDiff', 'MasseyBestDiff',
]

# Map: diff_name -> (T1_col, T2_col)
DIFF_MAP = {
    'SeedDiff': ('SeedNum', 'SeedNum'),
    'EloDiff': ('Elo', 'Elo'),
    'Glicko2MuDiff': ('Glicko2Mu', 'Glicko2Mu'),
    'Glicko2PhiDiff': ('Glicko2Phi', 'Glicko2Phi'),
    'BTDiff': ('BradleyTerry', 'BradleyTerry'),
    'WinRateDiff': ('WinRate', 'WinRate'),
    'MarginDiff': ('AvgMargin', 'AvgMargin'),
    'ScoreDiff': ('AvgScore', 'AvgScore'),
    'OppScoreDiff': ('AvgOppScore', 'AvgOppScore'),
    'eFGDiff': ('Det_eFGPct', 'Det_eFGPct'),
    'TODiff': ('Det_TORate', 'Det_TORate'),
    'ORBDiff': ('Det_ORBRate', 'Det_ORBRate'),
    'FTRateDiff': ('Det_FTRate', 'Det_FTRate'),
    'OppeFGDiff': ('Det_OppeFGPct', 'Det_OppeFGPct'),
    'OppTODiff': ('Det_OppTORate', 'Det_OppTORate'),
    'DRBDiff': ('Det_DRBRate', 'Det_DRBRate'),
    'OppFTRateDiff': ('Det_OppFTRate', 'Det_OppFTRate'),
    'OffEffDiff': ('Det_OffEff', 'Det_OffEff'),
    'DefEffDiff': ('Det_DefEff', 'Det_DefEff'),
    'NetEffDiff': ('Det_NetEff', 'Det_NetEff'),
    'AstRateDiff': ('Det_AstRate', 'Det_AstRate'),
    'StlRateDiff': ('Det_StlRate', 'Det_StlRate'),
    'BlkRateDiff': ('Det_BlkRate', 'Det_BlkRate'),
    'PossDiff': ('Det_Poss', 'Det_Poss'),
    'Recent14WRDiff': ('Recent14_WinRate', 'Recent14_WinRate'),
    'Recent14MarginDiff': ('Recent14_AvgMargin', 'Recent14_AvgMargin'),
    'Recent30WRDiff': ('Recent30_WinRate', 'Recent30_WinRate'),
    'Recent30MarginDiff': ('Recent30_AvgMargin', 'Recent30_AvgMargin'),
    'SOSDiff': ('SOS', 'SOS'),
    'ConfStrDiff': ('ConfStrength', 'ConfStrength'),
    'MasseyAvgDiff': ('MasseyAvgRank', 'MasseyAvgRank'),
    'MasseyMedianDiff': ('MasseyMedianRank', 'MasseyMedianRank'),
    'MasseyBestDiff': ('MasseyBestRank', 'MasseyBestRank'),
}

# All feature columns used in the model
# Diffs + T1/T2 individual features for key metrics
INDIVIDUAL_FEATS = [
    'T1_SeedNum', 'T2_SeedNum',
    'T1_Elo', 'T2_Elo',
    'T1_Glicko2Mu', 'T2_Glicko2Mu',
    'T1_Glicko2Phi', 'T2_Glicko2Phi',
    'T1_WinRate', 'T2_WinRate',
    'T1_Det_OffEff', 'T2_Det_OffEff',
    'T1_Det_DefEff', 'T2_Det_DefEff',
    'T1_Det_NetEff', 'T2_Det_NetEff',
    'T1_SOS', 'T2_SOS',
    'T1_MasseyAvgRank', 'T2_MasseyAvgRank',
]

FEAT_COLS = list(DIFF_FEAT_NAMES) + INDIVIDUAL_FEATS

print(f"Total features: {len(FEAT_COLS)}")
print(f"  Diff features: {len(DIFF_FEAT_NAMES)}")
print(f"  Individual features: {len(INDIVIDUAL_FEATS)}")""")

# =============================================================================
# Cell: Build matchup training data
# =============================================================================
add_code("""def build_matchup_df(tourney_df, feats_df):
    \"\"\"Build matchup-level training data from tournament results.

    Always sort TeamIDs so T1 < T2 for consistency.
    Label = 1 if T1 (lower ID) won, 0 if T2 won.
    \"\"\"
    feat_idx = feats_df.set_index(['Season', 'TeamID'])
    rows = []

    for _, r in tourney_df.iterrows():
        s = r['Season']
        t1, t2 = sorted([r['WTeamID'], r['LTeamID']])
        label = 1 if r['WTeamID'] == t1 else 0

        row = dict(Season=s, T1=t1, T2=t2, Label=label)

        # Pull all team features
        for col in TEAM_FEAT_COLS:
            for prefix, tid in [('T1', t1), ('T2', t2)]:
                try:
                    val = feat_idx.loc[(s, tid), col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                except (KeyError, IndexError):
                    val = np.nan
                row[f'{prefix}_{col}'] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute diff features
    for diff_name, (col1, col2) in DIFF_MAP.items():
        df[diff_name] = df[f'T1_{col1}'] - df[f'T2_{col2}']

    return df


# Build Men and Women training data
m_train = build_matchup_df(m_tourney, m_feats)
w_train = build_matchup_df(w_tourney, w_feats)

# Add gender indicator
m_train['IsWomen'] = 0
w_train['IsWomen'] = 1

# Combine for unified model
train_all = pd.concat([m_train, w_train], ignore_index=True)

print(f"Men train: {m_train.shape}, label balance: {m_train['Label'].mean():.3f}")
print(f"Women train: {w_train.shape}, label balance: {w_train['Label'].mean():.3f}")
print(f"Combined train: {train_all.shape}")
print(f"\\nNaN ratio (top 15):")
nan_ratio = train_all[FEAT_COLS].isna().mean().sort_values(ascending=False)
print(nan_ratio.head(15).to_string())""")

# =============================================================================
# Cell: Model hyperparameters
# =============================================================================
add_code("""SEEDS = [42, 123, 2024, 777, 999]

lgb_params = dict(
    objective='binary',
    metric='binary_logloss',
    verbosity=-1,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    device='gpu',
    gpu_use_dp=False,
)

xgb_params = dict(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method='hist',
    device='cuda',
    verbosity=0,
)

cat_params = dict(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    subsample=0.8,
    random_strength=1,
    task_type='GPU',
    verbose=0,
    eval_metric='Logloss',
)

print("Model hyperparameters configured.")
print(f"Seeds: {SEEDS}")
print(f"LGB device: {lgb_params['device']}")
print(f"XGB device: {xgb_params['device']}")
print(f"CatBoost: {cat_params['task_type']}")""")

# =============================================================================
# Cell: Expanding Window CV (no future leakage)
# =============================================================================
add_code("""def expanding_window_cv(train_df, feat_cols, min_train_seasons=5):
    \"\"\"Expanding window CV: train on seasons < S, validate on season S.

    This prevents any future data leakage — each validation year only uses
    data from prior years for training.
    \"\"\"
    seasons = sorted(train_df['Season'].unique())
    results = {model: [] for model in ['lgb', 'xgb', 'cat', 'logreg', 'ensemble']}
    all_oof_preds = []

    # Need at least min_train_seasons to start validation
    start_idx = min_train_seasons

    for i in range(start_idx, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]

        tr_mask = train_df['Season'].isin(train_seasons)
        va_mask = train_df['Season'] == val_season

        tr_df = train_df[tr_mask]
        va_df = train_df[va_mask]

        if len(va_df) < 5:
            continue

        X_tr = tr_df[feat_cols].values
        y_tr = tr_df['Label'].values
        X_va = va_df[feat_cols].values
        y_va = va_df['Label'].values

        # Fill NaN with training median
        med = np.nanmedian(X_tr, axis=0)
        med = np.where(np.isnan(med), 0, med)
        X_tr = np.where(np.isnan(X_tr), med, X_tr)
        X_va = np.where(np.isnan(X_va), med, X_va)

        preds_by_model = {}

        # LightGBM
        lgb_preds_seeds = []
        for seed in SEEDS:
            m = lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})
            m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            lgb_preds_seeds.append(m.predict_proba(X_va)[:, 1])
        preds_by_model['lgb'] = np.mean(lgb_preds_seeds, axis=0)

        # XGBoost
        xgb_preds_seeds = []
        for seed in SEEDS:
            m = xgb.XGBClassifier(**{**xgb_params, 'random_state': seed,
                                     'early_stopping_rounds': 50})
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            xgb_preds_seeds.append(m.predict_proba(X_va)[:, 1])
        preds_by_model['xgb'] = np.mean(xgb_preds_seeds, axis=0)

        # CatBoost
        cat_preds_seeds = []
        for seed in SEEDS:
            m = CatBoostClassifier(**{**cat_params, 'random_seed': seed})
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
            cat_preds_seeds.append(m.predict_proba(X_va)[:, 1])
        preds_by_model['cat'] = np.mean(cat_preds_seeds, axis=0)

        # Logistic Regression
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        lr.fit(X_tr_sc, y_tr)
        preds_by_model['logreg'] = lr.predict_proba(X_va_sc)[:, 1]

        # Ensemble: rank average + probability average, then blend
        rank_avg = np.zeros(len(y_va))
        prob_avg = np.zeros(len(y_va))
        for name, pred in preds_by_model.items():
            rank_avg += rankdata(pred)
            prob_avg += pred

        rank_avg /= len(preds_by_model)
        prob_avg /= len(preds_by_model)

        # Normalize rank avg to [0,1]
        rank_avg_norm = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min() + 1e-10)

        # Blend: 50% rank avg + 50% prob avg
        ensemble_pred = 0.5 * rank_avg_norm + 0.5 * prob_avg

        # Compute losses
        for name, pred in preds_by_model.items():
            pred_clipped = np.clip(pred, 0.025, 0.975)
            ll = log_loss(y_va, pred_clipped)
            bs = brier_score_loss(y_va, pred_clipped)
            results[name].append({
                'season': val_season, 'logloss': ll, 'brier': bs, 'n': len(y_va)
            })

        ens_clipped = np.clip(ensemble_pred, 0.025, 0.975)
        ens_ll = log_loss(y_va, ens_clipped)
        ens_bs = brier_score_loss(y_va, ens_clipped)
        results['ensemble'].append({
            'season': val_season, 'logloss': ens_ll, 'brier': ens_bs, 'n': len(y_va)
        })

        # Store OOF predictions
        for idx_va, row_idx in enumerate(va_df.index):
            all_oof_preds.append({
                'idx': row_idx, 'season': val_season, 'label': y_va[idx_va],
                'lgb': preds_by_model['lgb'][idx_va],
                'xgb': preds_by_model['xgb'][idx_va],
                'cat': preds_by_model['cat'][idx_va],
                'logreg': preds_by_model['logreg'][idx_va],
                'ensemble': ensemble_pred[idx_va],
            })

    return results, pd.DataFrame(all_oof_preds)


print("Running expanding window CV on combined Men + Women data...")
print("This takes a while — training 4 models x 5 seeds per validation year.")

cv_results, oof_df = expanding_window_cv(train_all, FEAT_COLS, min_train_seasons=5)

# Summary
print("\\n" + "="*70)
print("EXPANDING WINDOW CV RESULTS")
print("="*70)
for model_name, season_results in cv_results.items():
    if len(season_results) == 0:
        continue
    lls = [r['logloss'] for r in season_results]
    bss = [r['brier'] for r in season_results]
    ns = [r['n'] for r in season_results]
    # Weighted average by number of games
    total_n = sum(ns)
    wt_ll = sum(ll * n for ll, n in zip(lls, ns)) / total_n
    wt_bs = sum(bs * n for bs, n in zip(bss, ns)) / total_n
    print(f"  {model_name:10s}: LogLoss={wt_ll:.5f}  Brier={wt_bs:.5f}  (n={total_n})")

    wandb.log({
        f'cv_{model_name}_logloss': wt_ll,
        f'cv_{model_name}_brier': wt_bs,
    })""")

# =============================================================================
# Cell: Per-season CV breakdown
# =============================================================================
add_code("""print("Per-season breakdown:")
print(f"{'Season':>8s}", end="")
for model_name in ['lgb', 'xgb', 'cat', 'logreg', 'ensemble']:
    print(f"  {model_name:>10s}", end="")
print(f"  {'n':>5s}")

seasons_covered = sorted(set(r['season'] for r in cv_results['ensemble']))
for season in seasons_covered:
    print(f"{season:8d}", end="")
    for model_name in ['lgb', 'xgb', 'cat', 'logreg', 'ensemble']:
        season_data = [r for r in cv_results[model_name] if r['season'] == season]
        if season_data:
            print(f"  {season_data[0]['logloss']:10.5f}", end="")
        else:
            print(f"  {'N/A':>10s}", end="")
    n_data = [r for r in cv_results['ensemble'] if r['season'] == season]
    if n_data:
        print(f"  {n_data[0]['n']:5d}")
    else:
        print()""")

# =============================================================================
# Cell: Feature importance
# =============================================================================
add_code("""def get_feature_importance(train_df, feat_cols):
    \"\"\"Train a single LightGBM model and extract feature importances.\"\"\"
    X = train_df[feat_cols].values
    y = train_df['Label'].values

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0, med)
    X = np.where(np.isnan(X), med, X)

    m = lgb.LGBMClassifier(**{**lgb_params, 'random_state': 42, 'n_estimators': 500})
    m.fit(X, y)

    imp = pd.DataFrame({
        'feature': feat_cols,
        'importance': m.feature_importances_,
    }).sort_values('importance', ascending=False)

    return imp


feat_imp = get_feature_importance(train_all, FEAT_COLS)

print("Top 30 features:")
print(feat_imp.head(30).to_string(index=False))

# Log to W&B
wandb.log({
    'feature_importance': wandb.Table(
        dataframe=feat_imp.head(50),
        columns=['feature', 'importance']
    )
})""")

# =============================================================================
# Cell: Train final models on all data
# =============================================================================
add_code("""def train_final_models(train_df, feat_cols):
    \"\"\"Train all models on full training data, return models and median for NaN fill.\"\"\"
    X = train_df[feat_cols].values
    y = train_df['Label'].values

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0, med)
    X = np.where(np.isnan(X), med, X)

    models = {'lgb': [], 'xgb': [], 'cat': [], 'logreg': []}

    for seed in SEEDS:
        # LightGBM
        m_lgb = lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})
        m_lgb.fit(X, y)
        models['lgb'].append(m_lgb)

        # XGBoost
        m_xgb = xgb.XGBClassifier(**{**xgb_params, 'random_state': seed})
        m_xgb.fit(X, y)
        models['xgb'].append(m_xgb)

        # CatBoost
        m_cat = CatBoostClassifier(**{**cat_params, 'random_seed': seed})
        m_cat.fit(X, y)
        models['cat'].append(m_cat)

    # Logistic Regression (single, no seed variation needed)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    lr.fit(X_sc, y)
    models['logreg'] = (lr, scaler)

    return models, med


print("Training final models on all historical data...")
final_models, final_med = train_final_models(train_all, FEAT_COLS)
print(f"Trained: {len(SEEDS)} seeds x 3 tree models + 1 LogReg")""")

# =============================================================================
# Cell: Prediction function
# =============================================================================
add_code("""def predict_submission(sub_df, m_feats, w_feats, models, med, feat_cols):
    \"\"\"Generate predictions for a submission DataFrame.\"\"\"
    sub = sub_df.copy()
    sub[['Season', 'T1', 'T2']] = sub['ID'].str.split('_', expand=True).astype(int)

    # Determine Men vs Women by TeamID range
    sub['IsWomen'] = (sub['T1'] >= 3000).astype(int)

    # Build per-team lookup
    m_idx = m_feats.set_index(['Season', 'TeamID'])
    w_idx = w_feats.set_index(['Season', 'TeamID'])

    rows = []
    for _, r in sub.iterrows():
        s, t1, t2 = r['Season'], r['T1'], r['T2']
        idx = w_idx if r['IsWomen'] else m_idx

        row = {}
        for col in TEAM_FEAT_COLS:
            for prefix, tid in [('T1', t1), ('T2', t2)]:
                try:
                    val = idx.loc[(s, tid), col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                except (KeyError, IndexError):
                    val = np.nan
                row[f'{prefix}_{col}'] = val

        # Compute diff features
        for diff_name, (col1, col2) in DIFF_MAP.items():
            v1 = row.get(f'T1_{col1}', np.nan)
            v2 = row.get(f'T2_{col2}', np.nan)
            if pd.notna(v1) and pd.notna(v2):
                row[diff_name] = v1 - v2
            else:
                row[diff_name] = np.nan

        rows.append(row)

    X_df = pd.DataFrame(rows)

    # Ensure all feat_cols exist
    for col in feat_cols:
        if col not in X_df.columns:
            X_df[col] = np.nan

    X = X_df[feat_cols].values
    X = np.where(np.isnan(X), med, X)

    # Predict with all models
    preds_by_model = {}

    for name in ['lgb', 'xgb', 'cat']:
        seed_preds = []
        for m in models[name]:
            seed_preds.append(m.predict_proba(X)[:, 1])
        preds_by_model[name] = np.mean(seed_preds, axis=0)

    lr, scaler = models['logreg']
    X_sc = scaler.transform(X)
    preds_by_model['logreg'] = lr.predict_proba(X_sc)[:, 1]

    # Ensemble: rank average + probability average blend
    rank_avg = np.zeros(len(X))
    prob_avg = np.zeros(len(X))
    for name, pred in preds_by_model.items():
        rank_avg += rankdata(pred)
        prob_avg += pred

    rank_avg /= len(preds_by_model)
    prob_avg /= len(preds_by_model)

    rank_avg_norm = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min() + 1e-10)
    ensemble_pred = 0.5 * rank_avg_norm + 0.5 * prob_avg

    # Clip
    ensemble_pred = np.clip(ensemble_pred, 0.025, 0.975)

    sub['Pred'] = ensemble_pred

    # Also store individual model predictions for analysis
    for name, pred in preds_by_model.items():
        sub[f'Pred_{name}'] = np.clip(pred, 0.025, 0.975)

    return sub


print("Prediction function ready.")""")

# =============================================================================
# Cell: Generate Stage 1 predictions (historical validation)
# =============================================================================
add_code("""print("Generating Stage 1 predictions (historical validation)...")
s1_result = predict_submission(sub_s1, m_feats, w_feats, final_models, final_med, FEAT_COLS)

print(f"Stage 1 predictions: {len(s1_result):,} rows")
print(f"Pred range: {s1_result['Pred'].min():.4f} - {s1_result['Pred'].max():.4f}")
print(f"Pred mean: {s1_result['Pred'].mean():.4f}")

# Breakdown by gender
men_mask = s1_result['T1'] < 3000
women_mask = s1_result['T1'] >= 3000
print(f"\\nMen predictions: {men_mask.sum():,}, mean: {s1_result.loc[men_mask, 'Pred'].mean():.4f}")
print(f"Women predictions: {women_mask.sum():,}, mean: {s1_result.loc[women_mask, 'Pred'].mean():.4f}")

# Per-model stats
for name in ['lgb', 'xgb', 'cat', 'logreg']:
    col = f'Pred_{name}'
    if col in s1_result.columns:
        print(f"  {name}: mean={s1_result[col].mean():.4f}, std={s1_result[col].std():.4f}")""")

# =============================================================================
# Cell: Generate Stage 2 predictions (2026 tournament)
# =============================================================================
add_code("""if sub_s2 is not None:
    print("Generating Stage 2 predictions (2026 tournament)...")
    s2_result = predict_submission(sub_s2, m_feats, w_feats, final_models, final_med, FEAT_COLS)

    print(f"Stage 2 predictions: {len(s2_result):,} rows")
    print(f"Pred range: {s2_result['Pred'].min():.4f} - {s2_result['Pred'].max():.4f}")

    # Use Stage 2 for submission
    submission = s2_result[['ID', 'Pred']].sort_values('ID').reset_index(drop=True)
else:
    print("Stage 2 not available — using Stage 1 for submission.")
    submission = s1_result[['ID', 'Pred']].sort_values('ID').reset_index(drop=True)

print(f"\\nSubmission shape: {submission.shape}")
print(submission.head(10))""")

# =============================================================================
# Cell: Save submission
# =============================================================================
add_code("""submission.to_csv('submission.csv', index=False)
print(f"submission.csv saved: {len(submission):,} rows")

# Prediction distribution stats
stats = {
    'submission_rows': len(submission),
    'pred_mean': float(submission['Pred'].mean()),
    'pred_std': float(submission['Pred'].std()),
    'pred_min': float(submission['Pred'].min()),
    'pred_max': float(submission['Pred'].max()),
    'pred_median': float(submission['Pred'].median()),
    'pred_q25': float(submission['Pred'].quantile(0.25)),
    'pred_q75': float(submission['Pred'].quantile(0.75)),
}

for k, v in stats.items():
    print(f"  {k}: {v}")

wandb.log(stats)""")

# =============================================================================
# Cell: Prediction distribution visualization
# =============================================================================
add_code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall distribution
axes[0].hist(submission['Pred'].values, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Predicted Probability', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].set_title('Prediction Distribution (All)', fontsize=16)
axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.5)

# Men vs Women (from Stage 1 result for richer analysis)
if 's1_result' in dir():
    men_preds = s1_result.loc[s1_result['T1'] < 3000, 'Pred'].values
    women_preds = s1_result.loc[s1_result['T1'] >= 3000, 'Pred'].values
    axes[1].hist(men_preds, bins=50, alpha=0.6, label=f'Men (n={len(men_preds)})', edgecolor='black')
    axes[1].hist(women_preds, bins=50, alpha=0.6, label=f'Women (n={len(women_preds)})', edgecolor='black')
    axes[1].set_xlabel('Predicted Probability', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=14)
    axes[1].set_title('Men vs Women Distribution', fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('pred_distribution.png', dpi=100, bbox_inches='tight')
plt.show()
print("Distribution plot saved.")""")

# =============================================================================
# Cell: Feature importance visualization
# =============================================================================
add_code("""fig, ax = plt.subplots(figsize=(10, 12))

top_n = 30
top_feats = feat_imp.head(top_n)
ax.barh(range(top_n), top_feats['importance'].values, color='steelblue', edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats['feature'].values, fontsize=12)
ax.set_xlabel('Feature Importance', fontsize=14)
ax.set_title(f'Top {top_n} Feature Importances (LightGBM)', fontsize=16)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()
print("Feature importance plot saved.")""")

# =============================================================================
# Cell: W&B finish
# =============================================================================
add_code("""wandb.finish()
print("W&B offline run saved.")
print("\\nNotebook complete. submission.csv ready for submission.")""")


# =============================================================================
# Build the notebook file
# =============================================================================
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

out_path = "march-mania-2026-v2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path} ({len(cells)} cells)")

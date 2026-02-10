!pip install duckdb -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ====== Settings ======
PITCHER_ID = 506433  # Yu Darvish MLBAM ID
YEARS = [2021, 2022, 2023, 2024, 2025]
GAME_TYPE = 'R'  # Regular season only
# ======================

# Load from Kaggle dataset (much faster than API calls)
df_all = pd.read_csv('/kaggle/input/japan-mlb-pitchers-batters-statcast/japanese_mlb_pitching.csv')
df_raw = df_all[df_all['pitcher'] == PITCHER_ID].copy()
df_raw['season'] = df_raw['game_year']

for year in YEARS:
    n = len(df_raw[df_raw['season'] == year])
    print(f'  {year}: {n:,} pitches')

print(f'\nTotal (raw): {len(df_raw):,} pitches')

# Filter regular season only
con = duckdb.connect()
df = con.execute(f"""
    SELECT * FROM df_raw WHERE game_type = '{GAME_TYPE}'
""").df()
print(f'Total (regular season): {len(df):,} pitches')

# === Text Summary (for Claude Code review) ===
summary = con.execute("""
    SELECT
        season,
        COUNT(*) as pitches,
        COUNT(DISTINCT game_date) as games,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(MAX(release_speed), 1) as max_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin,
        COUNT(DISTINCT pitch_type) as pitch_types
    FROM df
    GROUP BY season
    ORDER BY season
""").df()

print('=== Season-by-Season Overview ===')
print(summary.to_string(index=False))
print(f'\nTotal: {len(df):,} pitches across {len(YEARS)} seasons')

# Which pitch types were used each year?
arsenal = con.execute("""
    SELECT
        season,
        pitch_type,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY season), 1) as pct,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY season, pitch_type
    ORDER BY season, count DESC
""").df()

print('=== Pitch Arsenal by Season ===')
for year in YEARS:
    year_data = arsenal[arsenal['season'] == year]
    print(f'\n--- {year} ---')
    print(year_data[['pitch_type', 'count', 'pct', 'avg_velo', 'avg_spin']].to_string(index=False))

# Pivot for stacked bar chart
mix_pivot = arsenal.pivot_table(index='season', columns='pitch_type', values='pct', fill_value=0)

# Chart
mix_pivot.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Set3')
plt.title('Yu Darvish - Pitch Mix Evolution (2021-2025)')
plt.xlabel('Season')
plt.ylabel('Usage %')
plt.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Pitch Mix Changes (% usage) ===')
print(mix_pivot.round(1).to_string())

# Year-over-year biggest changes
if len(YEARS) >= 2:
    first = mix_pivot.loc[YEARS[0]]
    last = mix_pivot.loc[YEARS[-1]]
    diff = (last - first).sort_values()
    print(f'\n=== Biggest Changes ({YEARS[0]} → {YEARS[-1]}) ===')
    for pitch, change in diff.items():
        if abs(change) >= 1.0:
            direction = '↑' if change > 0 else '↓'
            print(f'  {pitch}: {first[pitch]:.1f}% → {last[pitch]:.1f}% ({direction}{abs(change):.1f}%)')

# Fastball (FF) velocity trend across years
velo_by_year = con.execute("""
    SELECT
        season,
        pitch_type,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin,
        COUNT(*) as count
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY season, pitch_type
    ORDER BY season
""").df()

# Get top 4 most used pitches overall
top_pitches = con.execute("""
    SELECT pitch_type FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY pitch_type
    ORDER BY COUNT(*) DESC
    LIMIT 4
""").df()['pitch_type'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for pitch in top_pitches:
    data = velo_by_year[velo_by_year['pitch_type'] == pitch]
    axes[0].plot(data['season'], data['avg_velo'], marker='o', label=pitch, linewidth=2)
    axes[1].plot(data['season'], data['avg_spin'], marker='o', label=pitch, linewidth=2)

axes[0].set_title('Average Velocity by Season')
axes[0].set_xlabel('Season')
axes[0].set_ylabel('Velocity (mph)')
axes[0].legend()

axes[1].set_title('Average Spin Rate by Season')
axes[1].set_xlabel('Season')
axes[1].set_ylabel('Spin Rate (rpm)')
axes[1].legend()

plt.suptitle('Yu Darvish - Velocity & Spin Trends (2021-2025)')
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Velocity & Spin by Year (Top 4 Pitches) ===')
for pitch in top_pitches:
    data = velo_by_year[velo_by_year['pitch_type'] == pitch]
    print(f'\n{pitch}:')
    print(data[['season', 'avg_velo', 'avg_spin', 'count']].to_string(index=False))

# Fastball velocity by inning, per season
ff_type = 'FF' if 'FF' in top_pitches else top_pitches[0]

fatigue = con.execute(f"""
    SELECT
        season,
        inning,
        ROUND(AVG(release_speed), 1) as avg_velo,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type = '{ff_type}' AND inning <= 8
    GROUP BY season, inning
    HAVING COUNT(*) >= 5
    ORDER BY season, inning
""").df()

fig, ax = plt.subplots(figsize=(12, 6))

for year in YEARS:
    data = fatigue[fatigue['season'] == year]
    if len(data) > 0:
        ax.plot(data['inning'], data['avg_velo'], marker='o', label=str(year), linewidth=2)

ax.set_xlabel('Inning')
ax.set_ylabel(f'{ff_type} Velocity (mph)')
ax.set_title(f'Yu Darvish - {ff_type} Velocity by Inning (2021-2025)')
ax.set_xticks(range(1, 9))
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print(f'\n=== {ff_type} Velocity by Inning ===')
fatigue_pivot = fatigue.pivot_table(index='inning', columns='season', values='avg_velo')
print(fatigue_pivot.to_string())

# Velocity drop (1st inning vs last inning with data)
print(f'\n=== Velocity Drop (1st inning → last inning) ===')
for year in YEARS:
    data = fatigue[fatigue['season'] == year]
    if len(data) >= 2:
        first_velo = data.iloc[0]['avg_velo']
        last_velo = data.iloc[-1]['avg_velo']
        last_inn = int(data.iloc[-1]['inning'])
        drop = last_velo - first_velo
        print(f'  {year}: {first_velo} → {last_velo} (inn {last_inn}) = {drop:+.1f} mph')

# Velocity by pitch count within game, per season
pitch_count_effect = con.execute(f"""
    WITH pitch_seq AS (
        SELECT
            season,
            game_pk,
            release_speed,
            ROW_NUMBER() OVER(PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) as pitch_num
        FROM df
        WHERE pitch_type = '{ff_type}'
    )
    SELECT
        season,
        CASE
            WHEN pitch_num <= 25 THEN '1-25'
            WHEN pitch_num <= 50 THEN '26-50'
            WHEN pitch_num <= 75 THEN '51-75'
            WHEN pitch_num <= 100 THEN '76-100'
            ELSE '100+'
        END as pitch_range,
        COUNT(*) as pitches,
        ROUND(AVG(release_speed), 1) as avg_velo
    FROM pitch_seq
    GROUP BY season, pitch_range
    ORDER BY season, pitch_range
""").df()

print('=== Velocity by Pitch Count in Game ===')
for year in YEARS:
    data = pitch_count_effect[pitch_count_effect['season'] == year]
    print(f'\n--- {year} ---')
    print(data[['pitch_range', 'pitches', 'avg_velo']].to_string(index=False))

# Whiff rate by pitch type by season (FIXED: includes hit_into_play in denominator)
whiff = con.execute("""
    SELECT
        season,
        pitch_type,
        COUNT(*) as total_pitches,
        SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) as whiffs,
        SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END) as total_swings,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 1) as whiff_rate
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY season, pitch_type
    ORDER BY season, total_pitches DESC
""").df()

# Chart: whiff rate for top pitches across years
fig, ax = plt.subplots(figsize=(12, 6))

for pitch in top_pitches:
    data = whiff[whiff['pitch_type'] == pitch]
    if len(data) > 0:
        ax.plot(data['season'], data['whiff_rate'], marker='o', label=pitch, linewidth=2)

ax.set_xlabel('Season')
ax.set_ylabel('Whiff Rate (%)')
ax.set_title('Yu Darvish - Whiff Rate Evolution by Pitch Type (2021-2025)')
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Whiff Rate by Pitch Type by Season ===')
whiff_pivot = whiff.pivot_table(index='pitch_type', columns='season', values='whiff_rate')
print(whiff_pivot.round(1).to_string())

# Changes
print(f'\n=== Whiff Rate Changes ({YEARS[0]} → {YEARS[-1]}) ===')
for pitch in top_pitches:
    data = whiff[whiff['pitch_type'] == pitch]
    if len(data) >= 2:
        first_val = data[data['season'] == YEARS[0]]['whiff_rate'].values
        last_val = data[data['season'] == YEARS[-1]]['whiff_rate'].values
        if len(first_val) > 0 and len(last_val) > 0:
            change = last_val[0] - first_val[0]
            print(f'  {pitch}: {first_val[0]:.1f}% → {last_val[0]:.1f}% ({change:+.1f}%)')

# Two-strike pitch selection by season
two_strike = con.execute("""
    SELECT
        season,
        pitch_type,
        COUNT(*) as pitches,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY season), 1) as pct,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 1) as whiff_rate
    FROM df
    WHERE strikes = 2 AND pitch_type IS NOT NULL
    GROUP BY season, pitch_type
    ORDER BY season, pitches DESC
""").df()

print('=== Two-Strike Pitch Selection by Season ===')
for year in YEARS:
    data = two_strike[two_strike['season'] == year].head(5)
    print(f'\n--- {year} ---')
    print(data[['pitch_type', 'pitches', 'pct', 'whiff_rate']].to_string(index=False))

# Two-strike pitch mix change chart
ts_pivot = two_strike.pivot_table(index='season', columns='pitch_type', values='pct', fill_value=0)

ts_pivot.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Set2')
plt.title('Yu Darvish - Two-Strike Pitch Mix (2021-2025)')
plt.xlabel('Season')
plt.ylabel('Usage %')
plt.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Two-Strike Mix Changes ===')
print(ts_pivot.round(1).to_string())

# Count-based pitch selection (FIXED: Full Count checked before Behind)
count_analysis = con.execute("""
    SELECT
        season,
        CASE
            WHEN balls = 3 AND strikes = 2 THEN 'Full Count'
            WHEN balls > strikes THEN 'Behind'
            WHEN strikes > balls THEN 'Ahead'
            ELSE 'Even'
        END as count_situation,
        pitch_type,
        COUNT(*) as pitches,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(
            PARTITION BY season,
            CASE
                WHEN balls = 3 AND strikes = 2 THEN 'Full Count'
                WHEN balls > strikes THEN 'Behind'
                WHEN strikes > balls THEN 'Ahead'
                ELSE 'Even'
            END
        ), 1) as pct
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY season, count_situation, pitch_type
    ORDER BY season, count_situation, pitches DESC
""").df()

# Show top 3 pitches per situation per year
print('=== Pitch Selection by Count Situation ===')
for year in YEARS:
    print(f'\n=== {year} ===')
    for situation in ['Ahead', 'Even', 'Behind', 'Full Count']:
        data = count_analysis[
            (count_analysis['season'] == year) &
            (count_analysis['count_situation'] == situation)
        ].head(3)
        if len(data) > 0:
            top_str = ', '.join([f"{r['pitch_type']} {r['pct']}%" for _, r in data.iterrows()])
            print(f'  {situation}: {top_str}')

print('=' * 60)
print('YU DARVISH 2021-2025 EVOLUTION SUMMARY')
print('=' * 60)

# Games & Pitches
print('\n[Workload]')
for _, row in summary.iterrows():
    print(f'  {int(row["season"])}: {int(row["games"])} games, {int(row["pitches"]):,} pitches, avg {row["avg_velo"]} mph')

# Pitch mix biggest changes
print(f'\n[Pitch Mix Changes ({YEARS[0]} → {YEARS[-1]})]')
if len(YEARS) >= 2:
    first = mix_pivot.loc[YEARS[0]]
    last = mix_pivot.loc[YEARS[-1]]
    diff = (last - first).sort_values()
    for pitch, change in diff.items():
        if abs(change) >= 2.0:
            direction = 'increased' if change > 0 else 'decreased'
            print(f'  {pitch}: {direction} by {abs(change):.1f}% ({first[pitch]:.1f}% → {last[pitch]:.1f}%)')

# Velocity trend
print(f'\n[Fastball Velocity Trend]')
ff_yearly = velo_by_year[velo_by_year['pitch_type'] == ff_type]
for _, row in ff_yearly.iterrows():
    print(f'  {int(row["season"])}: {row["avg_velo"]} mph ({int(row["count"])} pitches)')

# Best whiff pitch per year
print(f'\n[Best Whiff Rate Pitch per Year]')
for year in YEARS:
    year_whiff = whiff[(whiff['season'] == year) & (whiff['total_swings'] >= 30)]
    if len(year_whiff) > 0:
        best = year_whiff.loc[year_whiff['whiff_rate'].idxmax()]
        print(f'  {year}: {best["pitch_type"]} ({best["whiff_rate"]}%)')

print('\n' + '=' * 60)

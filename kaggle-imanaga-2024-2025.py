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
PITCHER_ID = 684007  # Shota Imanaga MLBAM ID
YEARS = [2024, 2025]
GAME_TYPE = 'R'  # Regular season only
ASB_DATE = '2025-07-15'  # All-Star Break cutoff for 1H/2H split
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
    SELECT *,
        CASE
            WHEN season = 2024 THEN '2024'
            WHEN season = 2025 AND game_date < '{ASB_DATE}' THEN '2025-1H'
            ELSE '2025-2H'
        END as period
    FROM df_raw
    WHERE game_type = '{GAME_TYPE}'
""").df()

print(f'Total (regular season): {len(df):,} pitches')
print(f'\nPeriod breakdown:')
for period in ['2024', '2025-1H', '2025-2H']:
    n = len(df[df['period'] == period])
    print(f'  {period}: {n:,} pitches')

PERIODS = ['2024', '2025-1H', '2025-2H']

summary = con.execute("""
    SELECT
        period,
        COUNT(*) as pitches,
        COUNT(DISTINCT game_date) as games,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(MAX(release_speed), 1) as max_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin,
        COUNT(DISTINCT pitch_type) as pitch_types
    FROM df
    GROUP BY period
    ORDER BY period
""").df()

print('=== Period Overview ===')
print(summary.to_string(index=False))
print(f'\nTotal: {len(df):,} pitches')

arsenal = con.execute("""
    SELECT
        period,
        pitch_type,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY period), 1) as pct,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY period, pitch_type
    ORDER BY period, count DESC
""").df()

print('=== Pitch Arsenal by Period ===')
for period in PERIODS:
    data = arsenal[arsenal['period'] == period]
    print(f'\n--- {period} ---')
    print(data[['pitch_type', 'count', 'pct', 'avg_velo', 'avg_spin']].to_string(index=False))

mix_pivot = arsenal.pivot_table(index='period', columns='pitch_type', values='pct', fill_value=0)
mix_pivot = mix_pivot.reindex(PERIODS)

mix_pivot.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Set3')
plt.title('Shota Imanaga - Pitch Mix by Period')
plt.xlabel('Period')
plt.ylabel('Usage %')
plt.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Pitch Mix (% usage) ===')
print(mix_pivot.round(1).to_string())

# 2024 vs 2025-2H changes
if '2024' in mix_pivot.index and '2025-2H' in mix_pivot.index:
    first = mix_pivot.loc['2024']
    last = mix_pivot.loc['2025-2H']
    diff = (last - first).sort_values()
    print(f'\n=== Biggest Changes (2024 → 2025-2H) ===')
    for pitch, change in diff.items():
        if abs(change) >= 1.0:
            direction = '↑' if change > 0 else '↓'
            print(f'  {pitch}: {first[pitch]:.1f}% → {last[pitch]:.1f}% ({direction}{abs(change):.1f}%)')

velo_by_period = con.execute("""
    SELECT
        period,
        pitch_type,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin,
        COUNT(*) as count
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY period, pitch_type
    ORDER BY period
""").df()

top_pitches = con.execute("""
    SELECT pitch_type FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY pitch_type
    ORDER BY COUNT(*) DESC
    LIMIT 4
""").df()['pitch_type'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

period_order = {p: i for i, p in enumerate(PERIODS)}
for pitch in top_pitches:
    data = velo_by_period[velo_by_period['pitch_type'] == pitch].copy()
    data['period_idx'] = data['period'].map(period_order)
    data = data.sort_values('period_idx')
    axes[0].plot(data['period'], data['avg_velo'], marker='o', label=pitch, linewidth=2)
    axes[1].plot(data['period'], data['avg_spin'], marker='o', label=pitch, linewidth=2)

axes[0].set_title('Average Velocity by Period')
axes[0].set_ylabel('Velocity (mph)')
axes[0].legend()

axes[1].set_title('Average Spin Rate by Period')
axes[1].set_ylabel('Spin Rate (rpm)')
axes[1].legend()

plt.suptitle('Shota Imanaga - Velocity & Spin Trends')
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Velocity & Spin by Period (Top Pitches) ===')
for pitch in top_pitches:
    data = velo_by_period[velo_by_period['pitch_type'] == pitch]
    print(f'\n{pitch}:')
    print(data[['period', 'avg_velo', 'avg_spin', 'count']].to_string(index=False))

monthly_velo = con.execute("""
    SELECT
        season,
        EXTRACT(MONTH FROM game_date::DATE) as month,
        pitch_type,
        ROUND(AVG(release_speed), 1) as avg_velo,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type IN (SELECT pitch_type FROM df GROUP BY pitch_type ORDER BY COUNT(*) DESC LIMIT 3)
    GROUP BY season, month, pitch_type
    HAVING COUNT(*) >= 10
    ORDER BY season, month
""").df()

# Plot 2025 monthly trend
fig, ax = plt.subplots(figsize=(12, 6))
df_2025_monthly = monthly_velo[monthly_velo['season'] == 2025]
for pitch in df_2025_monthly['pitch_type'].unique():
    data = df_2025_monthly[df_2025_monthly['pitch_type'] == pitch]
    ax.plot(data['month'], data['avg_velo'], marker='o', label=pitch, linewidth=2)

ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5, label='ASB')
ax.set_xlabel('Month')
ax.set_ylabel('Velocity (mph)')
ax.set_title('Shota Imanaga - 2025 Monthly Velocity Trend')
ax.set_xticks(range(3, 11))
ax.set_xticklabels(['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print('=== 2025 Monthly Velocity ===')
for pitch in df_2025_monthly['pitch_type'].unique():
    data = df_2025_monthly[df_2025_monthly['pitch_type'] == pitch]
    print(f'\n{pitch}:')
    print(data[['month', 'avg_velo', 'pitches']].to_string(index=False))

# Also show 2024 for comparison
print('\n=== 2024 Monthly Velocity ===')
df_2024_monthly = monthly_velo[monthly_velo['season'] == 2024]
for pitch in df_2024_monthly['pitch_type'].unique():
    data = df_2024_monthly[df_2024_monthly['pitch_type'] == pitch]
    print(f'\n{pitch}:')
    print(data[['month', 'avg_velo', 'pitches']].to_string(index=False))

ff_type = 'FF' if 'FF' in top_pitches else top_pitches[0]

fatigue = con.execute(f"""
    SELECT
        period,
        inning,
        ROUND(AVG(release_speed), 1) as avg_velo,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type = '{ff_type}' AND inning <= 8
    GROUP BY period, inning
    HAVING COUNT(*) >= 5
    ORDER BY period, inning
""").df()

fig, ax = plt.subplots(figsize=(12, 6))
for period in PERIODS:
    data = fatigue[fatigue['period'] == period]
    if len(data) > 0:
        ax.plot(data['inning'], data['avg_velo'], marker='o', label=period, linewidth=2)

ax.set_xlabel('Inning')
ax.set_ylabel(f'{ff_type} Velocity (mph)')
ax.set_title(f'Shota Imanaga - {ff_type} Velocity by Inning')
ax.set_xticks(range(1, 9))
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print(f'\n=== {ff_type} Velocity by Inning ===')
fatigue_pivot = fatigue.pivot_table(index='inning', columns='period', values='avg_velo')
if len(fatigue_pivot.columns) > 0:
    fatigue_pivot = fatigue_pivot.reindex(columns=PERIODS)
print(fatigue_pivot.to_string())

print(f'\n=== Velocity Drop (1st inning → last inning) ===')
for period in PERIODS:
    data = fatigue[fatigue['period'] == period]
    if len(data) >= 2:
        first_velo = data.iloc[0]['avg_velo']
        last_velo = data.iloc[-1]['avg_velo']
        last_inn = int(data.iloc[-1]['inning'])
        drop = last_velo - first_velo
        print(f'  {period}: {first_velo} → {last_velo} (inn {last_inn}) = {drop:+.1f} mph')

whiff = con.execute("""
    SELECT
        period,
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
    GROUP BY period, pitch_type
    ORDER BY period, total_pitches DESC
""").df()

fig, ax = plt.subplots(figsize=(12, 6))
period_order = {p: i for i, p in enumerate(PERIODS)}
for pitch in top_pitches:
    data = whiff[whiff['pitch_type'] == pitch].copy()
    data['period_idx'] = data['period'].map(period_order)
    data = data.sort_values('period_idx')
    if len(data) > 0:
        ax.plot(data['period'], data['whiff_rate'], marker='o', label=pitch, linewidth=2)

ax.set_ylabel('Whiff Rate (%)')
ax.set_title('Shota Imanaga - Whiff Rate by Pitch Type')
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Whiff Rate by Pitch Type ===')
whiff_pivot = whiff.pivot_table(index='pitch_type', columns='period', values='whiff_rate')
if len(whiff_pivot.columns) > 0:
    whiff_pivot = whiff_pivot.reindex(columns=PERIODS)
print(whiff_pivot.round(1).to_string())

two_strike = con.execute("""
    SELECT
        period,
        pitch_type,
        COUNT(*) as pitches,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY period), 1) as pct,
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
    GROUP BY period, pitch_type
    ORDER BY period, pitches DESC
""").df()

print('=== Two-Strike Pitch Selection ===')
for period in PERIODS:
    data = two_strike[two_strike['period'] == period].head(5)
    print(f'\n--- {period} ---')
    print(data[['pitch_type', 'pitches', 'pct', 'whiff_rate']].to_string(index=False))

count_analysis = con.execute("""
    SELECT
        period,
        CASE
            WHEN balls = 3 AND strikes = 2 THEN 'Full Count'
            WHEN balls > strikes THEN 'Behind'
            WHEN strikes > balls THEN 'Ahead'
            ELSE 'Even'
        END as count_situation,
        pitch_type,
        COUNT(*) as pitches,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(
            PARTITION BY period,
            CASE
                WHEN balls = 3 AND strikes = 2 THEN 'Full Count'
                WHEN balls > strikes THEN 'Behind'
                WHEN strikes > balls THEN 'Ahead'
                ELSE 'Even'
            END
        ), 1) as pct
    FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY period, count_situation, pitch_type
    ORDER BY period, count_situation, pitches DESC
""").df()

print('=== Pitch Selection by Count Situation ===')
for period in PERIODS:
    print(f'\n=== {period} ===')
    for situation in ['Ahead', 'Even', 'Behind', 'Full Count']:
        data = count_analysis[
            (count_analysis['period'] == period) &
            (count_analysis['count_situation'] == situation)
        ].head(3)
        if len(data) > 0:
            top_str = ', '.join([f"{r['pitch_type']} {r['pct']}%" for _, r in data.iterrows()])
            print(f'  {situation}: {top_str}')

batted = con.execute("""
    SELECT
        period,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(AVG(launch_angle), 1) as avg_launch_angle,
        ROUND(100.0 * SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) / COUNT(*), 1) as hard_hit_pct,
        ROUND(AVG(estimated_ba_using_speedangle), 3) as avg_xBA,
        ROUND(AVG(estimated_woba_using_speedangle), 3) as avg_xwOBA
    FROM df
    WHERE launch_speed IS NOT NULL
    GROUP BY period
    ORDER BY period
""").df()

print('=== Batted Ball Results by Period ===')
print(batted.to_string(index=False))

# By pitch type
batted_by_pitch = con.execute("""
    SELECT
        period,
        pitch_type,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(AVG(estimated_ba_using_speedangle), 3) as avg_xBA
    FROM df
    WHERE launch_speed IS NOT NULL AND pitch_type IS NOT NULL
    GROUP BY period, pitch_type
    HAVING COUNT(*) >= 10
    ORDER BY period, batted_balls DESC
""").df()

print('\n=== Batted Ball by Pitch Type (min 10 BIP) ===')
for period in PERIODS:
    data = batted_by_pitch[batted_by_pitch['period'] == period]
    print(f'\n--- {period} ---')
    print(data[['pitch_type', 'batted_balls', 'avg_exit_velo', 'avg_xBA']].to_string(index=False))

# Time Through Order analysis
# at_bat_number resets per game, we approximate TTO by grouping at_bat_number
tto = con.execute("""
    WITH batter_pa AS (
        SELECT
            period,
            game_pk,
            batter,
            at_bat_number,
            DENSE_RANK() OVER(PARTITION BY game_pk, batter ORDER BY at_bat_number) as pa_num
        FROM df
        GROUP BY period, game_pk, batter, at_bat_number
    ),
    tto_tagged AS (
        SELECT
            d.*,
            CASE
                WHEN b.pa_num = 1 THEN '1st'
                WHEN b.pa_num = 2 THEN '2nd'
                ELSE '3rd+'
            END as tto
        FROM df d
        JOIN batter_pa b ON d.game_pk = b.game_pk AND d.batter = b.batter AND d.at_bat_number = b.at_bat_number
    )
    SELECT
        period,
        tto,
        COUNT(*) as pitches,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 1) as whiff_rate
    FROM tto_tagged
    GROUP BY period, tto
    ORDER BY period, tto
""").df()

print('=== Whiff Rate by Time Through Order ===')
tto_pivot = tto.pivot_table(index='tto', columns='period', values='whiff_rate')
tto_pivot = tto_pivot.reindex(columns=PERIODS)
print(tto_pivot.round(1).to_string())

# Batted ball quality by TTO
tto_batted = con.execute("""
    WITH batter_pa AS (
        SELECT
            period,
            game_pk,
            batter,
            at_bat_number,
            DENSE_RANK() OVER(PARTITION BY game_pk, batter ORDER BY at_bat_number) as pa_num
        FROM df
        GROUP BY period, game_pk, batter, at_bat_number
    ),
    tto_tagged AS (
        SELECT
            d.*,
            CASE
                WHEN b.pa_num = 1 THEN '1st'
                WHEN b.pa_num = 2 THEN '2nd'
                ELSE '3rd+'
            END as tto
        FROM df d
        JOIN batter_pa b ON d.game_pk = b.game_pk AND d.batter = b.batter AND d.at_bat_number = b.at_bat_number
    )
    SELECT
        period,
        tto,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(AVG(estimated_woba_using_speedangle), 3) as avg_xwOBA
    FROM tto_tagged
    WHERE launch_speed IS NOT NULL
    GROUP BY period, tto
    ORDER BY period, tto
""").df()

print('\n=== Batted Ball by Time Through Order ===')
for period in PERIODS:
    data = tto_batted[tto_batted['period'] == period]
    print(f'\n--- {period} ---')
    print(data[['tto', 'batted_balls', 'avg_exit_velo', 'avg_xwOBA']].to_string(index=False))

# L/R splits - pitch usage and effectiveness
lr_arsenal = con.execute("""
    SELECT
        period,
        stand,
        pitch_type,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY period, stand), 1) as pct,
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
    GROUP BY period, stand, pitch_type
    HAVING COUNT(*) >= 10
    ORDER BY period, stand, count DESC
""").df()

print('=== Pitch Usage & Whiff Rate by Batter Side ===')
for period in PERIODS:
    print(f'\n=== {period} ===')
    for side in ['L', 'R']:
        data = lr_arsenal[(lr_arsenal['period'] == period) & (lr_arsenal['stand'] == side)]
        print(f'\n  vs {side}HB:')
        print(data[['pitch_type', 'count', 'pct', 'whiff_rate']].to_string(index=False))

# Batted ball by side
lr_batted = con.execute("""
    SELECT
        period,
        stand,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(AVG(estimated_woba_using_speedangle), 3) as avg_xwOBA,
        ROUND(100.0 * SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) / COUNT(*), 1) as hard_hit_pct
    FROM df
    WHERE launch_speed IS NOT NULL
    GROUP BY period, stand
    ORDER BY period, stand
""").df()

print('\n=== Batted Ball by Batter Side ===')
print(lr_batted.to_string(index=False))

# ST-specific L/R splits
lr_st = con.execute("""
    SELECT
        period,
        stand,
        COUNT(*) as pitches,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 1) as whiff_rate,
        ROUND(AVG(CASE WHEN launch_speed IS NOT NULL THEN estimated_ba_using_speedangle END), 3) as xBA_on_contact
    FROM df
    WHERE pitch_type = 'ST'
    GROUP BY period, stand
    ORDER BY period, stand
""").df()

print('\n=== ST (Sweeper) Left/Right Splits ===')
print(lr_st.to_string(index=False))

# Monthly batted ball metrics
monthly_batted = con.execute("""
    SELECT
        season,
        EXTRACT(MONTH FROM game_date::DATE) as month,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(100.0 * SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) / COUNT(*), 1) as hard_hit_pct,
        ROUND(AVG(estimated_woba_using_speedangle), 3) as avg_xwOBA
    FROM df
    WHERE launch_speed IS NOT NULL
    GROUP BY season, month
    HAVING COUNT(*) >= 20
    ORDER BY season, month
""").df()

# Plot xwOBA monthly
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for year in YEARS:
    data = monthly_batted[monthly_batted['season'] == year]
    axes[0].plot(data['month'], data['avg_xwOBA'], marker='o', label=str(year), linewidth=2)
    axes[1].plot(data['month'], data['hard_hit_pct'], marker='o', label=str(year), linewidth=2)

axes[0].axvline(x=7, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('xwOBA')
axes[0].set_title('Monthly xwOBA (lower = better)')
axes[0].legend()

axes[1].axvline(x=7, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Hard Hit %')
axes[1].set_title('Monthly Hard Hit % (lower = better)')
axes[1].legend()

plt.suptitle('Shota Imanaga - Monthly Batted Ball Quality')
plt.tight_layout()
plt.show()

# === Text Summary ===
print('=== Monthly Batted Ball Metrics ===')
for year in YEARS:
    data = monthly_batted[monthly_batted['season'] == year]
    print(f'\n--- {year} ---')
    print(data[['month', 'batted_balls', 'avg_exit_velo', 'hard_hit_pct', 'avg_xwOBA']].to_string(index=False))

# FS zone analysis
# zone 1-9 = strike zone, 11-14 = chase/waste zones
fs_zone = con.execute("""
    SELECT
        period,
        CASE
            WHEN zone BETWEEN 1 AND 9 THEN 'In Zone'
            WHEN zone BETWEEN 11 AND 14 THEN 'Chase'
            ELSE 'Waste'
        END as zone_type,
        COUNT(*) as pitches,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY period), 1) as pct,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 1) as whiff_rate,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END) / COUNT(*), 1) as swing_rate
    FROM df
    WHERE pitch_type = 'FS' AND zone IS NOT NULL
    GROUP BY period, zone_type
    ORDER BY period, zone_type
""").df()

print('=== FS Zone Analysis ===')
for period in PERIODS:
    data = fs_zone[fs_zone['period'] == period]
    print(f'\n--- {period} ---')
    print(data[['zone_type', 'pitches', 'pct', 'swing_rate', 'whiff_rate']].to_string(index=False))

# FS location scatter by period
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, period in enumerate(PERIODS):
    fs_data = con.execute(f"""
        SELECT plate_x, plate_z, description
        FROM df
        WHERE pitch_type = 'FS' AND period = '{period}'
          AND plate_x IS NOT NULL AND plate_z IS NOT NULL
    """).df()

    whiff_mask = fs_data['description'].isin(['swinging_strike', 'swinging_strike_blocked'])

    axes[i].scatter(fs_data[~whiff_mask]['plate_x'], fs_data[~whiff_mask]['plate_z'],
                    alpha=0.3, s=20, c='gray', label='Other')
    axes[i].scatter(fs_data[whiff_mask]['plate_x'], fs_data[whiff_mask]['plate_z'],
                    alpha=0.7, s=30, c='red', label='Whiff')

    # Strike zone box (approximate)
    axes[i].plot([-0.83, 0.83, 0.83, -0.83, -0.83],
                 [1.5, 1.5, 3.5, 3.5, 1.5], 'k-', linewidth=1)
    axes[i].set_xlim(-2.5, 2.5)
    axes[i].set_ylim(0, 5)
    axes[i].set_title(f'FS Location - {period}')
    axes[i].set_xlabel('Plate X')
    axes[i].set_ylabel('Plate Z')
    axes[i].legend(fontsize=8)
    axes[i].set_aspect('equal')

plt.suptitle('Shota Imanaga - FS (Split-finger) Location by Period')
plt.tight_layout()
plt.show()

# Average FS location
fs_location = con.execute("""
    SELECT
        period,
        ROUND(AVG(plate_x), 2) as avg_plate_x,
        ROUND(AVG(plate_z), 2) as avg_plate_z,
        ROUND(AVG(pfx_x), 1) as avg_h_break,
        ROUND(AVG(pfx_z), 1) as avg_v_break,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type = 'FS' AND plate_x IS NOT NULL
    GROUP BY period
    ORDER BY period
""").df()

print('\n=== FS Average Location & Movement ===')
print(fs_location.to_string(index=False))

print('=' * 60)
print('SHOTA IMANAGA 2024-2025 ANALYSIS SUMMARY')
print('=' * 60)

# Games & Pitches
print('\n[Workload]')
for _, row in summary.iterrows():
    print(f'  {row["period"]}: {int(row["games"])} games, {int(row["pitches"]):,} pitches, avg {row["avg_velo"]} mph')

# Pitch mix changes
print(f'\n[Pitch Mix Changes]')
for period_pair in [('2024', '2025-1H'), ('2025-1H', '2025-2H')]:
    p1, p2 = period_pair
    if p1 in mix_pivot.index and p2 in mix_pivot.index:
        first = mix_pivot.loc[p1]
        last = mix_pivot.loc[p2]
        diff = (last - first).sort_values()
        changes = [(p, c) for p, c in diff.items() if abs(c) >= 2.0]
        if changes:
            print(f'  {p1} → {p2}:')
            for pitch, change in changes:
                direction = '↑' if change > 0 else '↓'
                print(f'    {pitch}: {first[pitch]:.1f}% → {last[pitch]:.1f}% ({direction}{abs(change):.1f}%)')

# Velocity
print(f'\n[Fastball Velocity]')
ff_data = velo_by_period[velo_by_period['pitch_type'] == ff_type]
for _, row in ff_data.iterrows():
    print(f'  {row["period"]}: {row["avg_velo"]} mph ({int(row["count"])} pitches)')

# Best whiff pitch per period
print(f'\n[Best Whiff Rate Pitch]')
for period in PERIODS:
    period_whiff = whiff[(whiff['period'] == period) & (whiff['total_swings'] >= 20)]
    if len(period_whiff) > 0:
        best = period_whiff.loc[period_whiff['whiff_rate'].idxmax()]
        print(f'  {period}: {best["pitch_type"]} ({best["whiff_rate"]}%)')

# Batted ball
print(f'\n[Batted Ball Quality]')
for _, row in batted.iterrows():
    print(f'  {row["period"]}: xwOBA {row["avg_xwOBA"]}, Hard Hit {row["hard_hit_pct"]}%, Exit Velo {row["avg_exit_velo"]} mph')

print('\n' + '=' * 60)

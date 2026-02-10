!pip install duckdb -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

# ====== Settings ======
PITCHER_ID = 579328  # Yusei Kikuchi MLBAM ID
YEARS = list(range(2019, 2026))
GAME_TYPE = 'R'  # Regular season only
TRADE_DATE = '2024-07-30'  # Traded to Astros on Jul 29
# ======================

PERIOD_ORDER = ['2019', '2020', '2021', '2022', '2023', '2024-TOR', '2024-HOU', '2025']
KEY_PERIODS = ['2022', '2023', '2024-TOR', '2024-HOU', '2025']

TEAM_MAP = {
    '2019': 'SEA', '2020': 'SEA', '2021': 'SEA',
    '2022': 'TOR', '2023': 'TOR', '2024-TOR': 'TOR',
    '2024-HOU': 'HOU', '2025': 'LAA'
}

# Load from Kaggle dataset (much faster than API calls)
df_all = pd.read_csv('/kaggle/input/japan-mlb-pitchers-batters-statcast/japanese_mlb_pitching.csv')
df_raw = df_all[df_all['pitcher'] == PITCHER_ID].copy()
df_raw['season'] = df_raw['game_year']

for year in YEARS:
    n = len(df_raw[df_raw['season'] == year])
    if n > 0:
        print(f'  {year}: {n:,} pitches')

print(f'\nTotal (raw): {len(df_raw):,} pitches')

# Filter regular season + add period column
con = duckdb.connect()
df = con.execute(f"""
    SELECT *,
        CASE
            WHEN season = 2024 AND game_date::DATE < '{TRADE_DATE}' THEN '2024-TOR'
            WHEN season = 2024 THEN '2024-HOU'
            ELSE CAST(season AS VARCHAR)
        END as period
    FROM df_raw
    WHERE game_type = '{GAME_TYPE}'
""").df()

print(f'Total (regular season): {len(df):,} pitches')
print(f'\nPeriod breakdown:')
for period in PERIOD_ORDER:
    n = len(df[df['period'] == period])
    if n > 0:
        print(f'  {period} ({TEAM_MAP.get(period, "?")}): {n:,} pitches')

PERIOD_SORT = """CASE period
    WHEN '2019' THEN 1 WHEN '2020' THEN 2 WHEN '2021' THEN 3
    WHEN '2022' THEN 4 WHEN '2023' THEN 5 WHEN '2024-TOR' THEN 6
    WHEN '2024-HOU' THEN 7 WHEN '2025' THEN 8
END"""

summary = con.execute(f"""
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
    ORDER BY {PERIOD_SORT}
""").df()

print('=== Career Overview ===')
print(summary.to_string(index=False))
print()
for _, row in summary.iterrows():
    team = TEAM_MAP.get(row['period'], '?')
    print(f'  {row["period"]} ({team}): {int(row["games"])} GS, {int(row["pitches"]):,} pitches')

arsenal = con.execute(f"""
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
    ORDER BY {PERIOD_SORT}, count DESC
""").df()

print('=== Pitch Arsenal by Period ===')
for period in KEY_PERIODS:
    data = arsenal[arsenal['period'] == period]
    if len(data) > 0:
        print(f'\n--- {period} ({TEAM_MAP.get(period, "?")}) ---')
        print(data[['pitch_type', 'count', 'pct', 'avg_velo', 'avg_spin']].to_string(index=False))

top_pitches_overall = con.execute("""
    SELECT pitch_type FROM df
    WHERE pitch_type IS NOT NULL
    GROUP BY pitch_type
    HAVING COUNT(*) >= 50
    ORDER BY COUNT(*) DESC
""").df()['pitch_type'].tolist()

mix_pivot = arsenal.pivot_table(index='period', columns='pitch_type', values='pct', fill_value=0)
available_periods = [p for p in PERIOD_ORDER if p in mix_pivot.index]
mix_pivot = mix_pivot.reindex(available_periods)

plot_cols = [col for col in top_pitches_overall if col in mix_pivot.columns]
mix_pivot[plot_cols].plot(kind='bar', stacked=True, figsize=(14, 7), colormap='Set2')
plt.title('Yusei Kikuchi - Pitch Mix Evolution (2019-2025)')
plt.xlabel('Period')
plt.ylabel('Usage %')
plt.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Pitch Mix (% usage) ===')
print(mix_pivot[plot_cols].round(1).to_string())

print(f'\n=== Key Changes ===')
for pair in [('2023', '2024-TOR'), ('2024-TOR', '2024-HOU'), ('2024-HOU', '2025')]:
    p1, p2 = pair
    if p1 in mix_pivot.index and p2 in mix_pivot.index:
        first = mix_pivot.loc[p1]
        last = mix_pivot.loc[p2]
        diff = (last - first).sort_values()
        changes = [(p, c) for p, c in diff.items() if abs(c) >= 3.0]
        if changes:
            print(f'\n  {p1} -> {p2}:')
            for pitch, change in changes:
                direction = '+' if change > 0 else ''
                print(f'    {pitch}: {first[pitch]:.1f}% -> {last[pitch]:.1f}% ({direction}{change:.1f}%)')

# Identify slider pitch type(s) - could be SL or ST
slider_types = con.execute("""
    SELECT pitch_type, COUNT(*) as cnt
    FROM df
    WHERE pitch_type IN ('SL', 'ST', 'SV', 'FC')
    GROUP BY pitch_type
    ORDER BY cnt DESC
""").df()
print('=== Breaking Ball Types ===')
print(slider_types.to_string(index=False))

slider_type = slider_types.iloc[0]['pitch_type'] if len(slider_types) > 0 else 'SL'
print(f'\nPrimary slider type: {slider_type}')

# Slider analysis by period
sl_analysis = con.execute(f"""
    SELECT
        period,
        COUNT(*) as pitches,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM df d2 WHERE d2.period = df.period), 1) as usage_pct,
        ROUND(AVG(release_speed), 1) as avg_velo,
        ROUND(AVG(release_spin_rate), 0) as avg_spin,
        ROUND(AVG(pfx_x), 1) as h_break,
        ROUND(AVG(pfx_z), 1) as v_break,
        ROUND(100.0 * SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked'
        ) THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN description IN (
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'foul_bunt',
            'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
        ) THEN 1 ELSE 0 END), 0), 1) as whiff_rate,
        ROUND(AVG(CASE WHEN launch_speed IS NOT NULL THEN estimated_ba_using_speedangle END), 3) as xBA_contact
    FROM df
    WHERE pitch_type = '{slider_type}'
    GROUP BY period
    ORDER BY {PERIOD_SORT}
""").df()

print(f'\n=== {slider_type} Analysis by Period ===')
print(sl_analysis.to_string(index=False))

# Slider location scatter: 2024-TOR vs 2024-HOU vs 2025
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
compare_periods = ['2024-TOR', '2024-HOU', '2025']
for i, period in enumerate(compare_periods):
    sl_data = con.execute(f"""
        SELECT plate_x, plate_z, description
        FROM df
        WHERE pitch_type = '{slider_type}' AND period = '{period}'
          AND plate_x IS NOT NULL AND plate_z IS NOT NULL
    """).df()
    if len(sl_data) > 0:
        whiff_mask = sl_data['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
        axes[i].scatter(sl_data[~whiff_mask]['plate_x'], sl_data[~whiff_mask]['plate_z'],
                        alpha=0.3, s=20, c='gray', label='Other')
        axes[i].scatter(sl_data[whiff_mask]['plate_x'], sl_data[whiff_mask]['plate_z'],
                        alpha=0.7, s=30, c='red', label='Whiff')
    axes[i].plot([-0.83, 0.83, 0.83, -0.83, -0.83],
                 [1.5, 1.5, 3.5, 3.5, 1.5], 'k-', linewidth=1)
    axes[i].set_xlim(-2.5, 2.5)
    axes[i].set_ylim(0, 5)
    axes[i].set_title(f'{slider_type} - {period}')
    axes[i].set_xlabel('Plate X')
    axes[i].set_ylabel('Plate Z')
    axes[i].legend(fontsize=8)
    axes[i].set_aspect('equal')

plt.suptitle(f'Yusei Kikuchi - {slider_type} Location: Before & After Trade')
plt.tight_layout()
plt.show()

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

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
period_idx = {p: i for i, p in enumerate(PERIOD_ORDER)}
for pitch in top_pitches:
    data = velo_by_period[velo_by_period['pitch_type'] == pitch].copy()
    data['idx'] = data['period'].map(period_idx)
    data = data.dropna(subset=['idx']).sort_values('idx')
    if len(data) >= 2:
        axes[0].plot(data['period'], data['avg_velo'], marker='o', label=pitch, linewidth=2)
        axes[1].plot(data['period'], data['avg_spin'], marker='o', label=pitch, linewidth=2)

for ax in axes:
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
axes[0].set_title('Average Velocity by Period')
axes[0].set_ylabel('Velocity (mph)')
axes[1].set_title('Average Spin Rate by Period')
axes[1].set_ylabel('Spin Rate (rpm)')
plt.suptitle('Yusei Kikuchi - Velocity & Spin Trends (2019-2025)')
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Velocity & Spin by Period (Top Pitches) ===')
for pitch in top_pitches:
    data = velo_by_period[velo_by_period['pitch_type'] == pitch]
    available = data[data['period'].isin(PERIOD_ORDER)]
    print(f'\n{pitch}:')
    print(available[['period', 'avg_velo', 'avg_spin', 'count']].to_string(index=False))

ff_type = 'FF' if 'FF' in top_pitches else top_pitches[0]
fatigue_periods = ['2023', '2024-TOR', '2024-HOU', '2025']

fatigue = con.execute(f"""
    SELECT
        period,
        inning,
        ROUND(AVG(release_speed), 1) as avg_velo,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type = '{ff_type}' AND inning <= 8
      AND period IN ('2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, inning
    HAVING COUNT(*) >= 5
    ORDER BY period, inning
""").df()

fig, ax = plt.subplots(figsize=(12, 6))
for period in fatigue_periods:
    data = fatigue[fatigue['period'] == period]
    if len(data) > 0:
        ax.plot(data['inning'], data['avg_velo'], marker='o', label=period, linewidth=2)

ax.set_xlabel('Inning')
ax.set_ylabel(f'{ff_type} Velocity (mph)')
ax.set_title(f'Yusei Kikuchi - {ff_type} Velocity by Inning')
ax.set_xticks(range(1, 9))
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print(f'\n=== {ff_type} Velocity by Inning ===')
fatigue_pivot = fatigue.pivot_table(index='inning', columns='period', values='avg_velo')
fatigue_pivot = fatigue_pivot.reindex(columns=fatigue_periods)
print(fatigue_pivot.to_string())

print(f'\n=== Velocity Drop (1st -> last inning) ===')
for period in fatigue_periods:
    data = fatigue[fatigue['period'] == period]
    if len(data) >= 2:
        first_velo = data.iloc[0]['avg_velo']
        last_velo = data.iloc[-1]['avg_velo']
        last_inn = int(data.iloc[-1]['inning'])
        drop = last_velo - first_velo
        print(f'  {period}: {first_velo} -> {last_velo} (inn {last_inn}) = {drop:+.1f} mph')

whiff = con.execute(f"""
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
    ORDER BY {PERIOD_SORT}, total_pitches DESC
""").df()

fig, ax = plt.subplots(figsize=(14, 6))
period_idx = {p: i for i, p in enumerate(KEY_PERIODS)}
for pitch in top_pitches:
    data = whiff[(whiff['pitch_type'] == pitch) & (whiff['period'].isin(KEY_PERIODS))].copy()
    data['idx'] = data['period'].map(period_idx)
    data = data.dropna(subset=['idx']).sort_values('idx')
    if len(data) >= 2:
        ax.plot(data['period'], data['whiff_rate'], marker='o', label=pitch, linewidth=2)

ax.set_ylabel('Whiff Rate (%)')
ax.set_title('Yusei Kikuchi - Whiff Rate by Pitch Type (Key Periods)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

# === Text Summary ===
print('\n=== Whiff Rate by Pitch Type (Key Periods) ===')
whiff_key = whiff[whiff['period'].isin(KEY_PERIODS)]
whiff_pivot = whiff_key.pivot_table(index='pitch_type', columns='period', values='whiff_rate')
whiff_pivot = whiff_pivot.reindex(columns=KEY_PERIODS)
print(whiff_pivot.round(1).to_string())

two_strike = con.execute(f"""
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
      AND period IN ('2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, pitch_type
    ORDER BY {PERIOD_SORT}, pitches DESC
""").df()

print('=== Two-Strike Pitch Selection ===')
for period in ['2023', '2024-TOR', '2024-HOU', '2025']:
    data = two_strike[two_strike['period'] == period].head(5)
    if len(data) > 0:
        print(f'\n--- {period} ({TEAM_MAP.get(period, "?")}) ---')
        print(data[['pitch_type', 'pitches', 'pct', 'whiff_rate']].to_string(index=False))

batted = con.execute(f"""
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
    ORDER BY {PERIOD_SORT}
""").df()

print('=== Batted Ball Results by Period ===')
print(batted.to_string(index=False))

# By pitch type (key periods)
batted_by_pitch = con.execute(f"""
    SELECT
        period,
        pitch_type,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(AVG(estimated_ba_using_speedangle), 3) as avg_xBA
    FROM df
    WHERE launch_speed IS NOT NULL AND pitch_type IS NOT NULL
      AND period IN ('2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, pitch_type
    HAVING COUNT(*) >= 10
    ORDER BY {PERIOD_SORT}, batted_balls DESC
""").df()

print('\n=== Batted Ball by Pitch Type (min 10 BIP) ===')
for period in ['2023', '2024-TOR', '2024-HOU', '2025']:
    data = batted_by_pitch[batted_by_pitch['period'] == period]
    if len(data) > 0:
        print(f'\n--- {period} ---')
        print(data[['pitch_type', 'batted_balls', 'avg_exit_velo', 'avg_xBA']].to_string(index=False))

release = con.execute(f"""
    SELECT
        period,
        pitch_type,
        ROUND(AVG(release_pos_x), 2) as avg_rel_x,
        ROUND(AVG(release_pos_z), 2) as avg_rel_z,
        ROUND(AVG(release_extension), 2) as avg_extension,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type IS NOT NULL
      AND release_pos_x IS NOT NULL
      AND period IN ('2022', '2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, pitch_type
    HAVING COUNT(*) >= 20
    ORDER BY {PERIOD_SORT}, pitches DESC
""").df()

ff_release = release[release['pitch_type'] == ff_type]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(KEY_PERIODS)))
for i, period in enumerate(KEY_PERIODS):
    data = ff_release[ff_release['period'] == period]
    if len(data) > 0:
        axes[0].scatter(data['avg_rel_x'], data['avg_rel_z'], s=200, c=[colors[i]],
                       label=period, zorder=5, edgecolors='black')

axes[0].set_xlabel('Release X (ft, + = 1B side)')
axes[0].set_ylabel('Release Z (ft)')
axes[0].set_title(f'{ff_type} Release Point by Period')
axes[0].legend()

# Extension comparison
ext_data = ff_release[['period', 'avg_extension']].set_index('period')
ext_data = ext_data.reindex([p for p in KEY_PERIODS if p in ext_data.index])
if len(ext_data) > 0:
    axes[1].bar(ext_data.index, ext_data['avg_extension'], color=colors[:len(ext_data)])
    axes[1].set_ylabel('Extension (ft)')
    axes[1].set_title(f'{ff_type} Release Extension')
    axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Yusei Kikuchi - Release Point Analysis')
plt.tight_layout()
plt.show()

# === Text Summary ===
print(f'\n=== {ff_type} Release Point by Period ===')
print(ff_release[['period', 'avg_rel_x', 'avg_rel_z', 'avg_extension', 'pitches']].to_string(index=False))

fig, axes = plt.subplots(1, len(KEY_PERIODS), figsize=(4*len(KEY_PERIODS), 5))

for i, period in enumerate(KEY_PERIODS):
    pitch_data = con.execute(f"""
        SELECT
            pitch_type,
            AVG(pfx_x) as h_break,
            AVG(pfx_z) as v_break,
            COUNT(*) as cnt
        FROM df
        WHERE period = '{period}'
          AND pitch_type IS NOT NULL
          AND pfx_x IS NOT NULL
        GROUP BY pitch_type
        HAVING COUNT(*) >= 10
    """).df()
    for _, row in pitch_data.iterrows():
        axes[i].scatter(row['h_break'], row['v_break'], s=100, zorder=5)
        axes[i].annotate(row['pitch_type'], (row['h_break'], row['v_break']),
                        fontsize=10, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')
    axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    axes[i].set_xlabel('H-Break (in)')
    axes[i].set_ylabel('V-Break (in)')
    axes[i].set_title(period)
    axes[i].set_xlim(-25, 25)
    axes[i].set_ylim(-25, 25)

plt.suptitle('Yusei Kikuchi - Pitch Movement Profile (pitcher POV, inches)')
plt.tight_layout()
plt.show()

# === Text Summary ===
movement = con.execute(f"""
    SELECT
        period,
        pitch_type,
        ROUND(AVG(pfx_x), 1) as h_break,
        ROUND(AVG(pfx_z), 1) as v_break,
        COUNT(*) as pitches
    FROM df
    WHERE pitch_type IS NOT NULL AND pfx_x IS NOT NULL
      AND period IN ('2022', '2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, pitch_type
    HAVING COUNT(*) >= 20
    ORDER BY {PERIOD_SORT}, pitches DESC
""").df()

print('\n=== Pitch Movement by Period (inches) ===')
for period in KEY_PERIODS:
    data = movement[movement['period'] == period]
    if len(data) > 0:
        print(f'\n--- {period} ---')
        print(data[['pitch_type', 'h_break', 'v_break', 'pitches']].to_string(index=False))

lr_arsenal = con.execute(f"""
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
      AND period IN ('2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, stand, pitch_type
    HAVING COUNT(*) >= 10
    ORDER BY {PERIOD_SORT}, stand, count DESC
""").df()

print('=== Pitch Usage & Whiff Rate by Batter Side ===')
for period in ['2023', '2024-TOR', '2024-HOU', '2025']:
    print(f'\n=== {period} ({TEAM_MAP.get(period, "?")}) ===')
    for side in ['L', 'R']:
        data = lr_arsenal[(lr_arsenal['period'] == period) & (lr_arsenal['stand'] == side)]
        if len(data) > 0:
            print(f'\n  vs {side}HB:')
            print(data[['pitch_type', 'count', 'pct', 'whiff_rate']].to_string(index=False))

# Batted ball by side
lr_batted = con.execute(f"""
    SELECT
        period,
        stand,
        COUNT(*) as batted_balls,
        ROUND(AVG(launch_speed), 1) as avg_exit_velo,
        ROUND(AVG(estimated_woba_using_speedangle), 3) as avg_xwOBA,
        ROUND(100.0 * SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) / COUNT(*), 1) as hard_hit_pct
    FROM df
    WHERE launch_speed IS NOT NULL
      AND period IN ('2023', '2024-TOR', '2024-HOU', '2025')
    GROUP BY period, stand
    ORDER BY {PERIOD_SORT}, stand
""").df()

print('\n=== Batted Ball by Batter Side ===')
print(lr_batted.to_string(index=False))

tto = con.execute(f"""
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
        WHERE d.period IN ('2023', '2024-TOR', '2024-HOU', '2025')
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
    ORDER BY {PERIOD_SORT}, tto
""").df()

print('=== Whiff Rate by Time Through Order ===')
tto_pivot = tto.pivot_table(index='tto', columns='period', values='whiff_rate')
tto_pivot = tto_pivot.reindex(columns=['2023', '2024-TOR', '2024-HOU', '2025'])
print(tto_pivot.round(1).to_string())

# Batted ball quality by TTO
tto_batted = con.execute(f"""
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
        WHERE d.period IN ('2023', '2024-TOR', '2024-HOU', '2025')
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
    ORDER BY {PERIOD_SORT}, tto
""").df()

print('\n=== Batted Ball by Time Through Order ===')
for period in ['2023', '2024-TOR', '2024-HOU', '2025']:
    data = tto_batted[tto_batted['period'] == period]
    if len(data) > 0:
        print(f'\n--- {period} ---')
        print(data[['tto', 'batted_balls', 'avg_exit_velo', 'avg_xwOBA']].to_string(index=False))

print('=' * 65)
print('YUSEI KIKUCHI CAREER EVOLUTION SUMMARY (2019-2025)')
print('=' * 65)

# Career arc
print('\n[Career Arc]')
for _, row in summary.iterrows():
    team = TEAM_MAP.get(row['period'], '?')
    print(f'  {row["period"]} ({team}): {int(row["games"])} GS, {int(row["pitches"]):,} pitches, avg {row["avg_velo"]} mph')

# Slider revolution
print(f'\n[Slider Revolution: {slider_type} Usage]')
for period in PERIOD_ORDER:
    sl_data = arsenal[(arsenal['period'] == period) & (arsenal['pitch_type'] == slider_type)]
    if len(sl_data) > 0:
        pct = sl_data.iloc[0]['pct']
        print(f'  {period}: {pct}%')

# FF velocity
print(f'\n[Fastball Velocity ({ff_type})]')
ff_data = velo_by_period[velo_by_period['pitch_type'] == ff_type]
for _, row in ff_data.iterrows():
    if row['period'] in PERIOD_ORDER:
        print(f'  {row["period"]}: {row["avg_velo"]} mph')

# Best whiff pitch
print(f'\n[Best Whiff Rate Pitch]')
for period in KEY_PERIODS:
    period_whiff = whiff[(whiff['period'] == period) & (whiff['total_swings'] >= 20)]
    if len(period_whiff) > 0:
        best = period_whiff.loc[period_whiff['whiff_rate'].idxmax()]
        print(f'  {period}: {best["pitch_type"]} ({best["whiff_rate"]}%)')

# Batted ball
print(f'\n[Batted Ball Quality]')
for _, row in batted.iterrows():
    print(f'  {row["period"]}: xwOBA {row["avg_xwOBA"]}, Hard Hit {row["hard_hit_pct"]}%')

print('\n' + '=' * 65)

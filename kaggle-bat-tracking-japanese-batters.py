#!/usr/bin/env python
# coding: utf-8

# # Bat Tracking Analysis: Japanese MLB Batters (2024-2025)
# 
# Analyzing MLB's new Bat Tracking feature (introduced in 2024) for three Japan-affiliated batters.
# 
# ## Featured Players
# 1. **Shohei Ohtani** - LAD (Japanese)
# 2. **Seiya Suzuki** - CHC (Japanese)
# 3. **Lars Nootbaar** - STL (WBC 2023 Team Japan)
# 
# ## Data Source
# - [MLB Bat Tracking Leaderboard 2024-2025](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print('Libraries imported successfully')


# ## Load Data

# In[ ]:


# Load Bat Tracking data from Kaggle Dataset
df = pd.read_csv('/kaggle/input/mlb-bat-tracking-2024-2025/mlb_bat_tracking_2024_2025.csv')

print(f'Total records: {len(df)}')
print(f'Unique batters: {df["id"].nunique()}')
print(f'Columns: {len(df.columns)}')
df.head()


# ## Filter Japan-Affiliated Players

# In[ ]:


# Japan-affiliated batters (MLBAM IDs)
japanese_ids = {
    660271: 'Shohei Ohtani',
    673548: 'Seiya Suzuki',
    663457: 'Lars Nootbaar'
}

# Filter data
df_jp = df[df['id'].isin(japanese_ids.keys())].copy()
df_jp['player_name'] = df_jp['id'].map(japanese_ids)

print(f'Japan-affiliated batters: {len(df_jp)} records')
df_jp[['player_name', 'season', 'avg_bat_speed', 'swing_length', 'squared_up_per_bat_contact', 'hard_swing_rate']]


# ## Compare with MLB Average

# In[ ]:


# Calculate MLB averages (2024 only to avoid duplicates)
df_2024 = df[df['season'] == 2024]
mlb_avg = df_2024[['avg_bat_speed', 'swing_length', 'squared_up_per_bat_contact', 
                     'hard_swing_rate', 'whiff_per_swing']].mean()

print('MLB Average (2024):')
print(mlb_avg)

# Japanese batters (2024 only)
df_jp_2024 = df_jp[df_jp['season'] == 2024]
print('\nJapan-affiliated batters (2024):')
df_jp_2024[['player_name', 'avg_bat_speed', 'swing_length', 
            'squared_up_per_bat_contact', 'hard_swing_rate', 'whiff_per_swing']]


# ## Visualization 1: Bat Speed Comparison

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for 2024 only
players = df_jp_2024['player_name'].values
bat_speeds = df_jp_2024['avg_bat_speed'].values

# Bar chart
bars = ax.bar(players, bat_speeds, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.axhline(mlb_avg['avg_bat_speed'], color='gray', linestyle='--', linewidth=2, 
           label=f'MLB Average: {mlb_avg["avg_bat_speed"]:.2f} mph')

# Labels
ax.set_ylabel('Average Bat Speed (mph)', fontsize=12, fontweight='bold')
ax.set_title('Bat Speed Comparison: Japan-Affiliated MLB Batters vs MLB Average (2024)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, speed in zip(bars, bat_speeds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{speed:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()


# ## Visualization 2: Swing Length vs Bat Speed

# In[ ]:


fig, ax = plt.subplots(figsize=(12, 8))

# Plot all MLB batters (2024)
ax.scatter(df_2024['swing_length'], df_2024['avg_bat_speed'], 
           alpha=0.3, s=50, color='gray', label='MLB All Batters')

# Plot Japanese batters
colors = {'Shohei Ohtani': '#FF6B6B', 'Seiya Suzuki': '#4ECDC4', 'Lars Nootbaar': '#45B7D1'}
for idx, row in df_jp_2024.iterrows():
    ax.scatter(row['swing_length'], row['avg_bat_speed'], 
               s=200, color=colors[row['player_name']], 
               label=row['player_name'], edgecolor='black', linewidth=2)
    ax.annotate(row['player_name'], 
                (row['swing_length'], row['avg_bat_speed']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# MLB average lines
ax.axhline(mlb_avg['avg_bat_speed'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axvline(mlb_avg['swing_length'], color='red', linestyle='--', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Swing Length (ft)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Bat Speed (mph)', fontsize=12, fontweight='bold')
ax.set_title('Swing Length vs Bat Speed: Japan-Affiliated MLB Batters (2024)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ## Visualization 3: Radar Chart (Overall Profile)

# In[ ]:


from math import pi

# Select metrics for radar chart
metrics = ['avg_bat_speed', 'hard_swing_rate', 'squared_up_per_bat_contact', 
           'batted_ball_event_per_swing', 'contact']
metric_labels = ['Bat Speed', 'Hard Swing%', 'Squared-Up%', 'BBE/Swing', 'Contact']

# Create radar chart
fig = plt.figure(figsize=(14, 6))

for i, (idx, player_row) in enumerate(df_jp_2024.iterrows()):
    ax = fig.add_subplot(1, 3, i+1, projection='polar')

    # Get values and normalize (0-1 scale)
    values = []
    for metric in metrics:
        val = player_row[metric]
        norm_val = (val - df_2024[metric].min()) / (df_2024[metric].max() - df_2024[metric].min())
        values.append(norm_val)
    values += values[:1]  # Close the circle

    # Angles
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[player_row['player_name']])
    ax.fill(angles, values, alpha=0.25, color=colors[player_row['player_name']])

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(player_row['player_name'], fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.show()


# ## Key Findings
# 
# ### Bat Speed
# 1. **Shohei Ohtani**: Highest bat speed among the three (75.8 mph), above MLB average
# 2. **Seiya Suzuki & Lars Nootbaar**: Similar bat speed (~73.4 mph), near MLB average
# 
# ### Swing Characteristics
# - Each player shows distinct swing mechanics (efficiency vs power trade-offs)
# - Swing length varies, reflecting different batting approaches
# 
# ### Data Source
# - [MLB Bat Tracking Leaderboard 2024-2025](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025)
# 
# ### Related Analysis
# **Japanese MLB Pitchers:**
# - [Darvish Pitching Evolution (2021-2025)](https://www.kaggle.com/code/yasunorim/darvish-pitching-evolution)
# - [Imanaga Rookie to Sophomore Analysis](https://www.kaggle.com/code/yasunorim/imanaga-rookie-to-sophomore-pitching)
# - [Senga Ghost Fork Analysis](https://www.kaggle.com/code/yasunorim/senga-ghost-fork-analysis-2023-2025)
# - [Kikuchi Slider Revolution](https://www.kaggle.com/code/yasunorim/kikuchi-slider-revolution-2019-2025)

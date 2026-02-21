# Stanford RNA 3D Folding 2 — Template Matching v1 | W&B Offline Sync via kaggle-wandb-sync
# https://pypi.org/project/kaggle-wandb-sync/

# W&B must be set to offline BEFORE importing wandb
import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'stanford-rna-3d-folding-2'
os.environ['WANDB_RUN_GROUP'] = 'template'

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

import wandb

print('Libraries loaded.')
print(f'WANDB_MODE: {os.environ["WANDB_MODE"]}')

# --- Data path detection ---
INPUT_ROOT = Path('/kaggle/input')
SLUG = 'stanford-rna-3d-folding-2'

print('=== /kaggle/input/ structure ===')
for p in sorted(INPUT_ROOT.iterdir()):
    print(f'  {p.name}/')
    for sub in sorted(p.iterdir())[:5]:
        print(f'    {sub.name}')

DATA_DIR = None
for p in INPUT_ROOT.rglob('test_sequences.csv'):
    DATA_DIR = p.parent
    break

if DATA_DIR is None:
    raise FileNotFoundError(f'test_sequences.csv not found under {INPUT_ROOT}')

print(f'\nDATA_DIR: {DATA_DIR}')

# --- Load data ---
test_df = pd.read_csv(DATA_DIR / 'test_sequences.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
train_seq_df = pd.read_csv(DATA_DIR / 'train_sequences.csv')
train_labels_df = pd.read_csv(DATA_DIR / 'train_labels.csv')

print(f'Test sequences:     {len(test_df)} rows')
print(f'Sample submission:  {len(sample_sub)} rows')
print(f'Train sequences:    {len(train_seq_df)} rows')
print(f'Train labels:       {len(train_labels_df)} rows')

# --- Build training structure library ---
label_cols = train_labels_df.columns.tolist()
print(f'Train labels columns: {label_cols}')

train_labels_df['_target'] = train_labels_df['ID'].str.rsplit('_', n=1).str[0]

train_structures = {}
for target_id, group in train_labels_df.groupby('_target', sort=False):
    coords = group[['x_1', 'y_1', 'z_1']].values.astype(np.float64)
    if np.isnan(coords).any():
        nan_frac = np.isnan(coords).mean()
        if nan_frac > 0.5:
            continue
        for col in range(3):
            mask = np.isnan(coords[:, col])
            if mask.any() and not mask.all():
                valid = np.where(~mask)[0]
                coords[mask, col] = np.interp(
                    np.where(mask)[0], valid, coords[valid, col]
                )
    train_structures[target_id] = coords

train_lengths = {tid: len(c) for tid, c in train_structures.items()}
train_ids = np.array(list(train_lengths.keys()))
train_lens = np.array(list(train_lengths.values()))

print(f'\nTraining structures loaded: {len(train_structures)}')
print(f'Length range: {train_lens.min()} - {train_lens.max()} (mean: {train_lens.mean():.1f})')

sample_sub['_target'] = sample_sub['ID'].str.rsplit('_', n=1).str[0]
test_lengths = sample_sub.groupby('_target', sort=False).size()
print(f'\nTest sequences: {len(test_lengths)}')
print(f'Test length range: {test_lengths.min()} - {test_lengths.max()} (mean: {test_lengths.mean():.1f})')

# --- W&B init ---
SEQ_COL = 'sequence' if 'sequence' in test_df.columns else test_df.columns[1]
seq_lengths = test_df[SEQ_COL].str.len()

run = wandb.init(
    project='stanford-rna-3d-folding-2',
    name='template-v1',
    config={
        'approach': 'template_matching',
        'n_structures': 5,
        'n_templates': 5,
        'interpolation': 'linear',
        'n_train_structures': len(train_structures),
    }
)

wandb.log({
    'n_test_sequences': len(test_lengths),
    'seq_len_min': int(test_lengths.min()),
    'seq_len_max': int(test_lengths.max()),
    'seq_len_mean': float(test_lengths.mean()),
    'n_train_structures': len(train_structures),
    'train_len_min': int(train_lens.min()),
    'train_len_max': int(train_lens.max()),
    'train_len_mean': float(train_lens.mean()),
})


# --- Template matching prediction ---
N_TEMPLATES = 5


def interpolate_coords(coords, target_len):
    source_len = len(coords)
    if source_len == target_len:
        return coords.copy()
    src_positions = np.linspace(0, 1, source_len)
    tgt_positions = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, 3))
    for dim in range(3):
        f = interp1d(src_positions, coords[:, dim], kind='linear')
        result[:, dim] = f(tgt_positions)
    return result


def find_closest_templates(target_len, n=5):
    diffs = np.abs(train_lens - target_len)
    closest_idx = np.argsort(diffs)[:n]
    return [(train_ids[i], diffs[i]) for i in closest_idx]


submission = sample_sub.copy()
template_stats = []

for target_id, group in submission.groupby('_target', sort=False):
    test_len = len(group)
    idx = group.index
    templates = find_closest_templates(test_len, N_TEMPLATES)
    for s, (train_tid, len_diff) in enumerate(templates, 1):
        train_coords = train_structures[train_tid]
        pred_coords = interpolate_coords(train_coords, test_len)
        submission.loc[idx, f'x_{s}'] = pred_coords[:, 0].round(3)
        submission.loc[idx, f'y_{s}'] = pred_coords[:, 1].round(3)
        submission.loc[idx, f'z_{s}'] = pred_coords[:, 2].round(3)
    template_stats.append({
        'target_id': target_id,
        'test_len': test_len,
        'templates': [(tid, int(d)) for tid, d in templates],
        'avg_len_diff': float(np.mean([d for _, d in templates])),
    })

submission = submission.drop(columns=['_target'])

avg_diff = np.mean([s['avg_len_diff'] for s in template_stats])
max_diff = max(s['avg_len_diff'] for s in template_stats)
wandb.log({
    'avg_template_len_diff': avg_diff,
    'max_template_len_diff': max_diff,
})

print(f'Submission shape: {submission.shape}')
print(f'Average template length difference: {avg_diff:.1f}')
print(f'Max average length difference: {max_diff:.1f}')

# --- Save ---
OUTPUT_PATH = Path('/kaggle/working/submission.csv')
submission.to_csv(OUTPUT_PATH, index=False)
print(f'Saved: {OUTPUT_PATH}')

wandb.log({
    'n_submission_rows': len(submission),
    'submission_columns': len(submission.columns),
})

template_table = wandb.Table(
    columns=['target_id', 'test_len', 'template_1', 'template_2', 'template_3', 'template_4', 'template_5', 'avg_len_diff'],
    data=[
        [
            s['target_id'], s['test_len'],
            *[f"{tid}(±{d})" for tid, d in s['templates']],
            round(s['avg_len_diff'], 1),
        ]
        for s in template_stats
    ],
)
wandb.log({'template_assignments': template_table})

wandb.finish()
print('W&B run finished (offline).')

# Stanford RNA 3D Folding 2 — Baseline | W&B Offline Sync via kaggle-wandb-sync
# https://pypi.org/project/kaggle-wandb-sync/

# W&B must be set to offline BEFORE importing wandb
import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'stanford-rna-3d-folding-2'
os.environ['WANDB_RUN_GROUP'] = 'baseline'

import numpy as np
import pandas as pd
from pathlib import Path
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
print(f'Test sequences: {len(test_df)} rows')
print(f'Columns: {list(test_df.columns)}')

SEQ_COL = None
ID_COL = None

for priority in ('sequence', 'seq'):
    if priority in test_df.columns:
        SEQ_COL = priority
        break

if SEQ_COL is None:
    for col in test_df.columns:
        if 'seq' in col.lower() and col.lower() != 'all_sequences':
            SEQ_COL = col
            break

for col in test_df.columns:
    if col.lower() in ('target_id', 'id', 'sequence_id'):
        ID_COL = col

if SEQ_COL is None:
    raise ValueError(f'Could not detect sequence column. Columns: {list(test_df.columns)}')
if ID_COL is None:
    ID_COL = test_df.columns[0]

seq_lengths = test_df[SEQ_COL].str.len()

# --- W&B init ---
run = wandb.init(
    project='stanford-rna-3d-folding-2',
    name='baseline-v1',
    config={
        'approach': 'helical_geometry',
        'n_structures': 5,
        'helix_rise': 2.81,
        'helix_radius': 9.0,
        'helix_twist_deg': 32.7,
    }
)

wandb.log({
    'n_test_sequences': len(test_df),
    'seq_len_min': int(seq_lengths.min()),
    'seq_len_max': int(seq_lengths.max()),
    'seq_len_mean': float(seq_lengths.mean()),
})


# --- Baseline: A-form RNA helix geometry ---
def helix_coords(seq_len, rise=2.81, radius=9.0, twist_deg=32.7):
    twist_rad = np.radians(twist_deg)
    indices = np.arange(seq_len)
    x = radius * np.cos(indices * twist_rad)
    y = radius * np.sin(indices * twist_rad)
    z = indices * rise
    return np.stack([x, y, z], axis=1)


rows = []
for _, row in test_df.iterrows():
    seq_id = row[ID_COL]
    seq    = row[SEQ_COL]
    n      = len(seq)
    coords = helix_coords(n)

    for resid, nuc in enumerate(seq, start=1):
        entry = {'ID': seq_id, 'resname': nuc, 'resid': resid}
        for s in range(1, 6):
            entry[f'x_{s}'] = round(float(coords[resid - 1, 0]), 3)
            entry[f'y_{s}'] = round(float(coords[resid - 1, 1]), 3)
            entry[f'z_{s}'] = round(float(coords[resid - 1, 2]), 3)
        rows.append(entry)

submission = pd.DataFrame(rows)
print(f'Submission shape: {submission.shape}')

# --- Save ---
OUTPUT_PATH = Path('/kaggle/working/submission.csv')
submission.to_csv(OUTPUT_PATH, index=False)
print(f'Saved: {OUTPUT_PATH}')

wandb.log({
    'n_submission_rows': len(submission),
    'submission_columns': len(submission.columns),
})

wandb.finish()
print('W&B run finished (offline).')

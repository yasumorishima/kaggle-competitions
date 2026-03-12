"""Generate Stanford RNA 3D Folding 2 — v9 Multi-Approach Pipeline (optimized).

5 structure slots filled by 5 genuinely different prediction methods:
1. Nussinov secondary structure → 3D geometry (stems as A-form helices, loops as arcs)
2. MSA co-evolution contact map → distance geometry optimization (scipy minimize)
3. 1D ResNet regression trained on train_labels (sequence → xyz)
4. k-mer similarity template matching (not just length)
5. Ensemble refinement: average of structures 1-4 with contact-guided energy minimization

v9 optimizations vs v8 (timeout fix):
- Nussinov DP capped at 300 residues (O(n³) → helix fallback for longer)
- Co-evolution matrix: Python double loop → NumPy reshape+vectorize
- Distance geometry energy: Python loops → NumPy vectorized
- Ensemble energy: Python loops → NumPy vectorized
- L-BFGS maxiter 500→200, ftol 1e-6→1e-5
- ResNet epochs 30→15

GPU enabled for ResNet training.
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
    cells.append(
        {
            "cell_type": "code",
            "id": f"cell-{cell_counter:03d}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        }
    )


# ── Title ──
add_md(
    """# Stanford RNA 3D Folding 2 — v9 Multi-Approach Pipeline

**5 genuinely different methods, one per structure slot:**
1. **Nussinov 2D → 3D geometry** (stems as A-form helices, loops as smooth arcs)
2. **MSA co-evolution → distance geometry** (contact map from covariance → L-BFGS optimization)
3. **1D ResNet** (trained on competition train data, sequence features → xyz regression)
4. **k-mer template matching** (sequence similarity, not just length)
5. **Ensemble refinement** (average + contact-guided energy minimization)

Tools: [`kaggle-wandb-sync`](https://pypi.org/project/kaggle-wandb-sync/) / [`kaggle-notebook-deploy`](https://pypi.org/project/kaggle-notebook-deploy/)"""
)

# ── Setup ──
add_code(
    """import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'stanford-rna-3d-folding-2'

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb

print('Libraries loaded.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')"""
)

# ── W&B init ──
add_code(
    """run = wandb.init(
    project='stanford-rna-3d-folding-2',
    name='multi-approach-v9',
    tags=['nussinov', 'msa-coevo', 'resnet', 'kmer-template', 'ensemble', 'gpu'],
    config={
        'approach': 'multi_approach_5_slots',
        'slot_1': 'nussinov_3d',
        'slot_2': 'msa_distance_geometry',
        'slot_3': 'resnet_regression',
        'slot_4': 'kmer_template_matching',
        'slot_5': 'ensemble_refinement',
    },
)
print(f'W&B run: {run.name}')"""
)

# ── Load data ──
add_code(
    """INPUT_ROOT = Path('/kaggle/input')
SLUG = 'stanford-rna-3d-folding-2'

DATA_DIR = None
for p in INPUT_ROOT.rglob('test_sequences.csv'):
    DATA_DIR = p.parent
    break
if DATA_DIR is None:
    raise FileNotFoundError(f'test_sequences.csv not found under {INPUT_ROOT}')
print(f'DATA_DIR: {DATA_DIR}')

test_df = pd.read_csv(DATA_DIR / 'test_sequences.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
train_seq_df = pd.read_csv(DATA_DIR / 'train_sequences.csv')
train_labels_df = pd.read_csv(DATA_DIR / 'train_labels.csv')

MSA_DIR = DATA_DIR / 'MSA'
has_msa = MSA_DIR.exists()
print(f'MSA directory exists: {has_msa}')
if has_msa:
    msa_files = list(MSA_DIR.glob('*.fas')) + list(MSA_DIR.glob('*.fasta')) + list(MSA_DIR.glob('*.a3m'))
    print(f'MSA files: {len(msa_files)}')

SEQ_COL = 'sequence' if 'sequence' in test_df.columns else test_df.columns[1]

print(f'Test: {len(test_df)}, Train seq: {len(train_seq_df)}, Train labels: {len(train_labels_df)}')
print(f'Sample submission: {len(sample_sub)} rows')

# Parse submission structure
sample_sub['_target'] = sample_sub['ID'].str.rsplit('_', n=1).str[0]
test_lengths = sample_sub.groupby('_target', sort=False).size().to_dict()
print(f'Test sequences: {len(test_lengths)}')
for tid, tlen in list(test_lengths.items())[:5]:
    print(f'  {tid}: {tlen} residues')"""
)

# ── Build train structure library ──
add_code(
    """# Parse training structures (for template matching + ResNet training)
train_labels_df['_target'] = train_labels_df['ID'].str.rsplit('_', n=1).str[0]

train_structures = {}
for target_id, group in train_labels_df.groupby('_target', sort=False):
    coords = group[['x_1', 'y_1', 'z_1']].values.astype(np.float64)
    nan_frac = np.isnan(coords).mean()
    if nan_frac > 0.5:
        continue
    if nan_frac > 0:
        for col in range(3):
            mask = np.isnan(coords[:, col])
            if mask.any() and not mask.all():
                valid = np.where(~mask)[0]
                coords[mask, col] = np.interp(np.where(mask)[0], valid, coords[valid, col])
    train_structures[target_id] = coords

# Map train target_id to sequence
train_seq_map = {}
seq_col_train = 'sequence' if 'sequence' in train_seq_df.columns else train_seq_df.columns[1]
id_col_train = 'target_id' if 'target_id' in train_seq_df.columns else train_seq_df.columns[0]
for _, row in train_seq_df.iterrows():
    train_seq_map[row[id_col_train]] = row[seq_col_train]

train_ids = np.array(list(train_structures.keys()))
train_lens = np.array([len(train_structures[tid]) for tid in train_ids])
print(f'Training structures: {len(train_structures)}')
print(f'Length range: {train_lens.min()} - {train_lens.max()} (mean: {train_lens.mean():.1f})')"""
)

# ── Utilities ──
add_code(
    """# ========================================
# Shared utilities
# ========================================
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}

def seq_to_onehot(seq, max_len=None):
    \"\"\"Convert RNA sequence to one-hot encoding (N_bases x 5).\"\"\"
    n = len(seq) if max_len is None else max_len
    oh = np.zeros((n, 5), dtype=np.float32)
    for i, c in enumerate(seq[:n]):
        oh[i, BASE_MAP.get(c.upper(), 4)] = 1.0
    return oh

def interpolate_coords(coords, target_len):
    source_len = len(coords)
    if source_len == target_len:
        return coords.copy()
    src_pos = np.linspace(0, 1, source_len)
    tgt_pos = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, 3))
    for dim in range(3):
        f = interp1d(src_pos, coords[:, dim], kind='linear')
        result[:, dim] = f(tgt_pos)
    return result

def center_coords(coords):
    return coords - coords.mean(axis=0)

def kmer_profile(seq, k=3):
    \"\"\"Compute k-mer frequency profile.\"\"\"
    kmers = defaultdict(int)
    for i in range(len(seq) - k + 1):
        kmers[seq[i:i+k]] += 1
    total = max(sum(kmers.values()), 1)
    return {kmer: count / total for kmer, count in kmers.items()}

def kmer_distance(seq1, seq2, k=3):
    \"\"\"Cosine-like distance between k-mer profiles.\"\"\"
    p1 = kmer_profile(seq1, k)
    p2 = kmer_profile(seq2, k)
    all_kmers = set(p1.keys()) | set(p2.keys())
    v1 = np.array([p1.get(km, 0) for km in all_kmers])
    v2 = np.array([p2.get(km, 0) for km in all_kmers])
    dot = np.dot(v1, v2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-10:
        return 1.0
    return 1.0 - dot / norm

print('Utilities loaded.')"""
)

# ── Method 1: Nussinov 2D → 3D ──
add_code(
    """# ========================================
# Method 1: Nussinov secondary structure → 3D geometry
# ========================================
print('=' * 60)
print('  Method 1: Nussinov 2D → 3D Geometry')
print('=' * 60)

VALID_PAIRS = {('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')}
MIN_LOOP_SIZE = 3

NUSSINOV_MAX_LEN = 300  # O(n^3) — skip DP for longer sequences

def nussinov_dp(seq):
    \"\"\"Nussinov dynamic programming for RNA secondary structure.\"\"\"
    n = len(seq)
    if n > NUSSINOV_MAX_LEN:
        return []  # helix fallback for long sequences
    dp = np.zeros((n, n), dtype=int)

    for span in range(MIN_LOOP_SIZE + 1, n):
        for i in range(n - span):
            j = i + span
            # Case 1: j unpaired
            dp[i][j] = dp[i][j-1]
            # Case 2: i-j pair
            if (seq[i].upper(), seq[j].upper()) in VALID_PAIRS:
                val = dp[i+1][j-1] + 1
                dp[i][j] = max(dp[i][j], val)
            # Case 3: bifurcation
            for k in range(i+1, j):
                val = dp[i][k] + dp[k+1][j]
                dp[i][j] = max(dp[i][j], val)

    # Traceback
    pairs = []
    def traceback(i, j):
        if i >= j:
            return
        if dp[i][j] == dp[i][j-1]:
            traceback(i, j-1)
        elif (seq[i].upper(), seq[j].upper()) in VALID_PAIRS and dp[i][j] == dp[i+1][j-1] + 1:
            pairs.append((i, j))
            traceback(i+1, j-1)
        else:
            for k in range(i+1, j):
                if dp[i][j] == dp[i][k] + dp[k+1][j]:
                    traceback(i, k)
                    traceback(k+1, j)
                    break

    traceback(0, n-1)
    return pairs

def pairs_to_3d(seq_len, pairs, rise=2.81, radius=9.0, twist_deg=32.7):
    \"\"\"Convert secondary structure to 3D coordinates.
    Stems: A-form helix geometry.
    Loops/unpaired: smooth arc interpolation.
    \"\"\"
    coords = np.zeros((seq_len, 3))

    # Identify stems (consecutive base pairs)
    paired = {}
    for i, j in pairs:
        paired[i] = j
        paired[j] = i

    # Build segments: stem regions and loop regions
    assigned = np.zeros(seq_len, dtype=bool)
    twist_rad = np.radians(twist_deg)

    # Place paired residues as helical segments
    stem_groups = []
    used_pairs = set()
    for i, j in sorted(pairs):
        if (i, j) in used_pairs:
            continue
        stem = [(i, j)]
        used_pairs.add((i, j))
        ci, cj = i + 1, j - 1
        while ci < cj and (ci, cj) in [(p[0], p[1]) for p in pairs]:
            stem.append((ci, cj))
            used_pairs.add((ci, cj))
            ci += 1
            cj -= 1
        stem_groups.append(stem)

    # Place stems as short helices anchored along the backbone
    current_z = 0.0
    helix_segments = {}
    for sg in stem_groups:
        stem_len = len(sg)
        for k, (si, sj) in enumerate(sg):
            angle_i = k * twist_rad
            angle_j = angle_i + np.pi  # opposite side
            coords[si] = [radius * np.cos(angle_i), radius * np.sin(angle_i), current_z + k * rise]
            coords[sj] = [radius * np.cos(angle_j), radius * np.sin(angle_j), current_z + k * rise]
            assigned[si] = True
            assigned[sj] = True
        current_z += stem_len * rise + 5.0  # gap between stems

    # Interpolate unassigned residues
    assigned_idx = np.where(assigned)[0]
    if len(assigned_idx) > 1:
        for dim in range(3):
            unassigned = np.where(~assigned)[0]
            if len(unassigned) > 0:
                coords[unassigned, dim] = np.interp(unassigned, assigned_idx, coords[assigned_idx, dim])
    elif len(assigned_idx) <= 1:
        # Fall back to helix if no pairs found
        indices = np.arange(seq_len)
        coords[:, 0] = radius * np.cos(indices * twist_rad)
        coords[:, 1] = radius * np.sin(indices * twist_rad)
        coords[:, 2] = indices * rise

    return center_coords(coords)

# Test on a few sequences
t0 = time.time()
nussinov_predictions = {}
for tid in test_lengths:
    seq = test_df[test_df[test_df.columns[0]] == tid][SEQ_COL].values[0]
    pairs = nussinov_dp(seq)
    coords = pairs_to_3d(test_lengths[tid], pairs)
    nussinov_predictions[tid] = coords
    print(f'  {tid}: {test_lengths[tid]} res, {len(pairs)} base pairs')

print(f'Nussinov 3D done in {time.time()-t0:.1f}s')
wandb.log({'nussinov_time': time.time()-t0, 'nussinov_n_pairs_mean': np.mean([len(nussinov_dp(test_df[test_df[test_df.columns[0]] == tid][SEQ_COL].values[0])) for tid in list(test_lengths.keys())[:5]])})"""
)

# ── Method 2: MSA co-evolution → distance geometry ──
add_code(
    """# ========================================
# Method 2: MSA co-evolution → distance geometry
# ========================================
print('\\n' + '=' * 60)
print('  Method 2: MSA Co-evolution → Distance Geometry')
print('=' * 60)

def parse_msa(fasta_path, max_seqs=500):
    \"\"\"Parse FASTA/A3M MSA file.\"\"\"
    sequences = []
    current = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current:
                    sequences.append(''.join(current))
                current = []
            else:
                current.append(line.upper().replace('T', 'U'))
    if current:
        sequences.append(''.join(current))
    return sequences[:max_seqs]

def compute_coevolution_matrix(msa_seqs, query_len):
    \"\"\"Compute covariance-based co-evolution scores from MSA.\"\"\"
    if len(msa_seqs) < 5:
        return np.zeros((query_len, query_len))

    # Encode MSA (align to query length)
    n_seqs = len(msa_seqs)
    encoded = np.zeros((n_seqs, query_len, 5), dtype=np.float32)
    for i, seq in enumerate(msa_seqs):
        for j, c in enumerate(seq[:query_len]):
            encoded[i, j, BASE_MAP.get(c, 4)] = 1.0

    # Flatten per-position features
    flat = encoded.reshape(n_seqs, -1)
    # Covariance
    flat_centered = flat - flat.mean(axis=0)
    cov = flat_centered.T @ flat_centered / max(n_seqs - 1, 1)

    # Extract inter-position covariance — vectorized (no Python double loop)
    cov_reshaped = cov.reshape(query_len, 5, query_len, 5)
    # Frobenius norm of each (5,5) block = sqrt(sum of squares)
    contact_scores = np.sqrt((cov_reshaped ** 2).sum(axis=(1, 3)))
    # Zero out diagonal and make symmetric
    np.fill_diagonal(contact_scores, 0)
    # Keep upper triangle only then mirror
    contact_scores = np.triu(contact_scores, k=1)
    contact_scores = contact_scores + contact_scores.T

    return contact_scores

def distance_geometry_from_contacts(seq_len, contact_map, n_contacts_ratio=1.5):
    \"\"\"Optimize 3D coordinates from predicted contacts using L-BFGS.\"\"\"
    # Select top contacts
    n_contacts = int(seq_len * n_contacts_ratio)
    upper_tri = np.triu_indices(seq_len, k=2)
    scores = contact_map[upper_tri]
    top_idx = np.argsort(scores)[-n_contacts:]
    contact_pairs = [(upper_tri[0][k], upper_tri[1][k]) for k in top_idx]

    # Target distances: contacts ~6Å, sequential ~3.8Å
    CONTACT_DIST = 6.0
    SEQ_DIST = 3.8

    contact_i = np.array([p[0] for p in contact_pairs])
    contact_j = np.array([p[1] for p in contact_pairs])

    def energy(flat_coords):
        coords = flat_coords.reshape(-1, 3)
        # Sequential distance constraints — vectorized
        diffs_seq = coords[1:] - coords[:-1]
        dists_seq = np.sqrt((diffs_seq ** 2).sum(axis=1))
        loss = ((dists_seq - SEQ_DIST) ** 2).sum()
        # Contact constraints — vectorized
        if len(contact_i) > 0:
            diffs_c = coords[contact_j] - coords[contact_i]
            dists_c = np.sqrt((diffs_c ** 2).sum(axis=1))
            loss += 0.5 * ((dists_c - CONTACT_DIST) ** 2).sum()
        # Repulsion — vectorized with subsampled pairs
        rep_i = np.arange(0, seq_len, 3)
        rep_pairs_i = []
        rep_pairs_j = []
        for ri in rep_i:
            rj = np.arange(ri + 3, min(ri + 15, seq_len), 3)
            rep_pairs_i.extend([ri] * len(rj))
            rep_pairs_j.extend(rj)
        if rep_pairs_i:
            rpi = np.array(rep_pairs_i)
            rpj = np.array(rep_pairs_j)
            diffs_r = coords[rpj] - coords[rpi]
            dists_r = np.sqrt((diffs_r ** 2).sum(axis=1))
            mask_close = dists_r < 3.0
            if mask_close.any():
                loss += (10.0 * (3.0 - dists_r[mask_close]) ** 2).sum()
        return loss

    # Initialize with helix
    twist = np.radians(32.7)
    init_coords = np.zeros((seq_len, 3))
    idx = np.arange(seq_len)
    init_coords[:, 0] = 9.0 * np.cos(idx * twist)
    init_coords[:, 1] = 9.0 * np.sin(idx * twist)
    init_coords[:, 2] = idx * 2.81

    result = minimize(energy, init_coords.flatten(), method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-5})
    return center_coords(result.x.reshape(-1, 3))

# Process each test sequence
t0 = time.time()
coevo_predictions = {}

for tid in test_lengths:
    seq_len = test_lengths[tid]
    seq = test_df[test_df[test_df.columns[0]] == tid][SEQ_COL].values[0]

    # Try to find MSA file
    contact_map = np.zeros((seq_len, seq_len))
    if has_msa:
        msa_candidates = [f for f in msa_files if tid in f.stem]
        if msa_candidates:
            msa_seqs = parse_msa(msa_candidates[0])
            if len(msa_seqs) >= 5:
                contact_map = compute_coevolution_matrix(msa_seqs, seq_len)
                print(f'  {tid}: MSA={len(msa_seqs)} seqs, contact_max={contact_map.max():.3f}')

    if contact_map.max() < 0.01:
        # No MSA or weak signal — use Nussinov pairs as pseudo-contacts
        pairs = nussinov_dp(seq)
        for i, j in pairs:
            if i < seq_len and j < seq_len:
                contact_map[i, j] = 1.0
                contact_map[j, i] = 1.0
        print(f'  {tid}: No MSA, using {len(pairs)} Nussinov pairs as pseudo-contacts')

    coords = distance_geometry_from_contacts(seq_len, contact_map)
    coevo_predictions[tid] = coords

print(f'Distance geometry done in {time.time()-t0:.1f}s')
wandb.log({'dg_time': time.time()-t0})"""
)

# ── Method 3: 1D ResNet regression ──
add_code(
    """# ========================================
# Method 3: 1D ResNet — sequence → xyz regression
# ========================================
print('\\n' + '=' * 60)
print('  Method 3: 1D ResNet Regression')
print('=' * 60)

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)

class RNAResNet(nn.Module):
    def __init__(self, in_channels=5, hidden=128, n_blocks=6):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(hidden) for _ in range(n_blocks)])
        self.output_conv = nn.Conv1d(hidden, 3, 1)  # predict x, y, z

    def forward(self, x):
        # x: (batch, seq_len, in_channels) → permute to (batch, in_channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.output_conv(x)
        return x.permute(0, 2, 1)  # (batch, seq_len, 3)

class RNADataset(Dataset):
    def __init__(self, sequences, coords_list, max_len):
        self.sequences = sequences
        self.coords_list = coords_list
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        coords = self.coords_list[idx]
        oh = seq_to_onehot(seq, self.max_len)
        c = np.zeros((self.max_len, 3), dtype=np.float32)
        c[:len(coords)] = center_coords(coords)
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:len(coords)] = 1.0
        return torch.tensor(oh), torch.tensor(c), torch.tensor(mask)

# Prepare training data
MAX_LEN = 512  # cap for memory
train_seqs = []
train_coords_list = []
skipped = 0
for tid in train_structures:
    seq = train_seq_map.get(tid, '')
    coords = train_structures[tid]
    if len(seq) == 0 or len(coords) > MAX_LEN or len(seq) > MAX_LEN:
        skipped += 1
        continue
    # Ensure seq len matches coords len
    min_len = min(len(seq), len(coords))
    train_seqs.append(seq[:min_len])
    train_coords_list.append(coords[:min_len])

print(f'ResNet training samples: {len(train_seqs)} (skipped {skipped} with len > {MAX_LEN})')

# Train
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-3

dataset = RNADataset(train_seqs, train_coords_list, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = RNAResNet(in_channels=5, hidden=128, n_blocks=6).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_batches = 0
    for oh, coords_gt, mask in loader:
        oh, coords_gt, mask = oh.to(DEVICE), coords_gt.to(DEVICE), mask.to(DEVICE)
        pred = model(oh)
        # MSE loss only on valid positions
        diff = (pred - coords_gt) ** 2
        diff = diff * mask.unsqueeze(-1)
        loss = diff.sum() / mask.sum() / 3
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    if (epoch + 1) % 5 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}')
    wandb.log({'resnet_epoch': epoch+1, 'resnet_loss': avg_loss})

print(f'ResNet trained in {time.time()-t0:.1f}s')
wandb.log({'resnet_train_time': time.time()-t0})"""
)

# ── ResNet inference ──
add_code(
    """# ResNet inference on test
model.eval()
resnet_predictions = {}

with torch.no_grad():
    for tid in test_lengths:
        seq = test_df[test_df[test_df.columns[0]] == tid][SEQ_COL].values[0]
        seq_len = test_lengths[tid]

        if seq_len <= MAX_LEN:
            oh = torch.tensor(seq_to_onehot(seq, MAX_LEN)).unsqueeze(0).to(DEVICE)
            pred = model(oh).cpu().numpy()[0, :seq_len, :]
        else:
            # For sequences > MAX_LEN, predict in overlapping windows and stitch
            stride = MAX_LEN // 2
            all_preds = np.zeros((seq_len, 3))
            counts = np.zeros(seq_len)
            for start in range(0, seq_len, stride):
                end = min(start + MAX_LEN, seq_len)
                chunk_seq = seq[start:end]
                oh = torch.tensor(seq_to_onehot(chunk_seq, MAX_LEN)).unsqueeze(0).to(DEVICE)
                chunk_pred = model(oh).cpu().numpy()[0, :len(chunk_seq), :]
                # Shift z to align with position
                chunk_pred[:, 2] += start * 2.81
                all_preds[start:end] += chunk_pred
                counts[start:end] += 1
            counts = np.maximum(counts, 1)
            pred = all_preds / counts[:, None]

        resnet_predictions[tid] = center_coords(pred)
        print(f'  {tid}: {seq_len} res, pred range x=[{pred[:,0].min():.1f},{pred[:,0].max():.1f}]')

print('ResNet inference done.')
del model
gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()"""
)

# ── Method 4: k-mer template matching ──
add_code(
    """# ========================================
# Method 4: k-mer similarity template matching
# ========================================
print('\\n' + '=' * 60)
print('  Method 4: k-mer Template Matching')
print('=' * 60)

t0 = time.time()
kmer_predictions = {}

# Pre-compute k-mer profiles for all train sequences
train_profiles = {}
for tid in train_structures:
    seq = train_seq_map.get(tid, '')
    if seq:
        train_profiles[tid] = kmer_profile(seq, k=3)

for test_tid in test_lengths:
    test_seq = test_df[test_df[test_df.columns[0]] == test_tid][SEQ_COL].values[0]
    test_len = test_lengths[test_tid]
    test_prof = kmer_profile(test_seq, k=3)

    # Score all training sequences: weighted combination of k-mer similarity and length similarity
    scores = []
    for train_tid in train_structures:
        if train_tid not in train_profiles:
            continue
        train_seq = train_seq_map.get(train_tid, '')
        train_len = len(train_structures[train_tid])

        # k-mer distance
        all_kmers = set(test_prof.keys()) | set(train_profiles[train_tid].keys())
        v1 = np.array([test_prof.get(km, 0) for km in all_kmers])
        v2 = np.array([train_profiles[train_tid].get(km, 0) for km in all_kmers])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        kmer_sim = dot / max(norm, 1e-10)

        # Length similarity
        len_sim = 1.0 / (1.0 + abs(test_len - train_len) / max(test_len, 1))

        # Combined score (k-mer weighted more)
        combined = 0.7 * kmer_sim + 0.3 * len_sim
        scores.append((train_tid, combined))

    # Pick best match
    scores.sort(key=lambda x: -x[1])
    best_tid, best_score = scores[0]
    best_coords = train_structures[best_tid]
    pred_coords = interpolate_coords(best_coords, test_len)
    kmer_predictions[test_tid] = center_coords(pred_coords)
    print(f'  {test_tid}: best match={best_tid} (score={best_score:.3f}, template_len={len(best_coords)})')

print(f'k-mer template matching done in {time.time()-t0:.1f}s')
wandb.log({'kmer_time': time.time()-t0})"""
)

# ── Method 5: Ensemble refinement ──
add_code(
    """# ========================================
# Method 5: Ensemble refinement
# ========================================
print('\\n' + '=' * 60)
print('  Method 5: Ensemble Refinement')
print('=' * 60)

def refine_ensemble(coords_list, seq, n_steps=200):
    \"\"\"Average multiple structure predictions and refine with contact constraints.\"\"\"
    # Average
    avg = np.mean(coords_list, axis=0)

    # Nussinov contacts for constraint
    pairs = nussinov_dp(seq)
    n = len(avg)

    SEQ_DIST = 3.8
    CONTACT_DIST = 8.0

    # Pre-compute valid pair indices
    pair_i = np.array([i for i, j in pairs if i < n and j < n])
    pair_j = np.array([j for i, j in pairs if i < n and j < n])

    def energy(flat):
        c = flat.reshape(-1, 3)
        # Sequential — vectorized
        diffs = c[1:] - c[:-1]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        loss = ((dists - SEQ_DIST) ** 2).sum()
        # Contacts — vectorized
        if len(pair_i) > 0:
            diffs_p = c[pair_j] - c[pair_i]
            dists_p = np.sqrt((diffs_p ** 2).sum(axis=1))
            loss += 0.3 * ((dists_p - CONTACT_DIST) ** 2).sum()
        # Stay close to average
        loss += 0.1 * np.sum((c - avg) ** 2)
        return loss

    result = minimize(energy, avg.flatten(), method='L-BFGS-B',
                      options={'maxiter': n_steps, 'ftol': 1e-5})
    return center_coords(result.x.reshape(-1, 3))

t0 = time.time()
ensemble_predictions = {}

for tid in test_lengths:
    seq = test_df[test_df[test_df.columns[0]] == tid][SEQ_COL].values[0]
    method_coords = [
        nussinov_predictions[tid],
        coevo_predictions[tid],
        resnet_predictions[tid],
        kmer_predictions[tid],
    ]
    refined = refine_ensemble(method_coords, seq)
    ensemble_predictions[tid] = refined

print(f'Ensemble refinement done in {time.time()-t0:.1f}s')
wandb.log({'ensemble_refine_time': time.time()-t0})"""
)

# ── Build submission ──
add_code(
    """# ========================================
# Build final submission: 5 structures from 5 methods
# ========================================
print('\\n' + '=' * 60)
print('  Building Submission')
print('=' * 60)

all_methods = {
    1: ('Nussinov 3D', nussinov_predictions),
    2: ('MSA Distance Geometry', coevo_predictions),
    3: ('ResNet', resnet_predictions),
    4: ('k-mer Template', kmer_predictions),
    5: ('Ensemble Refined', ensemble_predictions),
}

submission = sample_sub.copy()

for target_id, group in submission.groupby('_target', sort=False):
    idx = group.index
    seq_len = len(group)

    for slot, (method_name, preds) in all_methods.items():
        coords = preds[target_id]
        # Ensure length matches
        if len(coords) != seq_len:
            coords = interpolate_coords(coords, seq_len)
        submission.loc[idx, f'x_{slot}'] = coords[:, 0].round(3)
        submission.loc[idx, f'y_{slot}'] = coords[:, 1].round(3)
        submission.loc[idx, f'z_{slot}'] = coords[:, 2].round(3)

submission = submission.drop(columns=['_target'])

# Validate
expected_cols = sample_sub.drop(columns=['_target']).columns.tolist()
assert list(submission.columns) == expected_cols, f'Column mismatch!'
assert len(submission) == len(sample_sub), f'Row count mismatch!'
assert not submission.isnull().any().any(), 'NaN in submission!'

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f'Submission saved: {submission.shape}')
print(f'\\nStructure slots:')
for slot, (name, _) in all_methods.items():
    print(f'  Slot {slot}: {name}')

wandb.log({
    'n_submission_rows': len(submission),
    'methods': [name for _, (name, _) in all_methods.items()],
})"""
)

# ── Summary + W&B finish ──
add_code(
    """print('\\n' + '=' * 60)
print('  SUMMARY')
print('=' * 60)
print(f'5 structure prediction methods:')
for slot, (name, preds) in all_methods.items():
    sizes = [len(preds[tid]) for tid in preds]
    print(f'  Slot {slot}: {name} (avg size: {np.mean(sizes):.0f})')
print(f'\\nSubmission: {submission.shape[0]} rows × {submission.shape[1]} columns')
print(f'All structures centered (mean=0) and NaN-free.')

wandb.finish()
print('\\nW&B offline run saved.')
print('Sync: kaggle-wandb-sync run . --kernel-id yasunorim/s6e3-rna-multi-approach-work')"""
)


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

out_path = "s6e3-rna-multi-approach-work.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path} ({len(cells)} cells)")

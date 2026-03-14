"""Generate Stanford RNA 3D Folding 2 — v4 RhoFold+ Pipeline.

5 structure slots:
1. RhoFold+ single-sequence prediction (pre-trained, Nature Methods)
2. NW+2D structure template matching (Top-1)
3. Fragment-based multi-template assembly (overlapping windows)
4. RhoFold+/template + MSA co-evolution SA refinement (long SA, multi-restart)
5. TM-score weighted ensemble with SA refinement

Strategy:
- Two-stage template search: k-mer pre-filter (top-50) → NW alignment (top-5)
- Secondary structure (Nussinov) used for template scoring
- Fragment assembly: 100-residue windows with 50-residue stride
- MSA SA refinement: 10000 steps, 3 restarts, pick lowest energy
- RhoFold+ loaded from tant64/rhofold Kaggle dataset
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
    """# Stanford RNA 3D Folding 2 — v4 RhoFold+ Pipeline

**5 structure slots:**
1. **RhoFold+** single-sequence inference (pre-trained on 23.7M RNA sequences)
2. **NW+2D template matching** (two-stage: k-mer pre-filter → NW + secondary structure scoring)
3. **Fragment-based multi-template assembly** (100-res windows, best template per window, stitched)
4. **RhoFold+/template + MSA co-evolution SA refinement** (10K steps × 3 restarts)
5. **TM-score weighted ensemble** with SA refinement

Tools: [`kaggle-wandb-sync`](https://pypi.org/project/kaggle-wandb-sync/) / [`kaggle-notebook-deploy`](https://pypi.org/project/kaggle-notebook-deploy/)"""
)

# ── Setup ──
add_code(
    """import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'stanford-rna-3d-folding-2'

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')
import time
import gc
import shutil
import subprocess

import torch
import torch.nn as nn

import wandb

print('Libraries loaded.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')"""
)

# ── Locate RhoFold+ dataset ──
add_code(
    """# ── Locate RhoFold+ dataset ──
RHOFOLD_ROOT = None
for candidate in sorted(Path('/kaggle/input').iterdir()):
    if 'rhofold' in candidate.name.lower():
        RHOFOLD_ROOT = candidate
        break

if RHOFOLD_ROOT is None:
    print('WARNING: RhoFold+ dataset not found in /kaggle/input/')
else:
    print(f'RhoFold+ dataset: {RHOFOLD_ROOT}')

# Discover structure: print all files (up to 80) to understand layout
if RHOFOLD_ROOT:
    file_list = []
    for p in sorted(RHOFOLD_ROOT.rglob('*')):
        if p.is_file():
            rel = p.relative_to(RHOFOLD_ROOT)
            size = p.stat().st_size
            file_list.append((str(rel), size))
    for rel, size in file_list[:80]:
        label = f'{size/1e6:.1f}MB' if size > 1e6 else f'{size/1e3:.1f}KB' if size > 1e3 else f'{size}B'
        print(f'  {rel} ({label})')
    if len(file_list) > 80:
        print(f'  ... and {len(file_list) - 80} more files')
    print(f'Total files: {len(file_list)}')"""
)

# ── RhoFold+ setup ──
add_code(
    """# ── Discover RhoFold+ weights and source code ──
rhofold_weights = None
rhofold_pkg_dir = None
inference_script = None

if RHOFOLD_ROOT:
    # Find weights (.pt files, prioritize largest)
    pt_files = list(RHOFOLD_ROOT.rglob('*.pt'))
    if pt_files:
        pt_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        rhofold_weights = pt_files[0]
        print(f'Weights: {rhofold_weights} ({rhofold_weights.stat().st_size/1e6:.1f}MB)')

    # Find rhofold Python package
    for init_file in RHOFOLD_ROOT.rglob('rhofold/__init__.py'):
        rhofold_pkg_dir = init_file.parent.parent
        break

    # Find inference.py
    for ipy in RHOFOLD_ROOT.rglob('inference.py'):
        inference_script = ipy
        break

    # Also check for config.py inside rhofold package
    if rhofold_pkg_dir:
        config_file = rhofold_pkg_dir / 'rhofold' / 'config.py'
        print(f'Package dir: {rhofold_pkg_dir}')
        print(f'Config exists: {config_file.exists()}')
        if inference_script:
            print(f'Inference script: {inference_script}')

RHOFOLD_AVAILABLE = (rhofold_weights is not None and rhofold_pkg_dir is not None)
print(f'\\nRhoFold+ available: {RHOFOLD_AVAILABLE}')"""
)

# ── Install RhoFold+ dependencies if needed ──
add_code(
    """# ── Install any missing dependencies for RhoFold+ ──
if RHOFOLD_AVAILABLE:
    # Install biopython from local dataset (offline — no internet)
    # Always try to install from wheel first, even if Bio exists (version mismatch possible)
    bp_wheels = sorted(Path('/kaggle/input').rglob('biopython*.whl'))
    print(f'biopython wheels found: {[w.name for w in bp_wheels]}')
    if bp_wheels:
        print(f'Installing biopython from: {bp_wheels[0].name}')
        r = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--quiet', '--no-deps', str(bp_wheels[0])],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            # Force reimport
            for mod_name in list(sys.modules.keys()):
                if mod_name == 'Bio' or mod_name.startswith('Bio.'):
                    del sys.modules[mod_name]
            import Bio
            print(f'biopython {Bio.__version__} installed OK')
        else:
            print(f'biopython install failed: {r.stderr[-300:]}')
    else:
        # No wheel — check if system biopython works
        try:
            import Bio.SeqIO
            print(f'System biopython OK: {Bio.__version__}')
        except (ImportError, ModuleNotFoundError) as e:
            print(f'WARNING: No biopython wheel and system Bio broken: {e}')
            print('RhoFold+ will fail without biopython')

    # Check other RhoFold dependencies
    missing = []
    for pkg in ['ml_collections', 'einops', 'tree']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        pip_names = {'tree': 'dm-tree'}
        install_list = [pip_names.get(p, p) for p in missing]
        print(f'Installing missing packages: {install_list}')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet'] + install_list,
            capture_output=True, text=True,
        )

    # Add package to sys.path
    sys.path.insert(0, str(rhofold_pkg_dir))

    # Verify import
    try:
        from rhofold.config import rhofold_config
        print('rhofold.config imported OK')
    except Exception as e:
        print(f'Import failed: {e}')
        # Try to diagnose
        rhofold_dir = rhofold_pkg_dir / 'rhofold'
        if rhofold_dir.exists():
            print(f'Files in rhofold/: {[p.name for p in rhofold_dir.iterdir()][:20]}')
        RHOFOLD_AVAILABLE = False"""
)

# ── Load RhoFold+ model ──
add_code(
    """# ── Load RhoFold+ model ──
rhofold_model = None

if RHOFOLD_AVAILABLE:
    try:
        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config

        rhofold_model = RhoFold(rhofold_config)

        # Load checkpoint
        ckpt = torch.load(str(rhofold_weights), map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Clean up keys
        cleaned = {}
        for k, v in state_dict.items():
            key = k.replace('module.', '') if k.startswith('module.') else k
            cleaned[key] = v

        missing, unexpected = rhofold_model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f'Missing keys: {len(missing)} (first 5: {missing[:5]})')
        if unexpected:
            print(f'Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})')

        rhofold_model = rhofold_model.to(DEVICE)
        rhofold_model.eval()
        n_params = sum(p.numel() for p in rhofold_model.parameters())
        print(f'RhoFold+ loaded: {n_params:,} parameters on {DEVICE}')

    except Exception as e:
        print(f'RhoFold+ load failed: {e}')
        import traceback
        traceback.print_exc()
        rhofold_model = None
        RHOFOLD_AVAILABLE = False

if not RHOFOLD_AVAILABLE:
    print('RhoFold+ NOT available — will rely on template methods')"""
)

# ── W&B init ──
add_code(
    """run = wandb.init(
    project='stanford-rna-3d-folding-2',
    name='rhofold-v5',
    tags=['rhofold+', 'nw-template', 'fragment-assembly', 'msa-refine', 'tm-ensemble'],
    config={
        'approach': 'rhofold_plus_v4',
        'rhofold_available': RHOFOLD_AVAILABLE,
        'slot_1': 'rhofold_single_seq',
        'slot_2': 'nw_2d_template_top1',
        'slot_3': 'fragment_assembly',
        'slot_4': 'msa_sa_refine',
        'slot_5': 'tm_weighted_ensemble_sa',
    },
)
print(f'W&B run: {run.name}')"""
)

# ── Load competition data ──
add_code(
    """INPUT_ROOT = Path('/kaggle/input')
DATA_DIR = None
for p in INPUT_ROOT.rglob('test_sequences.csv'):
    DATA_DIR = p.parent
    break
if DATA_DIR is None:
    raise FileNotFoundError('test_sequences.csv not found')
print(f'DATA_DIR: {DATA_DIR}')

test_df = pd.read_csv(DATA_DIR / 'test_sequences.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
train_seq_df = pd.read_csv(DATA_DIR / 'train_sequences.csv')
train_labels_df = pd.read_csv(DATA_DIR / 'train_labels.csv')

MSA_DIR = DATA_DIR / 'MSA'
has_msa = MSA_DIR.exists()
if has_msa:
    msa_files = list(MSA_DIR.glob('*.fas')) + list(MSA_DIR.glob('*.fasta')) + list(MSA_DIR.glob('*.a3m'))
    print(f'MSA files: {len(msa_files)}')

SEQ_COL = 'sequence' if 'sequence' in test_df.columns else test_df.columns[1]
ID_COL = test_df.columns[0]

print(f'Test: {len(test_df)}, Train seq: {len(train_seq_df)}, Train labels: {len(train_labels_df)}')
print(f'Sample submission: {len(sample_sub)} rows')

sample_sub['_target'] = sample_sub['ID'].str.rsplit('_', n=1).str[0]
test_lengths = sample_sub.groupby('_target', sort=False).size().to_dict()
print(f'Test sequences: {len(test_lengths)}')

# Test sequences dict
test_seqs = {}
for _, row in test_df.iterrows():
    test_seqs[row[ID_COL]] = row[SEQ_COL]"""
)

# ── Build train structure library ──
add_code(
    """# ── Parse training structures ──
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

# Map target_id to sequence
train_seq_map = {}
seq_col_train = 'sequence' if 'sequence' in train_seq_df.columns else train_seq_df.columns[1]
id_col_train = 'target_id' if 'target_id' in train_seq_df.columns else train_seq_df.columns[0]
for _, row in train_seq_df.iterrows():
    train_seq_map[row[id_col_train]] = row[seq_col_train]

# Only keep training entries with both sequence and structure
valid_train_ids = [tid for tid in train_structures if tid in train_seq_map]
print(f'Training structures: {len(train_structures)}, with sequence: {len(valid_train_ids)}')"""
)

# ── Utilities ──
add_code(
    """# ========================================
# Shared utilities
# ========================================
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
VALID_PAIRS = {('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')}

def interpolate_coords(coords, target_len):
    source_len = len(coords)
    if source_len == target_len:
        return coords.copy()
    if source_len == 0:
        return np.zeros((target_len, 3))
    src_pos = np.linspace(0, 1, source_len)
    tgt_pos = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, 3))
    for dim in range(3):
        f = interp1d(src_pos, coords[:, dim], kind='linear')
        result[:, dim] = f(tgt_pos)
    return result

def center_coords(coords):
    return coords - coords.mean(axis=0)

def kabsch_align(mobile, target):
    \"\"\"Kabsch alignment: rotate mobile to minimize RMSD to target.\"\"\"
    n = min(len(mobile), len(target))
    m = mobile[:n] - mobile[:n].mean(axis=0)
    t = target[:n] - target[:n].mean(axis=0)
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_m = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_m @ U.T
    aligned = (R @ m.T).T
    return aligned

def compute_tm_score(coords1, coords2):
    n = min(len(coords1), len(coords2))
    if n < 3:
        return 0.0
    c1 = coords1[:n] - coords1[:n].mean(axis=0)
    c2 = coords2[:n] - coords2[:n].mean(axis=0)
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    c2_aligned = (R @ c2.T).T
    d0 = 1.24 * (max(n, 19) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    di_sq = np.sum((c1 - c2_aligned) ** 2, axis=1)
    return float(np.sum(1.0 / (1.0 + di_sq / (d0 ** 2))) / n)

def simulated_annealing(coords_init, energy_fn, n_steps=3000,
                        T_start=100.0, T_end=0.1, perturb_scale=0.5):
    coords = coords_init.copy()
    best_coords = coords.copy()
    current_energy = energy_fn(coords)
    best_energy = current_energy
    cooling_rate = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    for step in range(n_steps):
        n = len(coords)
        n_perturb = max(1, n // 8)
        indices = np.random.choice(n, size=n_perturb, replace=False)
        perturbation = np.random.randn(n_perturb, 3) * perturb_scale
        new_coords = coords.copy()
        new_coords[indices] += perturbation
        new_energy = energy_fn(new_coords)
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(T, 1e-10)):
            coords = new_coords
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_coords = coords.copy()
        T *= cooling_rate
    return best_coords

def sa_multi_restart(coords_init, energy_fn, n_restarts=3, n_steps=5000,
                     T_start=100.0, T_end=0.05, perturb_scale=0.4):
    \"\"\"Run SA multiple times from perturbed starts, return best result.\"\"\"
    best_overall = coords_init.copy()
    best_energy = energy_fn(coords_init)

    for restart in range(n_restarts):
        if restart == 0:
            init = coords_init.copy()
        else:
            # Perturb initial coordinates
            init = coords_init + np.random.randn(*coords_init.shape) * 1.0
        result = simulated_annealing(init, energy_fn, n_steps, T_start, T_end, perturb_scale)
        e = energy_fn(result)
        if e < best_energy:
            best_energy = e
            best_overall = result
    return best_overall

print('Utilities loaded (SA with multi-restart, Kabsch, TM-score).')"""
)

# ── Nussinov secondary structure ──
add_code(
    """# ========================================
# Nussinov secondary structure prediction
# ========================================
MIN_LOOP_SIZE = 3
NUSSINOV_MAX_LEN = 500

def nussinov_dp(seq):
    \"\"\"Nussinov DP for RNA secondary structure. Returns list of (i,j) base pairs.\"\"\"
    n = len(seq)
    if n > NUSSINOV_MAX_LEN:
        return []
    dp = np.zeros((n, n), dtype=np.int32)
    for span in range(MIN_LOOP_SIZE + 1, n):
        for i in range(n - span):
            j = i + span
            dp[i][j] = dp[i][j-1]
            if (seq[i].upper(), seq[j].upper()) in VALID_PAIRS:
                dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 1)
            for k in range(i+1, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k+1][j])
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

def dot_bracket(seq, pairs):
    \"\"\"Convert pairs to dot-bracket string for comparison.\"\"\"
    n = len(seq)
    db = ['.'] * n
    for i, j in pairs:
        if i < n and j < n:
            db[i] = '('
            db[j] = ')'
    return ''.join(db)

def dot_bracket_similarity(db1, db2):
    \"\"\"Compute similarity between two dot-bracket strings.\"\"\"
    n = min(len(db1), len(db2))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if db1[i] == db2[i])
    return matches / n

print('Nussinov secondary structure prediction ready.')"""
)

# ── k-mer profile utilities ──
add_code(
    """# ========================================
# k-mer pre-filtering for fast template search
# ========================================

def kmer_profile(seq, k=4):
    \"\"\"Compute k-mer frequency vector.\"\"\"
    kmers = defaultdict(int)
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        kmers[seq[i:i+k]] += 1
    total = max(sum(kmers.values()), 1)
    return {kmer: count / total for kmer, count in kmers.items()}

def kmer_cosine(prof1, prof2):
    \"\"\"Cosine similarity between two k-mer profiles.\"\"\"
    all_kmers = set(prof1.keys()) | set(prof2.keys())
    v1 = np.array([prof1.get(km, 0) for km in all_kmers])
    v2 = np.array([prof2.get(km, 0) for km in all_kmers])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / max(norm, 1e-10)

# Pre-compute k-mer profiles for all training sequences
print('Pre-computing k-mer profiles...')
t0 = time.time()
train_kmer_profiles = {}
for tid in valid_train_ids:
    train_kmer_profiles[tid] = kmer_profile(train_seq_map[tid], k=4)
print(f'k-mer profiles computed for {len(train_kmer_profiles)} training sequences in {time.time()-t0:.1f}s')"""
)

# ── Vectorized NW alignment ──
add_code(
    """# ========================================
# Needleman-Wunsch alignment (numpy-optimized)
# ========================================

NW_MATCH = 4
NW_TRANSITION = 2
NW_TRANSVERSION = -2
NW_GAP_OPEN = -8
NW_GAP_EXTEND = -2
TRANSITIONS = {('A','G'), ('G','A'), ('C','U'), ('U','C'), ('C','T'), ('T','C')}

def nw_score_matrix(seq1, seq2):
    \"\"\"Build score matrix for two sequences using numpy broadcasting.\"\"\"
    s1 = np.array([c.upper() for c in seq1])
    s2 = np.array([c.upper() for c in seq2])
    n, m = len(s1), len(s2)

    # Match matrix
    match_mask = (s1[:, None] == s2[None, :])  # (n, m)

    # Transition matrix
    trans_pairs = set()
    for a, b in TRANSITIONS:
        trans_pairs.add((a, b))
    trans_mask = np.zeros((n, m), dtype=bool)
    for a, b in trans_pairs:
        trans_mask |= ((s1[:, None] == a) & (s2[None, :] == b))

    scores = np.full((n, m), NW_TRANSVERSION, dtype=np.float32)
    scores[match_mask] = NW_MATCH
    scores[trans_mask & ~match_mask] = NW_TRANSITION
    return scores

def needleman_wunsch_fast(seq1, seq2, max_dp_len=1500):
    \"\"\"NW alignment with numpy-optimized scoring.
    For sequences longer than max_dp_len, use diagonal band approximation.

    Returns: (nw_score_normalized, sequence_identity)
    \"\"\"
    n, m = len(seq1), len(seq2)

    if n > max_dp_len or m > max_dp_len:
        # Band approximation for very long sequences
        min_len = min(n, m)
        matches = 0
        for i in range(min_len):
            j = int(i * m / n) if n > 0 else i
            j = min(j, m - 1)
            if seq1[i].upper() == seq2[j].upper():
                matches += 1
        identity = matches / max(min_len, 1)
        norm_score = identity * NW_MATCH  # approximate
        return norm_score, identity

    # Pre-compute score matrix
    score_mat = nw_score_matrix(seq1, seq2)

    # DP
    dp = np.full((n + 1, m + 1), -1e9, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = NW_GAP_OPEN + (i - 1) * NW_GAP_EXTEND
    for j in range(1, m + 1):
        dp[0, j] = NW_GAP_OPEN + (j - 1) * NW_GAP_EXTEND

    # Row-by-row DP (can't fully vectorize due to dependencies, but numpy slicing helps)
    for i in range(1, n + 1):
        # Diagonal: dp[i-1, 0:m] + score_mat[i-1, 0:m]
        diag = dp[i-1, :m] + score_mat[i-1]
        # Gap in seq2: dp[i-1, 1:m+1] + gap_extend
        gap1 = dp[i-1, 1:m+1] + NW_GAP_EXTEND
        # Gap in seq1: dp[i, 0:m] + gap_extend
        # This depends on dp[i, j-1] so can't fully vectorize; do it in a loop
        # But we can at least vectorize the max of diag and gap1
        dp[i, 1:m+1] = np.maximum(diag, gap1)

        # Fix gap in seq1 (left-to-right dependency)
        for j in range(1, m + 1):
            dp[i, j] = max(dp[i, j], dp[i, j-1] + NW_GAP_EXTEND)

    # Traceback for identity
    final_score = dp[n, m]
    i, j = n, m
    matches = 0
    aligned_len = 0
    while i > 0 and j > 0:
        score = dp[i, j]
        diag = dp[i-1, j-1] + score_mat[i-1, j-1]
        if abs(score - diag) < 0.01:
            if seq1[i-1].upper() == seq2[j-1].upper():
                matches += 1
            aligned_len += 1
            i -= 1
            j -= 1
        elif abs(score - (dp[i-1, j] + NW_GAP_EXTEND)) < 0.01:
            aligned_len += 1
            i -= 1
        else:
            aligned_len += 1
            j -= 1
    aligned_len += i + j
    identity = matches / max(aligned_len, 1)
    norm_score = final_score / max(aligned_len, 1)

    return norm_score, identity

print('NW alignment ready (numpy-optimized).')"""
)

# ── Two-stage template search ──
add_code(
    """# ========================================
# Two-stage template search:
#   Stage 1: k-mer pre-filter (top-50 per test sequence)
#   Stage 2: NW alignment + 2D structure scoring (top-5)
# ========================================
print('\\n' + '=' * 60)
print('  Two-Stage Template Search')
print('=' * 60)

t0 = time.time()

# Pre-compute 2D structures for training sequences (only short ones)
print('Computing training secondary structures...')
train_2d = {}
for tid in valid_train_ids:
    seq = train_seq_map[tid]
    if len(seq) <= NUSSINOV_MAX_LEN:
        pairs = nussinov_dp(seq)
        train_2d[tid] = dot_bracket(seq, pairs)
print(f'  2D structures computed for {len(train_2d)}/{len(valid_train_ids)} training sequences')

# Compute test 2D structures
test_2d = {}
for tid in test_lengths:
    seq = test_seqs[tid]
    if len(seq) <= NUSSINOV_MAX_LEN:
        pairs = nussinov_dp(seq)
        test_2d[tid] = dot_bracket(seq, pairs)
print(f'  2D structures computed for {len(test_2d)}/{len(test_lengths)} test sequences')

# Two-stage search
KMER_TOP_N = 50
NW_TOP_K = 5
template_rankings = {}

for test_tid in test_lengths:
    test_seq = test_seqs[test_tid]
    test_len = test_lengths[test_tid]
    test_prof = kmer_profile(test_seq, k=4)

    # Stage 1: k-mer pre-filter
    kmer_scores = []
    for train_tid in valid_train_ids:
        train_len = len(train_structures[train_tid])
        # Length filter: skip if >3x ratio
        ratio = max(test_len, train_len) / max(min(test_len, train_len), 1)
        if ratio > 3.0:
            continue
        sim = kmer_cosine(test_prof, train_kmer_profiles[train_tid])
        len_sim = 1.0 / (1.0 + abs(test_len - train_len) / max(test_len, 1))
        kmer_scores.append((train_tid, 0.7 * sim + 0.3 * len_sim))

    kmer_scores.sort(key=lambda x: -x[1])
    candidates = [tid for tid, _ in kmer_scores[:KMER_TOP_N]]

    # Stage 2: NW alignment + 2D structure scoring
    nw_scores = []
    for train_tid in candidates:
        train_seq = train_seq_map[train_tid]
        nw_norm, identity = needleman_wunsch_fast(test_seq, train_seq)

        # 2D structure similarity bonus
        ss_bonus = 0.0
        if test_tid in test_2d and train_tid in train_2d:
            ss_sim = dot_bracket_similarity(test_2d[test_tid], train_2d[train_tid])
            ss_bonus = 0.2 * ss_sim

        # Composite: NW identity (0.5) + NW normalized score (0.15) + length sim (0.15) + 2D sim (0.2)
        train_len = len(train_structures[train_tid])
        len_sim = 1.0 / (1.0 + abs(test_len - train_len) / max(test_len, 1))
        composite = 0.5 * identity + 0.15 * max(nw_norm / NW_MATCH, 0) + 0.15 * len_sim + ss_bonus
        nw_scores.append((train_tid, composite, identity))

    nw_scores.sort(key=lambda x: -x[1])
    template_rankings[test_tid] = nw_scores[:NW_TOP_K]

    top = nw_scores[0] if nw_scores else ('none', 0, 0)
    print(f'  {test_tid} ({test_len}res): top={top[0]} score={top[1]:.3f} identity={top[2]:.3f} ({len(candidates)} candidates)')

elapsed = time.time() - t0
print(f'Two-stage template search done in {elapsed:.1f}s')
wandb.log({'template_search_time': elapsed})"""
)

# ── Method 1: RhoFold+ single-sequence inference ──
add_code(
    """# ========================================
# Method 1: RhoFold+ single-sequence prediction
# ========================================
print('\\n' + '=' * 60)
print('  Method 1: RhoFold+ Single-Sequence Prediction')
print('=' * 60)

t0 = time.time()
rhofold_predictions = {}

def parse_pdb_c1prime(pdb_path):
    \"\"\"Parse C1' atom coordinates from PDB file.\"\"\"
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and "C1'" in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords) if coords else None

if rhofold_model is not None:
    # Method A: Use RhoFold Python API (get_features → model.forward)
    tmp_dir = Path('/kaggle/working/rhofold_tmp')
    tmp_dir.mkdir(exist_ok=True)

    try:
        from rhofold.utils.alphabet import get_features
        from rhofold.utils.converter import output_to_pdb
        print('RhoFold get_features/output_to_pdb imported OK')

        for test_tid in test_lengths:
            test_seq = test_seqs[test_tid]
            test_len = test_lengths[test_tid]

            fasta_path = tmp_dir / f'{test_tid}.fasta'
            with open(fasta_path, 'w') as f:
                f.write(f'>{test_tid}\\n{test_seq}\\n')

            try:
                # single_seq_pred: use fasta as MSA (single sequence)
                data_dict = get_features(str(fasta_path), str(fasta_path))

                with torch.no_grad():
                    output = rhofold_model(
                        tokens=data_dict['tokens'].to(DEVICE),
                        rna_fm_tokens=data_dict['rna_fm_tokens'].to(DEVICE),
                        seq=data_dict['seq'],
                    )

                # Extract coordinates from last recycling output
                last_out = output[-1] if isinstance(output, list) else output
                if 'cord_tns_pred' in last_out:
                    # cord_tns_pred shape: [1, L, n_atoms, 3]
                    all_coords = last_out['cord_tns_pred'].cpu().numpy()[0]
                    # Take C1' atom (index varies, typically index 1 or use first heavy atom)
                    # RhoFold uses atom order: P, C4', ... — try to get C1' or fallback to first
                    if all_coords.ndim == 3:
                        c1_coords = all_coords[:, 1, :]  # C4' position as proxy
                    else:
                        c1_coords = all_coords
                    if len(c1_coords) != test_len:
                        c1_coords = interpolate_coords(c1_coords, test_len)
                    rhofold_predictions[test_tid] = center_coords(c1_coords)
                    print(f'  {test_tid}: forward() OK ({len(c1_coords)} atoms)')
                else:
                    # Try PDB output
                    pdb_path = tmp_dir / f'{test_tid}.pdb'
                    output_to_pdb(output, str(pdb_path))
                    coords = parse_pdb_c1prime(pdb_path)
                    if coords is not None and len(coords) > 0:
                        if len(coords) != test_len:
                            coords = interpolate_coords(coords, test_len)
                        rhofold_predictions[test_tid] = center_coords(coords)
                        print(f'  {test_tid}: PDB output OK ({len(coords)} atoms)')

            except Exception as e:
                print(f'  {test_tid}: forward failed: {type(e).__name__}: {str(e)[:200]}')

    except ImportError as e:
        print(f'Method A import failed: {e} — will try subprocess')

    shutil.rmtree(tmp_dir, ignore_errors=True)

# Method B: If API didn't work, try running inference.py as subprocess
if len(rhofold_predictions) == 0 and inference_script is not None and RHOFOLD_AVAILABLE:
    print('\\nTrying inference.py subprocess...')
    tmp_dir = Path('/kaggle/working/rhofold_sub')
    tmp_dir.mkdir(exist_ok=True)
    out_dir = Path('/kaggle/working/rhofold_out')
    out_dir.mkdir(exist_ok=True)

    for test_tid in test_lengths:
        test_seq = test_seqs[test_tid]
        fasta_path = tmp_dir / f'{test_tid}.fasta'
        with open(fasta_path, 'w') as f:
            f.write(f'>{test_tid}\\n{test_seq}\\n')

        out_sub = out_dir / test_tid
        out_sub.mkdir(exist_ok=True)

        try:
            cmd = [
                sys.executable, str(inference_script),
                '--input_fas', str(fasta_path),
                '--output_dir', str(out_sub),
                '--single_seq_pred', 'True',
                '--device', DEVICE,
                '--ckpt', str(rhofold_weights),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                     cwd=str(inference_script.parent))
            if result.returncode != 0:
                print(f'  {test_tid}: subprocess failed: {result.stderr[-300:]}')
            else:
                pdb_files = sorted(out_sub.rglob('*.pdb'))
                for pdb_path in pdb_files:
                    coords = parse_pdb_c1prime(pdb_path)
                    if coords is not None and len(coords) > 0:
                        if len(coords) != test_lengths[test_tid]:
                            coords = interpolate_coords(coords, test_lengths[test_tid])
                        rhofold_predictions[test_tid] = center_coords(coords)
                        print(f'  {test_tid}: subprocess OK ({len(coords)} atoms)')
                        break
        except subprocess.TimeoutExpired:
            print(f'  {test_tid}: subprocess timed out (600s)')
        except Exception as e:
            print(f'  {test_tid}: subprocess error: {e}')

    shutil.rmtree(tmp_dir, ignore_errors=True)

elapsed = time.time() - t0
print(f'\\nRhoFold+ results: {len(rhofold_predictions)}/{len(test_lengths)} sequences')
print(f'RhoFold+ total time: {elapsed:.1f}s')
wandb.log({'rhofold_time': elapsed, 'rhofold_success': len(rhofold_predictions)})

# Free GPU memory
if rhofold_model is not None:
    del rhofold_model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    print('RhoFold+ model unloaded, GPU memory freed.')"""
)

# ── Method 2: NW+2D template matching (Top-1) ──
add_code(
    """# ========================================
# Method 2: NW+2D Template Matching (Top-1)
# ========================================
print('\\n' + '=' * 60)
print('  Method 2: NW+2D Template Matching (Top-1)')
print('=' * 60)

t0 = time.time()
nw_template_predictions = {}

for test_tid in test_lengths:
    test_len = test_lengths[test_tid]
    rankings = template_rankings.get(test_tid, [])

    if not rankings:
        # Helix fallback
        idx = np.arange(test_len, dtype=np.float64)
        twist = np.radians(32.7)
        coords = np.column_stack([9.0 * np.cos(idx * twist), 9.0 * np.sin(idx * twist), idx * 2.81])
        nw_template_predictions[test_tid] = center_coords(coords)
        print(f'  {test_tid}: no template, helix fallback')
        continue

    best_tid = rankings[0][0]
    template_coords = train_structures[best_tid]
    pred = interpolate_coords(template_coords, test_len)
    nw_template_predictions[test_tid] = center_coords(pred)
    print(f'  {test_tid}: template={best_tid} (score={rankings[0][1]:.3f}, identity={rankings[0][2]:.3f})')

print(f'NW Template Top-1 done in {time.time()-t0:.1f}s')
wandb.log({'nw_template_time': time.time()-t0})"""
)

# ── Method 3: Fragment-based multi-template assembly ──
add_code(
    """# ========================================
# Method 3: Fragment-based Multi-Template Assembly
# ========================================
print('\\n' + '=' * 60)
print('  Method 3: Fragment-Based Multi-Template Assembly')
print('=' * 60)

FRAG_WINDOW = 100   # residues per fragment
FRAG_STRIDE = 50    # overlap between fragments
FRAG_KMER_TOP = 20  # k-mer pre-filter per fragment
FRAG_NW_TOP = 3     # NW alignment per fragment

t0 = time.time()
fragment_predictions = {}

for test_tid in test_lengths:
    test_seq = test_seqs[test_tid]
    test_len = test_lengths[test_tid]

    # For short sequences (< 2 * FRAG_WINDOW), use multi-template average instead
    if test_len < FRAG_WINDOW * 2:
        rankings = template_rankings.get(test_tid, [])
        if len(rankings) >= 2:
            top_k = min(3, len(rankings))
            templates = []
            weights = []
            for i in range(top_k):
                train_tid, score, _ = rankings[i]
                coords = interpolate_coords(train_structures[train_tid], test_len)
                templates.append(center_coords(coords))
                weights.append(score)
            weights = np.array(weights)
            weights /= weights.sum()
            ref = templates[0]
            composite = weights[0] * ref
            for i in range(1, top_k):
                aligned = kabsch_align(templates[i], ref)
                composite += weights[i] * aligned
            fragment_predictions[test_tid] = center_coords(composite)
            print(f'  {test_tid}: short seq, multi-template average ({top_k} templates)')
        else:
            fragment_predictions[test_tid] = nw_template_predictions[test_tid].copy()
            print(f'  {test_tid}: short seq, using Top-1 template')
        continue

    # Fragment-based assembly for long sequences
    assembled = np.zeros((test_len, 3))
    counts = np.zeros(test_len)

    n_frags = 0
    for start in range(0, test_len, FRAG_STRIDE):
        end = min(start + FRAG_WINDOW, test_len)
        frag_len = end - start
        if frag_len < 20:
            continue

        frag_seq = test_seq[start:end]
        frag_prof = kmer_profile(frag_seq, k=4)

        # k-mer pre-filter for this fragment
        frag_kmer_scores = []
        for train_tid in valid_train_ids:
            train_seq = train_seq_map[train_tid]
            train_len = len(train_structures[train_tid])
            # Look for training sequences that cover at least this fragment length
            if train_len < frag_len // 2:
                continue
            sim = kmer_cosine(frag_prof, train_kmer_profiles[train_tid])
            frag_kmer_scores.append((train_tid, sim))

        frag_kmer_scores.sort(key=lambda x: -x[1])
        frag_candidates = [tid for tid, _ in frag_kmer_scores[:FRAG_KMER_TOP]]

        # NW alignment for fragment
        frag_nw_scores = []
        for train_tid in frag_candidates:
            train_seq = train_seq_map[train_tid]
            # Align fragment against full training sequence
            _, identity = needleman_wunsch_fast(frag_seq, train_seq)
            frag_nw_scores.append((train_tid, identity))

        frag_nw_scores.sort(key=lambda x: -x[1])

        # Use top matches: weighted average of aligned templates for this fragment
        top_templates = frag_nw_scores[:FRAG_NW_TOP]
        if not top_templates:
            continue

        frag_coords = np.zeros((frag_len, 3))
        weight_sum = 0
        for train_tid, score in top_templates:
            # Extract best matching region from training structure
            train_coords = train_structures[train_tid]
            # Interpolate full training structure to get fragment-sized piece
            piece = interpolate_coords(train_coords, frag_len)
            frag_coords += score * piece
            weight_sum += score

        if weight_sum > 0:
            frag_coords /= weight_sum

        # Blend into assembled coordinates with triangular weight (higher in center)
        blend_weights = np.ones(frag_len)
        ramp = min(FRAG_STRIDE, frag_len // 4)
        if ramp > 0:
            blend_weights[:ramp] = np.linspace(0.5, 1.0, ramp)
            blend_weights[-ramp:] = np.linspace(1.0, 0.5, ramp)

        assembled[start:end] += frag_coords * blend_weights[:, None]
        counts[start:end] += blend_weights
        n_frags += 1

    # Normalize
    counts = np.maximum(counts, 1e-10)
    assembled /= counts[:, None]
    fragment_predictions[test_tid] = center_coords(assembled)
    print(f'  {test_tid}: {n_frags} fragments assembled ({test_len} res)')

print(f'Fragment assembly done in {time.time()-t0:.1f}s')
wandb.log({'fragment_assembly_time': time.time()-t0})"""
)

# ── MSA parsing ──
add_code(
    """# ========================================
# MSA utilities
# ========================================

def parse_msa(fasta_path, max_seqs=500):
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

def compute_contact_map(msa_seqs, query_len):
    if len(msa_seqs) < 5:
        return np.zeros((query_len, query_len))
    n_seqs = len(msa_seqs)
    encoded = np.zeros((n_seqs, query_len, 5), dtype=np.float32)
    for i, seq in enumerate(msa_seqs):
        for j, c in enumerate(seq[:query_len]):
            encoded[i, j, BASE_MAP.get(c, 4)] = 1.0
    flat = encoded.reshape(n_seqs, -1)
    flat_centered = flat - flat.mean(axis=0)
    cov = flat_centered.T @ flat_centered / max(n_seqs - 1, 1)
    cov_reshaped = cov.reshape(query_len, 5, query_len, 5)
    contact_scores = np.sqrt((cov_reshaped ** 2).sum(axis=(1, 3)))
    np.fill_diagonal(contact_scores, 0)
    contact_scores = np.triu(contact_scores, k=1)
    return contact_scores + contact_scores.T

print('MSA utilities ready.')"""
)

# ── Method 4: MSA co-evolution SA refinement ──
add_code(
    """# ========================================
# Method 4: Best prediction + MSA co-evolution SA refinement
#   - 10000 SA steps × 3 restarts
#   - Sequential distance + contact distance + repulsion + anchor constraints
# ========================================
print('\\n' + '=' * 60)
print('  Method 4: MSA Co-evolution SA Refinement (Heavy)')
print('=' * 60)

t0 = time.time()
msa_refined_predictions = {}

SA_STEPS = 10000
SA_RESTARTS = 3
CONTACT_DIST = 8.0
SEQ_DIST = 3.8
MIN_DIST = 3.2  # repulsion threshold

for test_tid in test_lengths:
    test_seq = test_seqs[test_tid]
    test_len = test_lengths[test_tid]

    # Choose best base prediction: RhoFold+ > fragment assembly > NW template
    if test_tid in rhofold_predictions:
        base_coords = rhofold_predictions[test_tid].copy()
        base_name = 'RhoFold+'
    elif test_tid in fragment_predictions:
        base_coords = fragment_predictions[test_tid].copy()
        base_name = 'fragment'
    else:
        base_coords = nw_template_predictions[test_tid].copy()
        base_name = 'NW-Top1'

    # Get MSA contact map
    contact_map = np.zeros((test_len, test_len))
    msa_depth = 0
    if has_msa:
        msa_candidates = [f for f in msa_files if test_tid in f.stem]
        if msa_candidates:
            msa_seqs = parse_msa(msa_candidates[0])
            msa_depth = len(msa_seqs)
            if msa_depth >= 5:
                contact_map = compute_contact_map(msa_seqs, test_len)

    # Also add Nussinov base pair contacts
    if test_tid in test_2d:
        pairs = nussinov_dp(test_seq)
        for i, j in pairs:
            if i < test_len and j < test_len:
                contact_map[i, j] = max(contact_map[i, j], 0.5)
                contact_map[j, i] = max(contact_map[j, i], 0.5)

    has_contacts = contact_map.max() > 0.01

    if not has_contacts and base_name != 'RhoFold+':
        # No contacts and no RhoFold+ → skip refinement
        msa_refined_predictions[test_tid] = base_coords
        print(f'  {test_tid}: no contacts, no RhoFold+, using {base_name} as-is')
        continue

    # Extract top contacts
    n_contacts = int(test_len * 2.0)
    upper_tri = np.triu_indices(test_len, k=3)  # skip very close residues
    scores = contact_map[upper_tri]
    top_idx = np.argsort(scores)[-n_contacts:]
    contact_i = upper_tri[0][top_idx]
    contact_j = upper_tri[1][top_idx]
    contact_weights = scores[top_idx]
    contact_weights = contact_weights / max(contact_weights.max(), 1e-10)

    # Pre-compute repulsion pairs (subsampled every 2nd residue)
    rep_pairs_i = []
    rep_pairs_j = []
    for ri in range(0, test_len, 2):
        for rj in range(ri + 3, min(ri + 12, test_len), 2):
            rep_pairs_i.append(ri)
            rep_pairs_j.append(rj)
    rep_i = np.array(rep_pairs_i) if rep_pairs_i else np.array([], dtype=int)
    rep_j = np.array(rep_pairs_j) if rep_pairs_j else np.array([], dtype=int)

    ANCHOR_WEIGHT = 0.05 if base_name == 'RhoFold+' else 0.02

    def energy(coords):
        # 1. Sequential distance constraint (backbone)
        diffs = coords[1:] - coords[:-1]
        dists = np.sqrt((diffs ** 2).sum(axis=1) + 1e-10)
        loss = 2.0 * ((dists - SEQ_DIST) ** 2).sum()

        # 2. MSA/Nussinov contact constraints (weighted)
        if len(contact_i) > 0:
            diffs_c = coords[contact_j] - coords[contact_i]
            dists_c = np.sqrt((diffs_c ** 2).sum(axis=1) + 1e-10)
            loss += (contact_weights * (dists_c - CONTACT_DIST) ** 2).sum()

        # 3. Repulsion: prevent steric clashes
        if len(rep_i) > 0:
            diffs_r = coords[rep_j] - coords[rep_i]
            dists_r = np.sqrt((diffs_r ** 2).sum(axis=1) + 1e-10)
            clash_mask = dists_r < MIN_DIST
            if clash_mask.any():
                loss += 20.0 * ((MIN_DIST - dists_r[clash_mask]) ** 2).sum()

        # 4. Anchor to base prediction
        loss += ANCHOR_WEIGHT * np.sum((coords - base_coords) ** 2)

        return loss

    # Multi-restart SA
    refined = sa_multi_restart(
        base_coords, energy,
        n_restarts=SA_RESTARTS, n_steps=SA_STEPS,
        T_start=80.0, T_end=0.01, perturb_scale=0.35,
    )
    msa_refined_predictions[test_tid] = center_coords(refined)
    e_before = energy(base_coords)
    e_after = energy(refined)
    print(f'  {test_tid}: {base_name} → SA refined (MSA={msa_depth}, contacts={len(contact_i)}, E: {e_before:.0f}→{e_after:.0f})')

elapsed = time.time() - t0
print(f'MSA SA refinement done in {elapsed:.1f}s')
wandb.log({'msa_refine_time': elapsed})"""
)

# ── Method 5: TM-score weighted ensemble with SA refinement ──
add_code(
    """# ========================================
# Method 5: TM-score weighted ensemble + SA refinement
# ========================================
print('\\n' + '=' * 60)
print('  Method 5: TM-score Weighted Ensemble + SA Refinement')
print('=' * 60)

t0 = time.time()
ensemble_predictions = {}

ENSEMBLE_SA_STEPS = 5000
ENSEMBLE_SA_RESTARTS = 2

for test_tid in test_lengths:
    test_seq = test_seqs[test_tid]
    test_len = test_lengths[test_tid]

    # Collect all available predictions
    candidates = []
    labels = []

    if test_tid in rhofold_predictions:
        candidates.append(rhofold_predictions[test_tid])
        labels.append('RhoFold+')

    candidates.append(nw_template_predictions[test_tid])
    labels.append('NW-Top1')

    if test_tid in fragment_predictions:
        candidates.append(fragment_predictions[test_tid])
        labels.append('Fragment')

    if test_tid in msa_refined_predictions:
        candidates.append(msa_refined_predictions[test_tid])
        labels.append('MSA-Refined')

    n_methods = len(candidates)

    if n_methods == 1:
        ensemble_predictions[test_tid] = candidates[0].copy()
        print(f'  {test_tid}: 1 method ({labels[0]})')
        continue

    # Ensure all same length
    for i in range(n_methods):
        if len(candidates[i]) != test_len:
            candidates[i] = interpolate_coords(candidates[i], test_len)
        candidates[i] = center_coords(candidates[i])

    # Pairwise TM-scores
    tm_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            tm = compute_tm_score(candidates[i], candidates[j])
            tm_matrix[i, j] = tm
            tm_matrix[j, i] = tm

    # Weight by average TM-score + method prior
    method_priors = {'RhoFold+': 2.0, 'MSA-Refined': 1.5, 'Fragment': 1.0, 'NW-Top1': 0.8}
    weights = np.zeros(n_methods)
    for i in range(n_methods):
        avg_tm = np.mean([tm_matrix[i, j] for j in range(n_methods) if j != i])
        prior = method_priors.get(labels[i], 1.0)
        weights[i] = avg_tm * prior

    weights = weights / max(weights.sum(), 1e-10)

    # Kabsch-align all to highest-weight reference
    ref_idx = np.argmax(weights)
    ref = candidates[ref_idx].copy()
    avg = np.zeros((test_len, 3))
    for i in range(n_methods):
        if i == ref_idx:
            avg += weights[i] * ref
        else:
            aligned = kabsch_align(candidates[i], ref)
            avg += weights[i] * aligned

    # SA refinement of ensemble average
    pairs = nussinov_dp(test_seq) if len(test_seq) <= NUSSINOV_MAX_LEN else []

    pair_i_arr = np.array([pi for pi, pj in pairs if pi < test_len and pj < test_len])
    pair_j_arr = np.array([pj for pi, pj in pairs if pi < test_len and pj < test_len])

    def ensemble_energy(coords):
        # Sequential distance
        diffs = coords[1:] - coords[:-1]
        dists = np.sqrt((diffs ** 2).sum(axis=1) + 1e-10)
        loss = 2.0 * ((dists - 3.8) ** 2).sum()
        # Base pair contacts
        if len(pair_i_arr) > 0:
            diffs_p = coords[pair_j_arr] - coords[pair_i_arr]
            dists_p = np.sqrt((diffs_p ** 2).sum(axis=1) + 1e-10)
            loss += 0.5 * ((dists_p - 8.0) ** 2).sum()
        # Anchor to weighted average
        loss += 0.08 * np.sum((coords - avg) ** 2)
        return loss

    refined = sa_multi_restart(
        avg, ensemble_energy,
        n_restarts=ENSEMBLE_SA_RESTARTS, n_steps=ENSEMBLE_SA_STEPS,
        T_start=40.0, T_end=0.02, perturb_scale=0.25,
    )
    ensemble_predictions[test_tid] = center_coords(refined)

    weight_str = ', '.join(f'{l}={w:.3f}' for l, w in zip(labels, weights))
    print(f'  {test_tid}: {weight_str}')

print(f'Ensemble done in {time.time()-t0:.1f}s')
wandb.log({'ensemble_time': time.time()-t0})"""
)

# ── Build submission ──
add_code(
    """# ========================================
# Build submission: assign 5 methods to 5 structure slots
# ========================================
print('\\n' + '=' * 60)
print('  Building Submission')
print('=' * 60)

# Strategy: Slot 1 = strongest single method, Slot 5 = ensemble
# The evaluation picks best-of-5, so diversity matters

all_methods = {}

# Slot 1: RhoFold+ or MSA-refined fallback (our strongest prediction)
slot1 = {}
for tid in test_lengths:
    if tid in rhofold_predictions:
        slot1[tid] = rhofold_predictions[tid]
    elif tid in msa_refined_predictions:
        slot1[tid] = msa_refined_predictions[tid]
    else:
        slot1[tid] = nw_template_predictions[tid]
all_methods[1] = ('Best Single (RhoFold+/MSA/NW)', slot1)

# Slot 2: NW+2D Template Top-1
all_methods[2] = ('NW+2D Template Top-1', nw_template_predictions)

# Slot 3: Fragment Assembly
all_methods[3] = ('Fragment Assembly', fragment_predictions)

# Slot 4: MSA SA Refined
all_methods[4] = ('MSA SA Refined', msa_refined_predictions)

# Slot 5: TM Ensemble + SA
all_methods[5] = ('TM Ensemble + SA', ensemble_predictions)

submission = sample_sub.copy()

for target_id, group in submission.groupby('_target', sort=False):
    idx = group.index
    seq_len = len(group)

    for slot, (method_name, preds) in all_methods.items():
        coords = preds.get(target_id)
        if coords is None:
            coords = nw_template_predictions[target_id]
        if len(coords) != seq_len:
            coords = interpolate_coords(coords, seq_len)
        coords = center_coords(coords)
        submission.loc[idx, f'x_{slot}'] = coords[:, 0].round(3)
        submission.loc[idx, f'y_{slot}'] = coords[:, 1].round(3)
        submission.loc[idx, f'z_{slot}'] = coords[:, 2].round(3)

submission = submission.drop(columns=['_target'])

# Validate
expected_cols = sample_sub.drop(columns=['_target']).columns.tolist()
assert list(submission.columns) == expected_cols, 'Column mismatch!'
assert len(submission) == len(sample_sub), 'Row count mismatch!'
assert not submission.isnull().any().any(), 'NaN in submission!'

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f'Submission saved: {submission.shape}')
print(f'\\nSlot assignment:')
for slot, (name, _) in all_methods.items():
    print(f'  Slot {slot}: {name}')

wandb.log({
    'n_submission_rows': len(submission),
    'rhofold_coverage': len(rhofold_predictions) / len(test_lengths),
})"""
)

# ── Summary + W&B finish ──
add_code(
    """print('\\n' + '=' * 60)
print('  SUMMARY')
print('=' * 60)
print(f'RhoFold+ predictions:    {len(rhofold_predictions):>3}/{len(test_lengths)}')
print(f'NW Template predictions: {len(nw_template_predictions):>3}/{len(test_lengths)}')
print(f'Fragment assembly:       {len(fragment_predictions):>3}/{len(test_lengths)}')
print(f'MSA SA refined:          {len(msa_refined_predictions):>3}/{len(test_lengths)}')
print(f'Ensemble predictions:    {len(ensemble_predictions):>3}/{len(test_lengths)}')
print(f'\\nSubmission: {submission.shape[0]} rows x {submission.shape[1]} columns')

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

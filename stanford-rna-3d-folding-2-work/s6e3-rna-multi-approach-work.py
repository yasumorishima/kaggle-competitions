# %% [markdown]
# # Stanford RNA 3D Folding 2 — v7 Advanced Template Matching
#
# **Strategy**: Multi-scale k-mer + Needleman-Wunsch alignment + secondary structure matching
# for robust template selection. Fragment assembly and physics-based refinement for diverse predictions.
#
# **5 structure slots:**
# 1. Best template (linear interpolation)
# 2. Best template (cubic spline interpolation)
# 3. Weighted average of top-5 templates (Kabsch-aligned)
# 4. Fragment-assembled structure from multiple templates
# 5. Physics-refined version of slot 3
#
# Tools: [`kaggle-wandb-sync`](https://pypi.org/project/kaggle-wandb-sync/) / [`kaggle-notebook-deploy`](https://pypi.org/project/kaggle-notebook-deploy/)

# %%
import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_PROJECT'] = 'stanford-rna-3d-folding-2'

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')
import wandb

T_START = time.time()
print('Libraries loaded.')

# %%
# ── Load data ──
INPUT_ROOT = Path('/kaggle/input')
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
val_seq_df = pd.read_csv(DATA_DIR / 'validation_sequences.csv')
val_labels_df = pd.read_csv(DATA_DIR / 'validation_labels.csv')

SEQ_COL = 'sequence' if 'sequence' in test_df.columns else test_df.columns[1]
ID_COL = test_df.columns[0]

print(f'Test: {len(test_df)}, Train seq: {len(train_seq_df)}, Train labels: {len(train_labels_df)}')
print(f'Validation seq: {len(val_seq_df)}, Validation labels: {len(val_labels_df)}')
print(f'Sample submission: {len(sample_sub)} rows')

# Parse test sequence lengths from sample_submission
sample_sub['_target'] = sample_sub['ID'].str.rsplit('_', n=1).str[0]
test_lengths = sample_sub.groupby('_target', sort=False).size().to_dict()
print(f'Test sequences: {len(test_lengths)}')

# %%
# ── Parse training structures ──
def parse_structures(labels_df):
    """Parse structure labels into dict of {target_id: coords_array}."""
    labels_df = labels_df.copy()
    labels_df['_target'] = labels_df['ID'].str.rsplit('_', n=1).str[0]
    structures = {}
    for target_id, group in labels_df.groupby('_target', sort=False):
        coords = group[['x_1', 'y_1', 'z_1']].values.astype(np.float64)
        nan_frac = np.isnan(coords).mean()
        if nan_frac > 0.5:
            continue
        if nan_frac > 0:
            for col in range(3):
                mask = np.isnan(coords[:, col])
                if mask.any() and not mask.all():
                    valid = np.where(~mask)[0]
                    coords[mask, col] = np.interp(
                        np.where(mask)[0], valid, coords[valid, col]
                    )
        structures[target_id] = coords
    return structures

train_structures = parse_structures(train_labels_df)
val_structures = parse_structures(val_labels_df)

# Build sequence maps
def build_seq_map(seq_df):
    seq_col = 'sequence' if 'sequence' in seq_df.columns else seq_df.columns[1]
    id_col = 'target_id' if 'target_id' in seq_df.columns else seq_df.columns[0]
    return {row[id_col]: row[seq_col] for _, row in seq_df.iterrows()}

train_seq_map = build_seq_map(train_seq_df)
val_seq_map = build_seq_map(val_seq_df)
test_seq_map = {row[ID_COL]: row[SEQ_COL] for _, row in test_df.iterrows()}

# Only keep training entries with both sequence and structure
valid_train_ids = [tid for tid in train_structures if tid in train_seq_map]
valid_val_ids = [tid for tid in val_structures if tid in val_seq_map]
print(f'Training structures with sequence: {len(valid_train_ids)}')
print(f'Validation structures with sequence: {len(valid_val_ids)}')
print(f'[TIMER] Data loaded: {time.time()-T_START:.1f}s')

# %%
# ── Multi-scale k-mer similarity ──

def kmer_profile(seq, k=4):
    """Compute k-mer frequency vector."""
    kmers = defaultdict(float)
    seq = seq.upper()
    n = len(seq)
    if n < k:
        return kmers
    for i in range(n - k + 1):
        kmers[seq[i:i+k]] += 1
    total = max(sum(kmers.values()), 1)
    return {kmer: count / total for kmer, count in kmers.items()}

def kmer_cosine(prof1, prof2):
    """Cosine similarity between two k-mer profiles."""
    all_kmers = set(prof1.keys()) | set(prof2.keys())
    if not all_kmers:
        return 0.0
    v1 = np.array([prof1.get(km, 0) for km in all_kmers])
    v2 = np.array([prof2.get(km, 0) for km in all_kmers])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / max(norm, 1e-10))

def multi_kmer_similarity(seq1, seq2, profiles1, profiles2, ks=(3, 4, 5, 6)):
    """Combine k-mer cosine similarities across multiple k values.

    profiles1/profiles2: dict of {k: profile} pre-computed for each sequence.
    """
    sims = []
    for k in ks:
        p1 = profiles1.get(k)
        p2 = profiles2.get(k)
        if p1 is None:
            p1 = kmer_profile(seq1, k)
        if p2 is None:
            p2 = kmer_profile(seq2, k)
        sims.append(kmer_cosine(p1, p2))
    return float(np.mean(sims))

# %%
# ── Needleman-Wunsch global alignment ──

def needleman_wunsch_identity(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """Compute normalized alignment score using Needleman-Wunsch algorithm.

    For sequences > 1000 bases, uses banded alignment (band_width=100) to save memory/time.
    """
    s1 = seq1.upper()
    s2 = seq2.upper()
    n, m = len(s1), len(s2)

    if n == 0 or m == 0:
        return 0.0

    # For very long sequences, use banded alignment
    if n > 1000 or m > 1000:
        return _banded_alignment_identity(s1, s2, match, mismatch, gap, band_width=100)

    # Full NW DP
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if s1[i - 1] == s2[j - 1] else mismatch
            dp[i, j] = max(
                dp[i - 1, j - 1] + s,
                dp[i - 1, j] + gap,
                dp[i, j - 1] + gap,
            )

    max_possible = match * max(n, m)
    return dp[n, m] / max_possible if max_possible > 0 else 0.0


def _banded_alignment_identity(seq1, seq2, match=2, mismatch=-1, gap=-1, band_width=100):
    """Banded Needleman-Wunsch for long sequences. Only fills cells within band_width of diagonal."""
    n, m = len(seq1), len(seq2)
    # Store only the band: for each row i, columns from max(0, j_center - bw) to min(m, j_center + bw)
    # where j_center = i * m / n (the diagonal)
    bw = band_width
    NEG_INF = -10**8

    prev = np.full(m + 1, NEG_INF, dtype=np.int32)
    prev[0] = 0
    for j in range(1, min(bw + 1, m + 1)):
        prev[j] = prev[j - 1] + gap

    for i in range(1, n + 1):
        curr = np.full(m + 1, NEG_INF, dtype=np.int32)
        j_center = int(i * m / n) if n > 0 else i
        j_lo = max(1, j_center - bw)
        j_hi = min(m, j_center + bw)

        # Handle column 0 if in band
        if j_lo <= 1:
            curr[0] = prev[0] + gap

        for j in range(j_lo, j_hi + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            diag = prev[j - 1] + s if prev[j - 1] > NEG_INF else NEG_INF
            up = prev[j] + gap if prev[j] > NEG_INF else NEG_INF
            left = curr[j - 1] + gap if curr[j - 1] > NEG_INF else NEG_INF
            curr[j] = max(diag, up, left)

        prev = curr

    max_possible = match * max(n, m)
    score = prev[m]
    if score <= NEG_INF:
        # Band too narrow — fall back to simple identity
        min_len = min(n, m)
        matches = sum(1 for k in range(min_len) if seq1[k] == seq2[k])
        return matches / max(n, m, 1)
    return score / max_possible if max_possible > 0 else 0.0

# %%
# ── Nussinov secondary structure prediction ──

# Standard RNA base pairs
_VALID_PAIRS = {
    ('A', 'U'), ('U', 'A'),
    ('G', 'C'), ('C', 'G'),
    ('G', 'U'), ('U', 'G'),  # wobble pair
}

def _nussinov_dp(seq, min_loop=3):
    """Nussinov algorithm: find maximum number of base pairs.

    Returns list of (i, j) pairs.
    """
    n = len(seq)
    dp = np.zeros((n, n), dtype=np.int16)

    # Fill DP table
    for span in range(min_loop + 1, n):
        for i in range(n - span):
            j = i + span
            # Case 1: j unpaired
            best = dp[i, j - 1]
            # Case 2: j pairs with some k
            for k in range(i, j - min_loop):
                if (seq[k], seq[j]) in _VALID_PAIRS:
                    score = 1
                    left = dp[i, k - 1] if k > i else 0
                    right = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0
                    best = max(best, left + score + right)
            dp[i, j] = best

    # Traceback
    pairs = []
    _nussinov_traceback(dp, seq, 0, n - 1, min_loop, pairs)
    return pairs


def _nussinov_traceback(dp, seq, i, j, min_loop, pairs):
    """Traceback for Nussinov DP."""
    if i >= j - min_loop:
        return
    if dp[i, j] == dp[i, j - 1]:
        _nussinov_traceback(dp, seq, i, j - 1, min_loop, pairs)
    else:
        for k in range(i, j - min_loop):
            if (seq[k], seq[j]) in _VALID_PAIRS:
                left = dp[i, k - 1] if k > i else 0
                right = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0
                if left + 1 + right == dp[i, j]:
                    pairs.append((k, j))
                    if k > i:
                        _nussinov_traceback(dp, seq, i, k - 1, min_loop, pairs)
                    if k + 1 <= j - 1:
                        _nussinov_traceback(dp, seq, k + 1, j - 1, min_loop, pairs)
                    return


def predict_dot_bracket(seq, max_len=300):
    """Predict RNA secondary structure as dot-bracket string.

    Truncates to max_len to keep computation reasonable (Nussinov is O(n^3)).
    """
    seq = seq.upper()
    n = min(len(seq), max_len)
    sub_seq = seq[:n]

    try:
        pairs = _nussinov_dp(sub_seq)
    except RecursionError:
        # For very deep recursion, return all unpaired
        return '.' * n

    db = list('.' * n)
    for i, j in pairs:
        if i < n and j < n:
            db[i] = '('
            db[j] = ')'
    return ''.join(db)


def secondary_structure_similarity(db1, db2):
    """Compare dot-bracket strings by matching paired/unpaired pattern.

    Counts positions where both are paired or both are unpaired.
    """
    n = min(len(db1), len(db2))
    if n == 0:
        return 0.0
    matches = 0
    for i in range(n):
        b1_paired = db1[i] != '.'
        b2_paired = db2[i] != '.'
        if b1_paired == b2_paired:
            matches += 1
    return matches / n

# %%
# ── Combined similarity ──

def combined_similarity(query_seq, train_seq, query_profiles, train_profiles,
                        query_db, train_db):
    """Combined similarity: 0.35*multi_kmer + 0.30*NW + 0.20*SS + 0.15*length.

    This weighting emphasizes sequence content (k-mer) and alignment quality (NW)
    while incorporating structural information (SS) and length compatibility.
    """
    kmer_sim = multi_kmer_similarity(query_seq, train_seq, query_profiles, train_profiles)
    nw_sim = needleman_wunsch_identity(query_seq, train_seq)
    ss_sim = secondary_structure_similarity(query_db, train_db)
    len_sim = 1.0 - abs(len(query_seq) - len(train_seq)) / max(len(query_seq), len(train_seq), 1)
    return 0.35 * kmer_sim + 0.30 * nw_sim + 0.20 * ss_sim + 0.15 * len_sim

# %%
# ── Pre-compute all features for training sequences ──

print('Pre-computing multi-scale k-mer profiles for training sequences...')
t0 = time.time()
KMER_KS = (3, 4, 5, 6)
train_kmer_profiles = {}  # {tid: {k: profile}}
for tid in valid_train_ids:
    seq = train_seq_map[tid]
    train_kmer_profiles[tid] = {k: kmer_profile(seq, k) for k in KMER_KS}
print(f'  Multi-scale k-mer profiles ({len(KMER_KS)} scales): {len(train_kmer_profiles)} sequences in {time.time()-t0:.1f}s')

print('Pre-computing secondary structures for training sequences...')
t0 = time.time()
train_dot_brackets = {}
for tid in valid_train_ids:
    seq = train_seq_map[tid]
    train_dot_brackets[tid] = predict_dot_bracket(seq, max_len=300)
print(f'  Secondary structures: {len(train_dot_brackets)} sequences in {time.time()-t0:.1f}s')
print(f'[TIMER] Pre-computation done: {time.time()-T_START:.1f}s')

# %%
# ── Coordinate manipulation functions ──

def interpolate_coords_linear(coords, target_len):
    """Linear interpolation of coordinates to target length."""
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

def interpolate_coords_spline(coords, target_len):
    """Cubic spline interpolation for smoother coordinate fitting."""
    source_len = len(coords)
    if source_len == target_len:
        return coords.copy()
    if source_len < 4:
        return interpolate_coords_linear(coords, target_len)
    src_pos = np.linspace(0, 1, source_len)
    tgt_pos = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, 3))
    for dim in range(3):
        cs = CubicSpline(src_pos, coords[:, dim])
        result[:, dim] = cs(tgt_pos)
    return result

def center_coords(coords):
    """Center coordinates at origin."""
    return coords - coords.mean(axis=0)

def kabsch_align(mobile, target):
    """Kabsch alignment: rotate mobile to minimize RMSD to target."""
    n = min(len(mobile), len(target))
    if n < 3:
        return mobile.copy()
    m = center_coords(mobile[:n])
    t = center_coords(target[:n])
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_m = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_m @ U.T
    aligned = (R @ center_coords(mobile).T).T + target[:n].mean(axis=0)
    return aligned

def compute_tm_score(coords1, coords2):
    """Compute TM-score between two structures."""
    n = min(len(coords1), len(coords2))
    if n < 3:
        return 0.0
    c1 = center_coords(coords1[:n])
    c2 = center_coords(coords2[:n])
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    c2_aligned = (R @ c2.T).T
    d0 = 1.24 * (max(n, 19) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    di_sq = np.sum((c1 - c2_aligned) ** 2, axis=1)
    return float(np.sum(1.0 / (1.0 + di_sq / (d0 ** 2))) / n)

def helix_coords(seq_len, rise=2.81, radius=9.0, twist_deg=32.7):
    """A-form RNA helix as fallback."""
    twist_rad = np.radians(twist_deg)
    indices = np.arange(seq_len)
    x = radius * np.cos(indices * twist_rad)
    y = radius * np.sin(indices * twist_rad)
    z = indices * rise
    return np.stack([x, y, z], axis=1)

print('Coordinate utilities loaded.')

# %%
# ── Fragment assembly ──

def fragment_assembly(query_seq, target_len, query_profiles, query_db,
                      fragment_size=50, overlap=25):
    """Assemble 3D structure from fragments of different templates.

    For each overlapping fragment of the query sequence, find the best matching
    template fragment and extract coordinates. Blend overlapping regions with
    Gaussian weighting for smooth transitions.
    """
    n = len(query_seq)
    coords = np.zeros((target_len, 3))
    weights = np.zeros(target_len)
    sigma = fragment_size / 4.0
    step = fragment_size - overlap

    for start in range(0, n, step):
        end = min(start + fragment_size, n)
        fragment_seq = query_seq[start:end]
        frag_len = end - start

        if frag_len < 5:
            continue

        # Find best template for this fragment using k-mer similarity (fast)
        frag_profiles = {k: kmer_profile(fragment_seq, k) for k in KMER_KS}
        best_tid = None
        best_sim = -1.0
        for tid in valid_train_ids:
            train_seq = train_seq_map[tid]
            # Quick k-mer similarity for fragment matching
            sim = multi_kmer_similarity(fragment_seq, train_seq, frag_profiles,
                                        train_kmer_profiles[tid])
            # Penalize length mismatch less for fragments (we only need a piece)
            if sim > best_sim:
                best_sim = sim
                best_tid = tid

        if best_tid is None:
            continue

        # Extract fragment coordinates from best template
        template_coords = train_structures[best_tid]
        template_len = len(template_coords)

        # Map fragment position to template position
        frac_start = start / max(n, 1)
        frac_end = end / max(n, 1)
        t_start = int(frac_start * template_len)
        t_end = min(int(frac_end * template_len), template_len)
        if t_end <= t_start:
            t_end = min(t_start + 1, template_len)

        frag_template_coords = template_coords[t_start:t_end]
        if len(frag_template_coords) == 0:
            continue

        # Map to target coordinates
        tgt_start = int(frac_start * target_len)
        tgt_end = min(int(frac_end * target_len), target_len)
        if tgt_end <= tgt_start:
            tgt_end = min(tgt_start + 1, target_len)
        tgt_frag_len = tgt_end - tgt_start

        # Interpolate fragment to target fragment length
        if len(frag_template_coords) >= 2 and tgt_frag_len >= 2:
            frag_interp = interpolate_coords_linear(frag_template_coords, tgt_frag_len)
        elif len(frag_template_coords) >= 1 and tgt_frag_len >= 1:
            frag_interp = np.tile(frag_template_coords[0], (tgt_frag_len, 1))
        else:
            continue

        # Gaussian-weighted blending
        center = tgt_frag_len / 2.0
        for i in range(tgt_frag_len):
            pos = tgt_start + i
            if pos < target_len:
                w = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
                coords[pos] += w * frag_interp[i]
                weights[pos] += w

    # Normalize
    zero_mask = weights < 1e-10
    weights = np.maximum(weights, 1e-10)
    result = coords / weights[:, None]

    # Fill any zero-weight positions with helix fallback
    if zero_mask.any():
        fallback = helix_coords(target_len)
        result[zero_mask] = fallback[zero_mask]

    return result

# %%
# ── Physics-based refinement ──

def physics_refine(coords, seq, n_steps=300):
    """Refine coordinates with RNA-specific physical constraints.

    Applies:
    1. Sequential bond length constraint (target: 3.8 Angstrom for RNA backbone)
    2. Steric repulsion (minimum non-bonded distance: 3.0 Angstrom)
    3. Base-pairing distance constraints from Nussinov prediction
    """
    SEQ_DIST = 3.8   # sequential neighbor backbone distance
    BOND_K = 10.0    # bond spring constant
    REPULSION_R = 3.0  # minimum non-bonded distance
    REPULSION_K = 2.0  # repulsion spring constant
    PAIR_DIST = 6.5    # base-pair distance target
    PAIR_K = 3.0       # pairing spring constant

    n = len(coords)
    if n < 4:
        return center_coords(coords)

    # Get base pairs from Nussinov (reuse pre-computed or compute)
    try:
        pairs = _nussinov_dp(seq.upper()[:min(n, 200)], min_loop=3)
    except (RecursionError, Exception):
        pairs = []

    # Filter pairs to valid range
    valid_pairs = [(i, j) for i, j in pairs if i < n and j < n]

    # For steric repulsion, only check nearby non-bonded pairs (sample for speed)
    # Build list of non-bonded pairs to check (not sequential neighbors, within range)
    repulsion_pairs = []
    # Sample at most 2000 pairs for speed
    rng = np.random.RandomState(42)
    max_repulsion_checks = min(2000, n * (n - 1) // 2)
    if n > 50:
        indices = rng.choice(n, size=(max_repulsion_checks, 2), replace=True)
        for idx in range(len(indices)):
            i, j = int(indices[idx, 0]), int(indices[idx, 1])
            if abs(i - j) > 3:
                repulsion_pairs.append((min(i, j), max(i, j)))
        repulsion_pairs = list(set(repulsion_pairs))
    else:
        for i in range(n):
            for j in range(i + 4, n):
                repulsion_pairs.append((i, j))

    def energy(flat):
        c = flat.reshape(-1, 3)
        loss = 0.0

        # 1. Bond length constraint
        diffs = c[1:] - c[:-1]
        dists = np.sqrt((diffs ** 2).sum(axis=1) + 1e-10)
        loss += BOND_K * ((dists - SEQ_DIST) ** 2).sum()

        # 2. Steric repulsion (sampled pairs)
        for i, j in repulsion_pairs:
            d = np.sqrt(((c[i] - c[j]) ** 2).sum() + 1e-10)
            if d < REPULSION_R:
                loss += REPULSION_K * (REPULSION_R - d) ** 2

        # 3. Base-pairing constraints
        for i, j in valid_pairs:
            d = np.sqrt(((c[i] - c[j]) ** 2).sum() + 1e-10)
            loss += PAIR_K * (d - PAIR_DIST) ** 2

        return loss

    def energy_grad(flat):
        c = flat.reshape(-1, 3)
        grad = np.zeros_like(c)

        # 1. Bond length gradient
        diffs = c[1:] - c[:-1]
        dists = np.sqrt((diffs ** 2).sum(axis=1, keepdims=True) + 1e-10)
        force = 2.0 * BOND_K * (dists - SEQ_DIST) * diffs / dists
        grad[:-1] += force
        grad[1:] -= force

        # 2. Steric repulsion gradient
        for i, j in repulsion_pairs:
            diff = c[i] - c[j]
            d = np.sqrt((diff ** 2).sum() + 1e-10)
            if d < REPULSION_R:
                f = 2.0 * REPULSION_K * (REPULSION_R - d) * (-diff / d)
                grad[i] += f
                grad[j] -= f

        # 3. Base-pairing gradient
        for i, j in valid_pairs:
            diff = c[i] - c[j]
            d = np.sqrt((diff ** 2).sum() + 1e-10)
            f = 2.0 * PAIR_K * (d - PAIR_DIST) * (diff / d)
            grad[i] += f
            grad[j] -= f

        return grad.flatten()

    result = minimize(
        energy,
        coords.flatten(),
        jac=energy_grad,
        method='L-BFGS-B',
        options={'maxiter': n_steps, 'ftol': 1e-6},
    )

    refined = result.x.reshape(-1, 3)
    return center_coords(refined)

print('Physics refinement loaded.')

# %%
# ── Find best templates for a query sequence ──

def find_best_templates(query_seq, top_k=10):
    """Find top-K most similar training structures to query sequence.

    Two-stage approach:
    1. Pre-filter top 80 candidates by multi-scale k-mer similarity (fast)
    2. Full scoring (NW alignment + secondary structure) on candidates

    Returns list of (train_id, similarity_score, coords) sorted by similarity desc.
    """
    query_profiles = {k: kmer_profile(query_seq, k) for k in KMER_KS}
    query_db = predict_dot_bracket(query_seq, max_len=300)
    query_len = len(query_seq)

    # Stage 1: Multi-scale k-mer pre-filter — find top 80 candidates
    kmer_scores = []
    for tid in valid_train_ids:
        sim = multi_kmer_similarity(query_seq, train_seq_map[tid],
                                    query_profiles, train_kmer_profiles[tid])
        # Length penalty: prefer templates of similar length
        train_len = len(train_structures[tid])
        len_ratio = min(query_len, train_len) / max(query_len, train_len, 1)
        adjusted_sim = sim * (0.3 + 0.7 * len_ratio)
        kmer_scores.append((tid, adjusted_sim))

    kmer_scores.sort(key=lambda x: -x[1])
    candidates = kmer_scores[:80]

    # Stage 2: Full combined scoring on candidates
    results = []
    for tid, _ in candidates:
        full_sim = combined_similarity(
            query_seq, train_seq_map[tid],
            query_profiles, train_kmer_profiles[tid],
            query_db, train_dot_brackets[tid],
        )
        coords = train_structures[tid]
        results.append((tid, full_sim, coords))

    results.sort(key=lambda x: -x[1])
    return results[:top_k], query_profiles, query_db

# %%
# ── Generate 5 structure predictions ──

def predict_structures(query_seq, target_len):
    """Generate 5 diverse structure predictions for a query sequence.

    Slot 1: Best template (linear interpolation)
    Slot 2: Best template (cubic spline interpolation)
    Slot 3: Weighted average of top-5 templates (Kabsch-aligned)
    Slot 4: Fragment-assembled structure
    Slot 5: Physics-refined version of slot 3
    """
    templates, query_profiles, query_db = find_best_templates(query_seq, top_k=10)

    structures = []

    if len(templates) == 0:
        h = helix_coords(target_len)
        return [h] * 5

    # ── Slot 1: Best template with linear interpolation ──
    tid1, sim1, coords1 = templates[0]
    slot1 = interpolate_coords_linear(coords1, target_len)
    structures.append(slot1)
    print(f'    Slot 1: {tid1} linear (sim={sim1:.4f}, len={len(coords1)}->{target_len})')

    # ── Slot 2: Best template with cubic spline interpolation ──
    slot2 = interpolate_coords_spline(coords1, target_len)
    structures.append(slot2)
    print(f'    Slot 2: {tid1} spline (sim={sim1:.4f})')

    # ── Slot 3: Weighted average of top-5 templates (Kabsch-aligned) ──
    top_5 = templates[:min(5, len(templates))]
    weights = np.array([t[1] for t in top_5])
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(top_5)) / len(top_5)

    ref_coords = interpolate_coords_linear(top_5[0][2], target_len)
    avg_coords = np.zeros((target_len, 3))
    for w, (tid, sim, coords) in zip(weights, top_5):
        interp = interpolate_coords_linear(coords, target_len)
        aligned = kabsch_align(interp, ref_coords)
        avg_coords += w * aligned
    structures.append(avg_coords)
    print(f'    Slot 3: Weighted avg of {len(top_5)} templates '
          f'(sims: {", ".join(f"{t[1]:.3f}" for t in top_5)})')

    # ── Slot 4: Fragment-assembled structure ──
    frag_coords = fragment_assembly(
        query_seq, target_len, query_profiles, query_db,
        fragment_size=50, overlap=25,
    )
    structures.append(frag_coords)
    print(f'    Slot 4: Fragment assembly (frag=50, overlap=25)')

    # ── Slot 5: Physics-refined version of slot 3 ──
    refined = physics_refine(avg_coords.copy(), query_seq, n_steps=300)
    # Re-scale to match slot 3's spread (physics refinement can shrink/expand)
    s3_scale = np.std(avg_coords)
    r_scale = np.std(refined)
    if r_scale > 1e-6:
        refined = refined * (s3_scale / r_scale)
    structures.append(refined)
    print(f'    Slot 5: Physics-refined weighted avg')

    return structures[:5]

# %%
# ── Validate on validation set ──
print('\n' + '=' * 60)
print('  Validation')
print('=' * 60)

val_tm_scores = []
val_slot_tm = {s: [] for s in range(5)}
t_val_start = time.time()

for v_idx, vid in enumerate(valid_val_ids):
    val_seq = val_seq_map[vid]
    val_true = val_structures[vid]
    target_len = len(val_true)

    print(f'\n  [{v_idx+1}/{len(valid_val_ids)}] {vid}: seq_len={len(val_seq)}, struct_len={target_len}')
    preds = predict_structures(val_seq, target_len)

    # Per-slot and best-of-5 TM-score
    best_tm = 0.0
    best_slot = -1
    for s_idx, pred in enumerate(preds):
        tm = compute_tm_score(pred, val_true)
        val_slot_tm[s_idx].append(tm)
        if tm > best_tm:
            best_tm = tm
            best_slot = s_idx

    val_tm_scores.append(best_tm)
    elapsed = time.time() - t_val_start
    slot_str = ' | '.join(f'S{s+1}={np.mean(val_slot_tm[s]):.4f}' for s in range(5) if val_slot_tm[s])
    print(f'  -> best TM={best_tm:.4f} (slot {best_slot+1}), '
          f'running mean={np.mean(val_tm_scores):.4f}, {elapsed:.0f}s')
    print(f'     Per-slot means: {slot_str}')

mean_val_tm = np.mean(val_tm_scores) if val_tm_scores else 0.0
median_val_tm = np.median(val_tm_scores) if val_tm_scores else 0.0
min_val_tm = min(val_tm_scores) if val_tm_scores else 0.0
max_val_tm = max(val_tm_scores) if val_tm_scores else 0.0

print(f'\n{"=" * 60}')
print(f'  Validation Summary')
print(f'{"=" * 60}')
print(f'  Best-of-5 TM-score: mean={mean_val_tm:.4f}, median={median_val_tm:.4f}')
print(f'  Range: [{min_val_tm:.4f}, {max_val_tm:.4f}]')
for s in range(5):
    if val_slot_tm[s]:
        print(f'  Slot {s+1} mean TM: {np.mean(val_slot_tm[s]):.4f} '
              f'(median={np.median(val_slot_tm[s]):.4f})')
print(f'[TIMER] Validation: {time.time()-T_START:.1f}s')

# %%
# ── W&B logging ──
run = wandb.init(
    project='stanford-rna-3d-folding-2',
    name='template-v7-advanced',
    tags=['template-matching', 'multi-kmer', 'needleman-wunsch', 'nussinov',
          'fragment-assembly', 'physics-refine', 'v7'],
    config={
        'approach': 'advanced_template_matching_v7',
        'slot_1': 'best_template_linear',
        'slot_2': 'best_template_spline',
        'slot_3': 'weighted_avg_top5_kabsch',
        'slot_4': 'fragment_assembly',
        'slot_5': 'physics_refined_avg',
        'n_train_structures': len(valid_train_ids),
        'n_val_structures': len(valid_val_ids),
        'kmer_ks': list(KMER_KS),
        'similarity': 'multi_kmer_0.35_nw_0.30_ss_0.20_len_0.15',
        'nw_match': 2,
        'nw_mismatch': -1,
        'nw_gap': -1,
        'nussinov_max_len': 300,
        'fragment_size': 50,
        'fragment_overlap': 25,
        'physics_n_steps': 300,
        'pre_filter_top_n': 80,
    },
)

wandb.log({
    'val_tm_score_mean': mean_val_tm,
    'val_tm_score_median': median_val_tm,
    'val_tm_score_min': min_val_tm,
    'val_tm_score_max': max_val_tm,
})
for s in range(5):
    if val_slot_tm[s]:
        wandb.log({
            f'val_slot{s+1}_tm_mean': np.mean(val_slot_tm[s]),
            f'val_slot{s+1}_tm_median': np.median(val_slot_tm[s]),
        })

# %%
# ── Generate test predictions ──
print('\n' + '=' * 60)
print('  Test Predictions')
print('=' * 60)

submission = sample_sub.copy()
t_test_start = time.time()

test_ids = list(test_lengths.keys())
for t_idx, target_id in enumerate(test_ids):
    seq = test_seq_map.get(target_id, '')
    target_len = test_lengths[target_id]

    print(f'\n  [{t_idx+1}/{len(test_ids)}] {target_id}: seq_len={len(seq)}, target_len={target_len}')

    if not seq:
        print(f'    WARNING: No sequence found, using helix')
        preds = [helix_coords(target_len)] * 5
    else:
        preds = predict_structures(seq, target_len)

    # Write to submission
    mask = submission['_target'] == target_id
    idx = submission.index[mask]

    for s in range(5):
        coords = preds[s]
        if len(coords) != target_len:
            coords = interpolate_coords_linear(coords, target_len)

        submission.loc[idx, f'x_{s+1}'] = coords[:, 0].round(3)
        submission.loc[idx, f'y_{s+1}'] = coords[:, 1].round(3)
        submission.loc[idx, f'z_{s+1}'] = coords[:, 2].round(3)

    elapsed = time.time() - t_test_start
    print(f'    Elapsed: {elapsed:.0f}s')

print(f'\n[TIMER] Test predictions: {time.time()-T_START:.1f}s')

# %%
# ── Save submission ──
submission = submission.drop(columns=['_target'])
submission.to_csv('/kaggle/working/submission.csv', index=False)

print(f'\nSubmission shape: {submission.shape}')
print(f'Expected columns: {list(submission.columns[:6])} ...')
print(f'NaN check: {submission.isnull().sum().sum()} total NaN values')

# Quick sanity check
for s in range(1, 6):
    x_col = f'x_{s}'
    y_col = f'y_{s}'
    z_col = f'z_{s}'
    x_range = submission[x_col].max() - submission[x_col].min()
    y_range = submission[y_col].max() - submission[y_col].min()
    z_range = submission[z_col].max() - submission[z_col].min()
    print(f'  Slot {s}: x_range={x_range:.1f}, y_range={y_range:.1f}, z_range={z_range:.1f}')

wandb.finish()

total_time = time.time() - T_START
print(f'\n[TIMER] Total: {total_time:.1f}s ({total_time/60:.1f}min)')
print('Done. submission.csv ready.')

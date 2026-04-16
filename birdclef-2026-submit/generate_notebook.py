#!/usr/bin/env python3
"""
BirdCLEF+ 2026 — 5fold Ensemble Submission (GPU)
================================================
Loads BEATs + 5 trained SED heads, emits submission.csv.
"""

import json
from pathlib import Path


def cell(src, t="code"):
    return {
        "cell_type": t,
        "metadata": {"trusted": True},
        "source": [l + "\n" for l in src.strip().split("\n")] if isinstance(src, str) else src,
        "outputs": [],
        **({"execution_count": None} if t == "code" else {}),
    }


def generate():
    cells = []

    cells.append(cell(
        "# BirdCLEF+ 2026 — 5fold Ensemble Submission\n\n"
        "BEATs frame embeddings -> scaler+PCA(768->256) -> 5x AttentionSEDHead -> mean -> sigmoid\n\n"
        "CV: 0.9117 ± 0.0024", "markdown"))

    cells.append(cell("""
import os, sys, json, time, math, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Probe GPU — Kaggle sometimes assigns Tesla P100 (SM 6.0) which PyTorch 2.10+cu128 can't use
device = torch.device("cpu")
if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        t = torch.zeros(16, device="cuda")
        (t + 1).cpu()  # force sync to catch kernel-image errors
        device = torch.device("cuda")
        print(f"GPU OK: {gpu_name}, {gpu_mem:.1f}GB")
    except Exception as e:
        print(f"GPU probe FAIL ({type(e).__name__}: {str(e)[:120]}) — falling back to CPU")
print(f"Device: {device}, PyTorch: {torch.__version__}")

def find_file(name):
    for p in Path("/kaggle/input").rglob(name):
        return p
    return None

def find_dir(name):
    for p in Path("/kaggle/input").rglob(name):
        if p.is_dir(): return p
    return None

DATA_DIR = find_dir("birdclef-2026") or Path("/kaggle/input/birdclef-2026")
# Normalize: if rglob picked competition root itself
if not (DATA_DIR / "taxonomy.csv").exists():
    for p in Path("/kaggle/input").rglob("taxonomy.csv"):
        DATA_DIR = p.parent; break
NPZ_PATH = find_file("embeddings.npz")
HEADS_DIR = find_file("fold0.pth").parent if find_file("fold0.pth") else None
print(f"DATA_DIR: {DATA_DIR}")
print(f"NPZ_PATH: {NPZ_PATH}")
print(f"HEADS_DIR: {HEADS_DIR}")
assert (DATA_DIR / "taxonomy.csv").exists(), "taxonomy.csv not found"
assert NPZ_PATH and NPZ_PATH.exists(), "embeddings.npz not found"
assert HEADS_DIR and HEADS_DIR.exists(), "fold heads not found"
SR = 16000
CHUNK = SR * 5
N_FRAMES_OUT = 8
NUM_CLASSES = 234
PCA_DIM = 256
HIDDEN_DIM = 512
"""))

    # BEATs loader (mirrors embed kernel)
    cells.append(cell("""
# BEATs loading — robust discovery across all /kaggle/input/*
CKPT_NAME = "BEATs_iter3_plus_AS2M.pt"
beats_ckpt = None
beats_code = None
input_dir = Path("/kaggle/input")
print(f"Mounted: {sorted(d.name for d in input_dir.iterdir() if d.is_dir())}")
for d in sorted(input_dir.iterdir()):
    if not d.is_dir(): continue
    for p in d.rglob(CKPT_NAME):
        beats_ckpt = p; break
    if beats_ckpt: break
# Prefer BEATs.py from the same dataset as the checkpoint (avoids hubfor/ variant
# that returns 527-dim AudioSet logits instead of frame embeddings)
if beats_ckpt:
    for p in beats_ckpt.parent.rglob("BEATs.py"):
        beats_code = p.parent; break
    if not beats_code:
        # Walk up to dataset root and search siblings
        root = beats_ckpt
        for _ in range(6):
            root = root.parent
            if root == input_dir or root == root.parent: break
            for p in root.rglob("BEATs.py"):
                beats_code = p.parent; break
            if beats_code: break
if not beats_code:
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir(): continue
        for p in d.rglob("BEATs.py"):
            beats_code = p.parent; break
        if beats_code: break
print(f"BEATs ckpt: {beats_ckpt}")
print(f"BEATs code: {beats_code}")
if beats_ckpt is None: raise FileNotFoundError("BEATs checkpoint not found")
if beats_code is None: raise FileNotFoundError("BEATs.py not found")

sys.path.insert(0, str(beats_code))
from BEATs import BEATs, BEATsConfig

raw = torch.load(beats_ckpt, map_location="cpu", weights_only=False)
cfg_dict = raw["cfg"]
model_state = raw["model"]
beats_cfg = BEATsConfig(cfg_dict)
beats_model = BEATs(beats_cfg)
beats_model.load_state_dict(model_state, strict=False)
# CRITICAL: remove predictor so extract_features returns (B, T, 768) embeddings
# instead of AudioSet classification logits (B, 527)
if hasattr(beats_model, "predictor"):
    beats_model.predictor = None
    print("predictor removed: extract_features will return (B, T, 768) embeddings")
beats_model.eval().to(device)
EMBED_DIM = beats_cfg.encoder_embed_dim
print(f"BEATs loaded: embed_dim={EMBED_DIM}")
"""))

    # Re-fit scaler + PCA from training embeddings (reproduce training)
    cells.append(cell("""
# Re-fit scaler + PCA (SEED=42 reproduces training)
npz = np.load(NPZ_PATH)
train_frame_emb = npz["frame_embeddings"]  # [N, 8, 768]
N, T, D = train_frame_emb.shape
print(f"Train frame emb: {train_frame_emb.shape}")

n_fit = min(50000, N * T)
flat_all = train_frame_emb.reshape(-1, D)
rng = np.random.RandomState(SEED)
fit_idx = rng.choice(flat_all.shape[0], n_fit, replace=False)

scaler = StandardScaler()
scaler.fit(flat_all[fit_idx])

pca = PCA(n_components=PCA_DIM)
pca.fit(scaler.transform(flat_all[fit_idx]))
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
del train_frame_emb, flat_all
gc.collect()
"""))

    # SED head model definition
    cells.append(cell("""
class AttentionSEDHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3, use_metadata=False):
        super().__init__()
        self.use_metadata = use_metadata
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes))
        if use_metadata:
            self.hour_embed = nn.Embedding(24, 32)
            self.meta_proj = nn.Linear(32, hidden_dim)
        self.mean_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes))
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, frame_emb, hour=None):
        x = self.proj(frame_emb)
        if self.use_metadata and hour is not None:
            h = self.meta_proj(self.hour_embed(hour))
            x = x + h.unsqueeze(1)
        att = F.softmax(self.attention(x), dim=1)
        frame_pred = self.classifier(x)
        sed_logits = (att * frame_pred).sum(dim=1)
        mean_x = x.mean(dim=1)
        mean_logits = self.mean_classifier(mean_x)
        g = torch.sigmoid(self.gate)
        return g * sed_logits + (1 - g) * mean_logits

# Load 5 fold heads
heads = []
for i in range(5):
    m = AttentionSEDHead(PCA_DIM, HIDDEN_DIM, NUM_CLASSES, dropout=0.3, use_metadata=True)
    state = torch.load(HEADS_DIR / f"fold{i}.pth", map_location=device, weights_only=False)
    m.load_state_dict(state)
    m.eval().to(device)
    heads.append(m)
print(f"Loaded {len(heads)} fold heads")
"""))

    # Taxonomy + column order
    cells.append(cell("""
taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
labels = sorted(taxonomy["primary_label"].astype(str).unique().tolist())
assert len(labels) == NUM_CLASSES, f"expected {NUM_CLASSES}, got {len(labels)}"

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_cols = list(sample.columns[1:])
# Verify same columns
assert set(sub_cols) == set(labels), "class sets differ"
# Column order mapping: sample order -> training order
sample_to_train = [labels.index(c) for c in sub_cols]
print(f"Classes: {len(labels)}, sample matches training (reordered)")
"""))

    # Filename hour parser
    cells.append(cell(r"""
import re
def parse_hour(stem):
    # BC2026_Test_0001_S05_20250227_010002 -> hour=01
    m = re.search(r"_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", stem)
    if m:
        return int(m.group(4))
    return 12  # default noon
"""))

    # Inference loop
    cells.append(cell("""
test_dir = DATA_DIR / "test_soundscapes"
test_files = sorted(test_dir.glob("*.ogg"))
print(f"Test soundscapes: {len(test_files)}")

# Sanity test: if no real test files, run pipeline on 1 train file to verify inference works
SANITY_TEST_FILE = None
if len(test_files) == 0:
    train_dir = DATA_DIR / "train_audio"
    for p in train_dir.rglob("*.ogg"):
        SANITY_TEST_FILE = p; break
    if SANITY_TEST_FILE:
        print(f"SANITY mode: inferring on {SANITY_TEST_FILE.name} to verify pipeline")

rows = []
t0 = time.time()
with torch.no_grad():
    # SANITY test: run pipeline once on a train file (60s: pad or repeat if shorter)
    if SANITY_TEST_FILE is not None:
        try:
            wav, orig_sr = torchaudio.load(SANITY_TEST_FILE)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
            if orig_sr != SR:
                wav = torchaudio.functional.resample(wav, orig_sr, SR)
            target_len = SR * 60
            if wav.shape[0] < target_len:
                reps = math.ceil(target_len / wav.shape[0])
                wav = wav.repeat(reps)[:target_len]
            else:
                wav = wav[:target_len]
            chunks = [wav[i*CHUNK:(i+1)*CHUNK] for i in range(12)]
            batch = torch.stack(chunks).to(device)
            pad_mask = torch.zeros(batch.shape, dtype=torch.bool, device=device)
            raw_out = beats_model.extract_features(batch, padding_mask=pad_mask)
            print(f"raw_out type={type(raw_out).__name__}, len={len(raw_out) if hasattr(raw_out,'__len__') else 'n/a'}")
            if isinstance(raw_out, tuple):
                for j, o in enumerate(raw_out):
                    print(f"  [{j}] shape={getattr(o,'shape','n/a')}")
            feats = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            print(f"feats.shape = {feats.shape}")
            if feats.ndim == 2:
                # [B*N_FRAMES, 768] flattened — reshape assuming same frames per batch
                total_elems = feats.shape[0]
                assert total_elems % batch.shape[0] == 0, f"cannot infer N_FRAMES: {total_elems} / {batch.shape[0]}"
                n_frames = total_elems // batch.shape[0]
                feats = feats.view(batch.shape[0], n_frames, feats.shape[-1])
                print(f"reshaped feats: {feats.shape}")
            indices = np.linspace(0, feats.shape[1] - 1, N_FRAMES_OUT, dtype=int)
            feats = feats[:, indices, :].cpu().numpy()
            flat = feats.reshape(-1, EMBED_DIM)
            flat = scaler.transform(flat)
            flat = pca.transform(flat)
            frame_emb = flat.reshape(12, N_FRAMES_OUT, PCA_DIM).astype(np.float32)
            frame_emb_t = torch.from_numpy(frame_emb).to(device)
            hour_t = torch.full((12,), 12, dtype=torch.long, device=device)
            probs_sum = torch.zeros(12, NUM_CLASSES, device=device)
            for m in heads:
                logits = m(frame_emb_t, hour=hour_t)
                probs_sum += torch.sigmoid(logits)
            probs = (probs_sum / len(heads)).cpu().numpy()
            print(f"SANITY OK: probs shape={probs.shape}, min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            print(f"SANITY top-3 idx of chunk 0: {probs[0].argsort()[-3:][::-1].tolist()}")
        except Exception as e:
            print(f"SANITY FAIL: {type(e).__name__}: {e}")
            raise  # kernel should fail so we know before LB submit

    for fi, fp in enumerate(test_files):
        stem = fp.stem
        hour = parse_hour(stem)
        try:
            wav, orig_sr = torchaudio.load(fp)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
            if orig_sr != SR:
                wav = torchaudio.functional.resample(wav, orig_sr, SR)
        except Exception as e:
            print(f"Load error {fp.name}: {e}")
            continue

        total_samples = wav.shape[0]
        total_chunks = math.ceil(total_samples / CHUNK)  # typically 12 for 60s
        # Build batch of all chunks in this file
        chunks = []
        for ci in range(total_chunks):
            s = ci * CHUNK
            e = s + CHUNK
            c = wav[s:e]
            if c.shape[0] < CHUNK:
                c = F.pad(c, (0, CHUNK - c.shape[0]))
            chunks.append(c)
        batch = torch.stack(chunks).to(device)
        pad_mask = torch.zeros(batch.shape, dtype=torch.bool, device=device)

        feats = beats_model.extract_features(batch, padding_mask=pad_mask)[0]
        # feats: [C, N_FRAMES_RAW, 768]
        indices = np.linspace(0, feats.shape[1] - 1, N_FRAMES_OUT, dtype=int)
        feats = feats[:, indices, :].cpu().numpy()  # [C, 8, 768]

        # scaler + PCA
        flat = feats.reshape(-1, EMBED_DIM)
        flat = scaler.transform(flat)
        flat = pca.transform(flat)
        frame_emb = flat.reshape(total_chunks, N_FRAMES_OUT, PCA_DIM).astype(np.float32)
        frame_emb_t = torch.from_numpy(frame_emb).to(device)
        hour_t = torch.full((total_chunks,), hour, dtype=torch.long, device=device)

        # Ensemble
        probs_sum = torch.zeros(total_chunks, NUM_CLASSES, device=device)
        for m in heads:
            logits = m(frame_emb_t, hour=hour_t)
            probs_sum += torch.sigmoid(logits)
        probs = (probs_sum / len(heads)).cpu().numpy()

        for ci in range(total_chunks):
            end_sec = (ci + 1) * 5
            row_id = f"{stem}_{end_sec}"
            rows.append((row_id, probs[ci]))

        if fi % 10 == 0:
            el = time.time() - t0
            print(f"[{fi+1}/{len(test_files)}] {el:.0f}s, {fi+1 and el/(fi+1):.2f}s/file")

print(f"Total inference: {time.time()-t0:.0f}s, {len(rows)} rows")
"""))

    # Build submission DataFrame with correct column order
    cells.append(cell("""
if len(rows) == 0:
    # Hidden test not revealed (private run) — emit sample_submission as fallback
    # At LB time Kaggle replaces test_soundscapes with real files; the kernel re-runs
    print("No test files — emitting sample_submission.csv as fallback")
    sample.to_csv("submission.csv", index=False)
else:
    row_ids = [r[0] for r in rows]
    probs_arr = np.stack([r[1] for r in rows])
    probs_sample_order = probs_arr[:, sample_to_train]
    sub = pd.DataFrame(probs_sample_order, columns=sub_cols)
    sub.insert(0, "row_id", row_ids)
    # Ensure all sample row_ids covered; fill missing with uniform
    pred_ids = set(row_ids)
    missing = sample[~sample["row_id"].isin(pred_ids)]
    if len(missing) > 0:
        print(f"Filling {len(missing)} missing rows with sample defaults")
        sub = pd.concat([sub, missing], ignore_index=True)
    # Reorder to sample order
    sub = sub.set_index("row_id").reindex(sample["row_id"]).reset_index()
    sub.to_csv("submission.csv", index=False)
print(f"submission.csv written")
out = pd.read_csv("submission.csv")
print(f"Shape: {out.shape}")
print(out.head(2))
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "birdclef-2026-beats-sed-submit.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Generated: {out}")


if __name__ == "__main__":
    generate()

#!/usr/bin/env python3
"""
BirdCLEF+ 2026 — BEATs Embedding Extraction (CPU only, GPU消費ゼロ)
====================================================================
35,549件の音声 → BEATs encoder → frame-level embedding (8frames × 768dim)
出力を Kaggle Dataset として保存 → 学習Notebookで参照
"""

import json
from pathlib import Path


def make_cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {"trusted": True},
        "source": source.strip().split("\n") if isinstance(source, str) else source,
        "outputs": [],
        **({"execution_count": None} if cell_type == "code" else {}),
    }


def generate():
    cells = []

    # ── Title ──
    cells.append(make_cell(
        "# BirdCLEF+ 2026 — BEATs Embedding Extraction\n\n"
        "**CPU only** — GPU消費ゼロ\n\n"
        "BEATs iter3+ AS2M encoder → 768-dim frame embeddings\n"
        "- 全35,549件の音声を処理\n"
        "- 8フレーム × 768次元に間引き（メモリ効率）\n"
        "- mean-pooled 768次元も同時保存\n"
        "- 出力: `/kaggle/working/embeddings.npz` (~900MB)",
        cell_type="markdown",
    ))

    # ── Setup ──
    cells.append(make_cell("""
import os, sys, gc, time, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
warnings.filterwarnings("ignore")
print(f"PyTorch: {torch.__version__}")
print(f"CPU cores: {os.cpu_count()}")
"""))

    # ── Load data ──
    cells.append(make_cell("""
import glob as _glob
_slug = "birdclef-2026"
_matches = _glob.glob(f"/kaggle/input/**/{_slug}", recursive=True)
DATA_DIR = Path(_matches[0]) if _matches else Path(f"/kaggle/input/{_slug}")
AUDIO_DIR = DATA_DIR / "train_audio"

train_df = pd.read_csv(DATA_DIR / "train.csv")
taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
labels = sorted(taxonomy["primary_label"].unique().tolist())
print(f"Samples: {len(train_df)}, Classes: {len(labels)}")
print(f"Audio dir exists: {AUDIO_DIR.exists()}")
"""))

    # ── Load BEATs ──
    cells.append(make_cell("""
# BEATs model loading
# Priority: yasunorim/beats-pretrained (checkpoint + code)
# Fallback: hubfor/microsoft-beats-model (code only — would need checkpoint elsewhere)
BEATS_CKPT = None
BEATS_CODE = None

# Check yasunorim dataset (has both checkpoint + code)
p1 = Path("/kaggle/input/beats-pretrained")
if (p1 / "BEATs_iter3_plus_AS2M.pt").exists():
    BEATS_CKPT = p1 / "BEATs_iter3_plus_AS2M.pt"
    BEATS_CODE = p1
    print(f"Checkpoint: {BEATS_CKPT} ({BEATS_CKPT.stat().st_size/1024/1024:.0f}MB)")

# Check hubfor dataset (code only)
p2 = Path("/kaggle/input/microsoft-beats-model")
if BEATS_CODE is None and (p2 / "BEATs.py").exists():
    BEATS_CODE = p2
    print(f"Code from hubfor dataset")

if BEATS_CKPT is None:
    raise RuntimeError(
        "BEATs checkpoint not found! "
        "Add 'yasunorim/beats-pretrained' to dataset_sources."
    )

sys.path.insert(0, str(BEATS_CODE))
from BEATs import BEATs, BEATsConfig

# Load and verify checkpoint
checkpoint = torch.load(BEATS_CKPT, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Handle both pre-trained and fine-tuned checkpoint formats
if "model" in checkpoint:
    model_state = checkpoint["model"]
elif "state_dict" in checkpoint:
    model_state = checkpoint["state_dict"]
else:
    raise RuntimeError(f"Unknown checkpoint format: {list(checkpoint.keys())}")

cfg_dict = checkpoint.get("cfg", checkpoint.get("config", {}))
beats_cfg = BEATsConfig(cfg_dict)
beats_model = BEATs(beats_cfg)

# Load with strict=False to handle fine-tuned models with extra head weights
missing, unexpected = beats_model.load_state_dict(model_state, strict=False)
if unexpected:
    print(f"Unexpected keys (ignored, likely classification head): {len(unexpected)}")
if missing:
    print(f"Missing keys: {missing}")

beats_model.eval()
EMBED_DIM = beats_cfg.encoder_embed_dim
print(f"BEATs loaded: embed_dim={EMBED_DIM}, layers={beats_cfg.encoder_layers}")
"""))

    # ── Test single file ──
    cells.append(make_cell("""
# Verify embedding extraction on a single file
test_file = AUDIO_DIR / train_df.iloc[0]["filename"]
wav, sr = torchaudio.load(test_file)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
wav = wav.squeeze(0)

SR = 16000
CHUNK = SR * 5  # 5 seconds

if sr != SR:
    wav = torchaudio.functional.resample(wav, sr, SR)
if wav.shape[0] > CHUNK:
    wav = wav[:CHUNK]
elif wav.shape[0] < CHUNK:
    wav = F.pad(wav, (0, CHUNK - wav.shape[0]))

with torch.no_grad():
    padding_mask = torch.zeros(1, wav.shape[0], dtype=torch.bool)
    features = beats_model.extract_features(wav.unsqueeze(0), padding_mask=padding_mask)[0]

N_FRAMES_RAW = features.shape[1]
print(f"Test file: {test_file.name}")
print(f"Input: {wav.shape[0]} samples ({wav.shape[0]/SR:.1f}s)")
print(f"Output: {features.shape} = [1, {N_FRAMES_RAW} frames, {EMBED_DIM} dim]")

# Determine subsampling: target 8 frames per clip
TARGET_FRAMES = 8
SUBSAMPLE_STRIDE = max(1, N_FRAMES_RAW // TARGET_FRAMES)
N_FRAMES_OUT = min(TARGET_FRAMES, N_FRAMES_RAW)
print(f"Subsample: stride={SUBSAMPLE_STRIDE}, output_frames={N_FRAMES_OUT}")
print(f"Memory estimate: {len(train_df)} × {N_FRAMES_OUT} × {EMBED_DIM} × 4 bytes "
      f"= {len(train_df) * N_FRAMES_OUT * EMBED_DIM * 4 / 1024/1024:.0f} MB")
"""))

    # ── Extract all embeddings ──
    cells.append(make_cell("""
BATCH_SIZE = 16  # conservative for CPU memory
filenames = train_df["filename"].tolist()
n_batches = math.ceil(len(filenames) / BATCH_SIZE)

all_mean_emb = np.zeros((len(filenames), EMBED_DIM), dtype=np.float32)
all_frame_emb = np.zeros((len(filenames), N_FRAMES_OUT, EMBED_DIM), dtype=np.float32)
error_indices = []

t0 = time.time()
with torch.no_grad():
    for bi in range(n_batches):
        start = bi * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(filenames))
        batch_files = filenames[start:end]
        actual_batch = end - start

        wavs = []
        for fname in batch_files:
            try:
                wav, orig_sr = torchaudio.load(AUDIO_DIR / fname)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0)
                if orig_sr != SR:
                    wav = torchaudio.functional.resample(wav, orig_sr, SR)
            except Exception as e:
                wav = torch.zeros(CHUNK)
                error_indices.append(start + len(wavs))

            # Center crop / pad to exactly 5s
            if wav.shape[0] > CHUNK:
                off = (wav.shape[0] - CHUNK) // 2
                wav = wav[off:off + CHUNK]
            elif wav.shape[0] < CHUNK:
                pad = CHUNK - wav.shape[0]
                wav = F.pad(wav, (pad // 2, pad - pad // 2))
            wavs.append(wav)

        wavs_tensor = torch.stack(wavs)
        padding_mask = torch.zeros(wavs_tensor.shape, dtype=torch.bool)
        features = beats_model.extract_features(wavs_tensor, padding_mask=padding_mask)[0]
        # features: [B, N_FRAMES_RAW, 768]

        # Mean pooling
        all_mean_emb[start:end] = features.mean(dim=1).numpy()

        # Subsample frames: take evenly spaced frames
        indices = np.linspace(0, features.shape[1] - 1, N_FRAMES_OUT, dtype=int)
        all_frame_emb[start:end] = features[:, indices, :].numpy()

        if bi % 200 == 0:
            elapsed = time.time() - t0
            speed = (end) / elapsed if elapsed > 0 else 0
            eta = (len(filenames) - end) / speed if speed > 0 else 0
            print(f"Batch {bi}/{n_batches} | {end}/{len(filenames)} files | "
                  f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s | "
                  f"{speed:.1f} files/s")

        # Periodic memory check
        if bi % 500 == 0:
            gc.collect()

total_time = time.time() - t0
print(f"\\nExtraction complete: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"Errors: {len(error_indices)} files")
print(f"Mean embeddings: {all_mean_emb.shape}")
print(f"Frame embeddings: {all_frame_emb.shape}")
"""))

    # ── Save ──
    cells.append(make_cell("""
# Save embeddings
output_path = "/kaggle/working/embeddings.npz"
np.savez_compressed(
    output_path,
    mean_embeddings=all_mean_emb,
    frame_embeddings=all_frame_emb,
    labels=np.array(labels),
    n_frames_raw=np.array([N_FRAMES_RAW]),
    subsample_stride=np.array([SUBSAMPLE_STRIDE]),
    error_indices=np.array(error_indices),
)
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"Saved: {output_path} ({size_mb:.1f} MB)")

# Sanity check: reload and verify
data = np.load(output_path)
print(f"Keys: {list(data.keys())}")
print(f"Mean emb: {data['mean_embeddings'].shape}, dtype={data['mean_embeddings'].dtype}")
print(f"Frame emb: {data['frame_embeddings'].shape}, dtype={data['frame_embeddings'].dtype}")

# Quick stats
print(f"\\nEmbedding stats:")
print(f"  Mean norm: {np.linalg.norm(data['mean_embeddings'], axis=1).mean():.2f}")
print(f"  Std across dims: {data['mean_embeddings'].std(axis=0).mean():.4f}")
print(f"  Any NaN: {np.isnan(data['mean_embeddings']).any()}")
"""))

    # ── Also save train.csv with indices for reproducibility ──
    cells.append(make_cell("""
# Save metadata for training notebook
train_df.to_csv("/kaggle/working/train_meta.csv", index=False)
taxonomy.to_csv("/kaggle/working/taxonomy.csv", index=False)
print("Metadata saved")
print(f"\\nTotal output files:")
for f in Path("/kaggle/working").iterdir():
    if f.is_file():
        print(f"  {f.name}: {f.stat().st_size/1024/1024:.1f} MB")
"""))

    # Build notebook
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4, "nbformat_minor": 4, "cells": cells,
    }
    for cell in notebook["cells"]:
        src = cell["source"]
        for i in range(len(src) - 1):
            if not src[i].endswith("\n"):
                src[i] = src[i] + "\n"

    output_path = Path(__file__).parent / "birdclef-2026-beats-embed.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    generate()

# %% [markdown]
# # BirdCLEF+ 2026 — eca_nfnet_l0 Submit (CPU)
#
# Inference kernel for eca_nfnet_l0 trained checkpoint.
# - Loads fold0 best checkpoint from mounted dataset
# - Mel spectrogram → model forward → sigmoid → submission.csv
# - CPU only, 90 min limit

# %%
import numpy as np
import pandas as pd
import os, sys, time, glob
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as TV
import timm
import librosa

START = time.time()
print(f"PyTorch {torch.__version__}")

# %%
# ══════════════════════════════════════════════════════════════
# LOAD CONFIG FROM CHECKPOINT
# ══════════════════════════════════════════════════════════════
WEIGHTS_DIR = Path("/kaggle/input/datasets/yasunorim/birdclef-2026-nfnet-weights")
DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")

# Find all fold checkpoints
ckpt_paths = sorted(WEIGHTS_DIR.glob("fold*_best.pt"))
if not ckpt_paths:
    # Fallback: search recursively
    ckpt_paths = sorted(WEIGHTS_DIR.rglob("fold*_best.pt"))
assert len(ckpt_paths) > 0, f"No checkpoints found in {WEIGHTS_DIR}"
print(f"Found {len(ckpt_paths)} checkpoint(s): {[p.name for p in ckpt_paths]}")

# Load config from first checkpoint
ckpt0 = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
cfg_dict = ckpt0["config"]
print(f"Config from checkpoint: {cfg_dict}")

# %%
# ══════════════════════════════════════════════════════════════
# CONFIG (from checkpoint)
# ══════════════════════════════════════════════════════════════
@dataclass
class Config:
    sr: int = cfg_dict.get("sr", 32000)
    chunk_duration: float = cfg_dict.get("chunk_duration", 5.0)
    n_mels: int = cfg_dict.get("n_mels", 128)
    n_fft: int = cfg_dict.get("n_fft", 2048)
    hop_length: int = cfg_dict.get("hop_length", 512)
    fmin: int = cfg_dict.get("fmin", 20)
    fmax: int = cfg_dict.get("fmax", 16000)
    top_db: float = 80.0
    power: float = 2.0
    target_size: tuple = tuple(cfg_dict.get("target_size", [224, 224]))
    in_channels: int = cfg_dict.get("in_channels", 3)
    backbone: str = cfg_dict.get("backbone", "eca_nfnet_l0")
    num_classes: int = cfg_dict.get("num_classes", 234)
    dropout: float = cfg_dict.get("dropout", 0.2)
    drop_path_rate: float = cfg_dict.get("drop_path_rate", 0.2)

    @property
    def chunk_samples(self) -> int:
        return int(self.sr * self.chunk_duration)

cfg = Config()
print(f"backbone={cfg.backbone}, num_classes={cfg.num_classes}, sr={cfg.sr}")

# %%
# ══════════════════════════════════════════════════════════════
# SPECIES
# ══════════════════════════════════════════════════════════════
sub_df = pd.read_csv(DATA_ROOT / "sample_submission.csv", nrows=1)
SPECIES = list(sub_df.columns[1:])
print(f"Species: {len(SPECIES)}")

# %%
# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION (must match training)
# ══════════════════════════════════════════════════════════════
class GEMPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class AttSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        logit = (att * cls).sum(dim=-1)
        return logit


class BirdModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=False,
            in_chans=cfg.in_channels, features_only=False,
            global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.gem = GEMPool(p_init=3.0)
        self.head = AttSEDHead(feat_dim, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.gem(feat)
        logit = self.head(pooled)
        return logit

# %%
# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
models = []
for ckpt_path in ckpt_paths:
    print(f"Loading: {ckpt_path.name}")
    model = BirdModel(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_auc = ckpt.get("val_auc", "?")
    print(f"  epoch={epoch}, val_auc={val_auc}")
    models.append(model)
print(f"\nLoaded {len(models)} model(s) for ensemble")

# %%
# ══════════════════════════════════════════════════════════════
# MEL TRANSFORM
# ══════════════════════════════════════════════════════════════
class MelTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=cfg.power,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.resize = TV.Resize(cfg.target_size, antialias=True)

    @torch.no_grad()
    def forward(self, waveforms):
        mel = self.db(self.mel(waveforms))
        mel = self.resize(mel)
        B = mel.shape[0]
        mel_flat = mel.reshape(B, -1)
        mel_min = mel_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = mel_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)

mel_transform = MelTransform(cfg)
mel_transform.eval()

# %%
# ══════════════════════════════════════════════════════════════
# FIND TEST FILES
# ══════════════════════════════════════════════════════════════
TEST_DIR = DATA_ROOT / "test_soundscapes"
TRAIN_SC_DIR = DATA_ROOT / "train_soundscapes"

test_files = sorted(TEST_DIR.glob("*.ogg"))
if len(test_files) == 0:
    print("No test soundscapes found, using train_soundscapes as fallback")
    test_files = sorted(TRAIN_SC_DIR.glob("*.ogg"))[:8]
print(f"Test files: {len(test_files)}")

# %%
# ══════════════════════════════════════════════════════════════
# AUDIO LOADING
# ══════════════════════════════════════════════════════════════
def load_soundscape(path):
    stem = Path(path).stem
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
    return y, stem

print("Loading audio files...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    audio_results = list(executor.map(load_soundscape, test_files))
print(f"  Loaded {len(audio_results)} files in {time.time()-t0:.1f}s")

# %%
# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════
CHUNK = cfg.chunk_samples
BATCH_SIZE = 32

all_row_ids = []
all_preds = []

print("Running inference...")
t0 = time.time()

for audio, stem in audio_results:
    n_chunks = max(1, len(audio) // CHUNK)
    padded_len = n_chunks * CHUNK
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))
    else:
        audio = audio[:padded_len]

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    chunks_tensor = torch.from_numpy(audio.reshape(n_chunks, CHUNK)).float()

    # Process in batches
    file_preds = []
    for start in range(0, n_chunks, BATCH_SIZE):
        batch = chunks_tensor[start:start + BATCH_SIZE]
        with torch.no_grad():
            mel = mel_transform(batch)
            # Ensemble: average logits across models
            logits_sum = None
            for model in models:
                logits = model(mel)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum = logits_sum + logits
            avg_logits = logits_sum / len(models)
            probs = torch.sigmoid(avg_logits).numpy()
        file_preds.append(probs)

    probs_all = np.concatenate(file_preds, axis=0)

    for i in range(n_chunks):
        end_sec = (i + 1) * int(cfg.chunk_duration)
        all_row_ids.append(f"{stem}_{end_sec}")
        all_preds.append(probs_all[i])

elapsed = time.time() - t0
print(f"  Inference done: {len(all_row_ids)} predictions in {elapsed:.1f}s")

# %%
# ══════════════════════════════════════════════════════════════
# BUILD SUBMISSION
# ══════════════════════════════════════════════════════════════
preds_array = np.stack(all_preds)
submission = pd.DataFrame(preds_array, columns=SPECIES)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])

missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids -- filling with zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in SPECIES:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)

extra = our_ids - expected_ids
if extra:
    print(f"Dropping {len(extra)} extra row_ids")
    submission = submission[submission["row_id"].isin(expected_ids)]

submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

total_time = time.time() - START
print(f"\nSubmission saved: {submission.shape}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(submission.head())

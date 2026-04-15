#!/usr/bin/env python3
"""
BirdCLEF+ 2026 — EXP001 SED Head Training (GPU, ~5-10min)
==========================================================
Loads precomputed BEATs embeddings from CPU notebook output.
Trains Attention SED head on frame-level embeddings.

Input: yasunorim/birdclef-2026-beats-embed (kernel output)
  - embeddings.npz: mean_embeddings [N,768] + frame_embeddings [N,8,768]
  - train_meta.csv, taxonomy.csv
"""

import json
import yaml
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
    config_path = Path(__file__).parent / "EXP" / "EXP001" / "config" / "child-exp000.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    config_str = json.dumps(cfg, indent=2, ensure_ascii=False)

    cells = []

    # ── Title ──
    cells.append(make_cell(
        "# BirdCLEF+ 2026 — Attention SED Head Training\n\n"
        "**GPU使用: ~5-10分のみ**\n\n"
        "BEATs frame embeddings (CPU notebookで事前計算済み) → Attention SED head\n\n"
        "### Differentiators\n"
        "1. BEATs embeddings (AudioSet 2M pretrained) — 昆虫/両生類にも強い\n"
        "2. Attention pooling — 鳴いているフレームに自動集中\n"
        "3. Focal loss — 234種の不均衡対策\n"
        "4. Metadata injection (hour of day)\n"
        "5. Embedding augmentation (mixup, dropout, noise)",
        cell_type="markdown",
    ))

    # ── Setup ──
    cells.append(make_cell("""
import os, sys, gc, ast, time, math, random, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
import wandb  # pre-installed in Kaggle image; internet disabled blocks pip install

warnings.filterwarnings("ignore")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
"""))

    # ── Load embeddings ──
    cells.append(make_cell("""
# Find embedding notebook output
import glob as _glob
EMB_MATCHES = _glob.glob("/kaggle/input/**/embeddings.npz", recursive=True)
if not EMB_MATCHES:
    raise RuntimeError(
        "Embeddings not found! Add 'yasunorim/birdclef-2026-beats-embed' to kernel_sources.")
EMB_PATH = EMB_MATCHES[0]
EMB_DIR = Path(EMB_PATH).parent

print(f"Loading embeddings from: {EMB_PATH}")
data = np.load(EMB_PATH)
mean_emb = data["mean_embeddings"]      # [N, 768]
frame_emb = data["frame_embeddings"]    # [N, 8, 768]
stored_labels = data["labels"].tolist()
n_frames_raw = int(data["n_frames_raw"][0])
error_indices = data["error_indices"]
print(f"Mean embeddings: {mean_emb.shape}")
print(f"Frame embeddings: {frame_emb.shape}")
print(f"Original frames per clip: {n_frames_raw}")
print(f"Error files: {len(error_indices)}")

# Load metadata
meta_path = EMB_DIR / "train_meta.csv"
if meta_path.exists():
    train_df = pd.read_csv(meta_path)
else:
    # Fallback to competition data
    _slug = "birdclef-2026"
    _m = _glob.glob(f"/kaggle/input/**/{_slug}", recursive=True)
    DATA_DIR = Path(_m[0]) if _m else Path(f"/kaggle/input/{_slug}")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

tax_path = EMB_DIR / "taxonomy.csv"
if tax_path.exists():
    taxonomy = pd.read_csv(tax_path)
else:
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")

labels = sorted(taxonomy["primary_label"].unique().tolist())
label2idx = {l: i for i, l in enumerate(labels)}
NUM_CLASSES = len(labels)
print(f"Samples: {len(train_df)}, Classes: {NUM_CLASSES}")

# Verify alignment
assert len(train_df) == mean_emb.shape[0], \\
    f"Mismatch: train_df={len(train_df)}, embeddings={mean_emb.shape[0]}"
"""))

    # ── Config ──
    cells.append(make_cell(f"""
CONFIG = json.loads('''{config_str}''')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
"""))

    # ── PCA + Targets ──
    cells.append(make_cell("""
# PCA on frame embeddings
N, T, D = frame_emb.shape
INPUT_DIM = D
mcfg = CONFIG["model"]

if mcfg.get("use_pca", False):
    pca_dim = mcfg["pca_dim"]
    print(f"PCA: {D} -> {pca_dim}")

    # Fit on random subset to save memory
    n_fit = min(50000, N * T)
    flat_all = frame_emb.reshape(-1, D)
    rng = np.random.RandomState(SEED)
    fit_idx = rng.choice(flat_all.shape[0], n_fit, replace=False)

    scaler = StandardScaler()
    scaler.fit(flat_all[fit_idx])
    flat_scaled = scaler.transform(flat_all)

    pca = PCA(n_components=pca_dim)
    pca.fit(flat_scaled[fit_idx])
    flat_pca = pca.transform(flat_scaled)

    frame_emb = flat_pca.reshape(N, T, pca_dim).astype(np.float32)
    INPUT_DIM = pca_dim
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    del flat_all, flat_scaled, flat_pca
    gc.collect()

# Prepare targets (multi-label)
targets = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
for i, row in train_df.iterrows():
    if row["primary_label"] in label2idx:
        targets[i, label2idx[row["primary_label"]]] = 1.0
    sec = row.get("secondary_labels", "[]")
    if isinstance(sec, str):
        try:
            sec = ast.literal_eval(sec)
        except (ValueError, SyntaxError):
            sec = []
    for sl in sec:
        if sl in label2idx:
            targets[i, label2idx[sl]] = 1.0

hours = np.full(len(train_df), 12, dtype=np.int64)  # TODO: extract from metadata

pos_per_class = targets.sum(axis=0)
print(f"Targets: {targets.shape}, mean positive rate: {targets.mean():.5f}")
print(f"Classes with >0 positives: {(pos_per_class > 0).sum()}/{NUM_CLASSES}")
print(f"Min/Max positives per class: {pos_per_class[pos_per_class>0].min():.0f} / {pos_per_class.max():.0f}")
"""))

    # ── Model ──
    cells.append(make_cell("""
class AttentionSEDHead(nn.Module):
    \"\"\"Attention-weighted Sound Event Detection head.\"\"\"

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3,
                 use_metadata=False):
        super().__init__()
        self.use_metadata = use_metadata

        # Project embeddings to hidden space
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-class attention: which frames matter for each species
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

        # Frame-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Metadata
        if use_metadata:
            self.hour_embed = nn.Embedding(24, 32)
            self.meta_proj = nn.Linear(32, hidden_dim)

        # Also use mean-pooled path for stability
        self.mean_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))  # learnable blend

    def forward(self, frame_emb, hour=None):
        x = self.proj(frame_emb)  # [B, T, hidden]

        if self.use_metadata and hour is not None:
            h = self.meta_proj(self.hour_embed(hour))  # [B, hidden]
            x = x + h.unsqueeze(1)

        # Attention path (SED)
        att = F.softmax(self.attention(x), dim=1)  # [B, T, C]
        frame_pred = self.classifier(x)             # [B, T, C]
        sed_logits = (att * frame_pred).sum(dim=1)  # [B, C]

        # Mean-pooled path (clip-level)
        mean_x = x.mean(dim=1)  # [B, hidden]
        mean_logits = self.mean_classifier(mean_x)  # [B, C]

        # Gated blend
        g = torch.sigmoid(self.gate)
        logits = g * sed_logits + (1 - g) * mean_logits

        return logits


class EmbeddingDataset(Dataset):
    def __init__(self, frame_emb, targets, hours, is_train=True, aug_cfg=None):
        self.frame_emb = torch.from_numpy(frame_emb).float()
        self.targets = torch.from_numpy(targets).float()
        self.hours = torch.from_numpy(hours).long()
        self.is_train = is_train
        self.aug_cfg = aug_cfg or {}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        emb = self.frame_emb[idx].clone()
        if self.is_train:
            # Embedding dropout (per-dimension)
            dp = self.aug_cfg.get("embedding_dropout", 0)
            if dp > 0 and random.random() < 0.5:
                mask = torch.bernoulli(torch.full((emb.shape[-1],), 1.0 - dp))
                emb = emb * mask / (1.0 - dp + 1e-8)
            # Gaussian noise
            ns = self.aug_cfg.get("gaussian_noise_std", 0)
            if ns > 0:
                emb = emb + torch.randn_like(emb) * ns
        return emb, self.targets[idx], self.hours[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()


def mixup_data(emb, tgt, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(emb.size(0), device=emb.device)
    return lam * emb + (1-lam) * emb[idx], lam * tgt + (1-lam) * tgt[idx]

print(f"Model input_dim={INPUT_DIM}, hidden_dim={mcfg['hidden_dim']}, classes={NUM_CLASSES}")
"""))

    # ── Training setup ──
    cells.append(make_cell("""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold = CONFIG["cv"]["fold"]
tcfg = CONFIG["training"]
aug_cfg = CONFIG.get("augmentation", {})

# CV split
skf = StratifiedKFold(n_splits=CONFIG["cv"]["n_folds"], shuffle=True, random_state=SEED)
train_idx, val_idx = list(skf.split(train_df, train_df["primary_label"]))[fold]

train_ds = EmbeddingDataset(frame_emb[train_idx], targets[train_idx],
                            hours[train_idx], is_train=True, aug_cfg=aug_cfg)
val_ds = EmbeddingDataset(frame_emb[val_idx], targets[val_idx],
                          hours[val_idx], is_train=False)

train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                          shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"]*2,
                        shuffle=False, num_workers=2, pin_memory=True)

model = AttentionSEDHead(
    INPUT_DIM, mcfg["hidden_dim"], NUM_CLASSES,
    mcfg["dropout"], mcfg.get("use_metadata", False)
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"SED Head: {n_params:,} params")
print(f"Fold {fold}: train={len(train_ds)}, val={len(val_ds)}")
print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
print(f"Estimated GPU memory: ~{n_params * 4 / 1024**2:.0f}MB (model only)")

optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                              weight_decay=tcfg["weight_decay"])
warmup = tcfg["warmup_epochs"]
total_ep = tcfg["epochs"]
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lambda ep: ep / warmup if ep < warmup
    else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (total_ep - warmup))))
criterion = FocalLoss(tcfg["focal_alpha"], tcfg["focal_gamma"], tcfg["label_smoothing"])

wandb.init(project="birdclef-2026", name=f"EXP001-c000-f{fold}", config=CONFIG)
"""))

    # ── Training loop ──
    cells.append(make_cell("""
best_auc = 0.0
patience_counter = 0
patience = tcfg.get("early_stopping_patience", 10)
gpu_start = time.time()

for epoch in range(total_ep):
    # Train
    model.train()
    losses = []
    for emb, tgt, hr in train_loader:
        emb, tgt, hr = emb.to(device), tgt.to(device), hr.to(device)
        if aug_cfg.get("mixup_alpha", 0) > 0 and random.random() < aug_cfg.get("mixup_prob", 0):
            emb, tgt = mixup_data(emb, tgt, aug_cfg["mixup_alpha"])
        optimizer.zero_grad()
        loss = criterion(model(emb, hr), tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()

    # Validate
    model.eval()
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for emb, tgt, hr in val_loader:
            emb, hr = emb.to(device), hr.to(device)
            preds_all.append(torch.sigmoid(model(emb, hr)).cpu().numpy())
            tgts_all.append(tgt.numpy())
    preds_arr = np.concatenate(preds_all)
    tgts_arr = np.concatenate(tgts_all)

    aucs = []
    for i in range(tgts_arr.shape[1]):
        if tgts_arr[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(tgts_arr[:, i], preds_arr[:, i]))
            except ValueError:
                pass
    val_auc = np.mean(aucs) if aucs else 0.0
    train_loss = np.mean(losses)

    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_auc": val_auc,
               "n_classes": len(aucs), "gate": torch.sigmoid(model.gate).item()})

    if epoch % 10 == 0 or val_auc > best_auc:
        elapsed_gpu = time.time() - gpu_start
        print(f"Ep {epoch+1}/{total_ep}: loss={train_loss:.4f} auc={val_auc:.4f} "
              f"({len(aucs)} cls) gate={torch.sigmoid(model.gate).item():.2f} "
              f"GPU={elapsed_gpu:.0f}s")

    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), "best_sed_head.pth")
        oof_df = pd.DataFrame(preds_arr, columns=labels)
        oof_df.to_csv("oof.csv", index=False)
        print(f"  -> Best: {best_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

gpu_time = time.time() - gpu_start
print(f"\\nDone. Best AUC: {best_auc:.4f}, GPU time: {gpu_time:.0f}s ({gpu_time/60:.1f}min)")
"""))

    # ── Per-class analysis ──
    cells.append(make_cell("""
# Per-class AUC analysis (which species are hard?)
model.load_state_dict(torch.load("best_sed_head.pth", map_location=device))
model.eval()

preds_all, tgts_all = [], []
with torch.no_grad():
    for emb, tgt, hr in val_loader:
        emb, hr = emb.to(device), hr.to(device)
        preds_all.append(torch.sigmoid(model(emb, hr)).cpu().numpy())
        tgts_all.append(tgt.numpy())
preds_arr = np.concatenate(preds_all)
tgts_arr = np.concatenate(tgts_all)

class_aucs = {}
for i, label in enumerate(labels):
    if tgts_arr[:, i].sum() > 0:
        try:
            class_aucs[label] = roc_auc_score(tgts_arr[:, i], preds_arr[:, i])
        except ValueError:
            pass

# Worst classes
sorted_aucs = sorted(class_aucs.items(), key=lambda x: x[1])
print("\\nWorst 20 classes:")
for label, auc in sorted_aucs[:20]:
    cls = taxonomy[taxonomy["primary_label"] == label]["class_name"].values
    cls_str = cls[0] if len(cls) > 0 else "?"
    n_pos = int(tgts_arr[:, label2idx[label]].sum())
    print(f"  {label:20s} ({cls_str:10s}) AUC={auc:.3f}  n={n_pos}")

print("\\nBest 10 classes:")
for label, auc in sorted_aucs[-10:]:
    print(f"  {label:20s} AUC={auc:.3f}")

# By class_name (Aves vs Insecta vs Amphibia)
print("\\nAUC by class_name:")
for cls_name in taxonomy["class_name"].unique():
    cls_labels = taxonomy[taxonomy["class_name"] == cls_name]["primary_label"].tolist()
    cls_a = [class_aucs[l] for l in cls_labels if l in class_aucs]
    if cls_a:
        print(f"  {cls_name:12s}: mean={np.mean(cls_a):.3f}, min={np.min(cls_a):.3f}, "
              f"n_species={len(cls_a)}")
"""))

    # ── Save results ──
    cells.append(make_cell("""
result = {
    "best_auc": best_auc,
    "fold": fold,
    "epochs_run": epoch + 1,
    "gpu_time_seconds": gpu_time,
    "n_classes_evaluated": len(aucs),
    "model": "AttentionSEDHead",
    "embedding": "BEATs_iter3_plus_AS2M",
    "input_dim": INPUT_DIM,
    "total_params": n_params,
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

wandb.log({"best_auc": best_auc, "gpu_time": gpu_time})
wandb.finish()
print("\\nFinal result:", json.dumps(result, indent=2))
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

    output_path = Path(__file__).parent / "birdclef-2026-beats-sed-work.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    generate()

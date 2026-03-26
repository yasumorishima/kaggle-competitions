"""
BirdCLEF+ 2026 — EXP001: BEATs Embedding + Attention SED Head
===============================================================
2-stage design to minimize GPU usage:
  Stage 1: Extract BEATs embeddings on CPU (no GPU needed)
  Stage 2: Train lightweight Attention SED head (GPU, minutes not hours)

Differentiators:
1. BEATs (AudioSet 2M pretrained) embeddings — richer than Perch for non-bird species
2. Attention Pooling SED — focuses on vocalizing frames, not entire clip
3. Metadata injection — hour/site as features (species are time/location dependent)
4. Focal loss — handles 234-class imbalance
"""

import os
import gc
import ast
import yaml
import math
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# Stage 1: BEATs Embedding Extraction (CPU only)
# ============================================================
def extract_embeddings(cfg: dict, train_df: pd.DataFrame, data_dir: Path):
    """Extract BEATs embeddings for all audio files. Runs on CPU."""
    import sys
    beats_dir = "/kaggle/input/beats-pretrained"
    if beats_dir not in sys.path:
        sys.path.insert(0, beats_dir)
    from BEATs import BEATs, BEATsConfig

    emb_cfg = cfg["embedding"]
    checkpoint_path = emb_cfg["pretrained_path"]
    sr = emb_cfg["sample_rate"]
    chunk_samples = sr * emb_cfg["chunk_duration"]

    # Load model on CPU
    print("Loading BEATs model...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    beats_config = BEATsConfig(checkpoint["cfg"])
    model = BEATs(beats_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"BEATs loaded: embed_dim={beats_config.encoder_embed_dim}")

    all_embeddings = []  # [N, embed_dim] — mean-pooled per clip
    all_frame_embeddings = []  # [N, max_frames, embed_dim] — for SED
    max_frames = 0

    audio_dir = data_dir / "train_audio"
    batch_size = emb_cfg["batch_size"]

    # Process in batches for memory efficiency
    filenames = train_df["filename"].tolist()
    n_batches = math.ceil(len(filenames) / batch_size)

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(filenames))
            batch_files = filenames[start:end]

            wavs = []
            for fname in batch_files:
                try:
                    wav, orig_sr = torchaudio.load(audio_dir / fname)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    wav = wav.squeeze(0)
                    if orig_sr != sr:
                        wav = torchaudio.functional.resample(wav, orig_sr, sr)
                except Exception:
                    wav = torch.zeros(chunk_samples)

                # Center crop or pad to chunk_duration
                if wav.shape[0] > chunk_samples:
                    offset = (wav.shape[0] - chunk_samples) // 2
                    wav = wav[offset:offset + chunk_samples]
                elif wav.shape[0] < chunk_samples:
                    pad = chunk_samples - wav.shape[0]
                    wav = F.pad(wav, (pad // 2, pad - pad // 2))

                wavs.append(wav)

            wavs = torch.stack(wavs)  # [B, T]
            padding_mask = torch.zeros(wavs.shape, dtype=torch.bool)
            features = model.extract_features(wavs, padding_mask=padding_mask)[0]
            # features: [B, N_frames, 768]

            # Mean-pooled embedding
            mean_emb = features.mean(dim=1).numpy()  # [B, 768]
            all_embeddings.append(mean_emb)

            # Frame-level for SED (store as variable-length)
            frame_emb = features.numpy()  # [B, N_frames, 768]
            all_frame_embeddings.append(frame_emb)
            max_frames = max(max_frames, frame_emb.shape[1])

            if batch_idx % 50 == 0:
                print(f"  Embedding batch {batch_idx}/{n_batches} "
                      f"({end}/{len(filenames)} files)")

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 768]
    print(f"Embeddings extracted: {all_embeddings.shape}")

    # Pad frame embeddings to uniform length
    frame_list = []
    for batch_frames in all_frame_embeddings:
        B, T, D = batch_frames.shape
        if T < max_frames:
            padded = np.zeros((B, max_frames, D), dtype=np.float32)
            padded[:, :T, :] = batch_frames
            frame_list.append(padded)
        else:
            frame_list.append(batch_frames)
    all_frame_embeddings = np.concatenate(frame_list, axis=0)  # [N, max_frames, 768]
    print(f"Frame embeddings: {all_frame_embeddings.shape}")

    # Save
    np.savez_compressed(
        emb_cfg["output_file"],
        mean_embeddings=all_embeddings,
        frame_embeddings=all_frame_embeddings,
    )
    print(f"Saved to {emb_cfg['output_file']}")

    del model
    gc.collect()
    return all_embeddings, all_frame_embeddings


# ============================================================
# Stage 2: Attention SED Head (lightweight, GPU)
# ============================================================
class AttentionSEDHead(nn.Module):
    """
    Takes precomputed frame-level embeddings and classifies with attention.
    Much lighter than full BEATs — trains in minutes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.3, use_metadata: bool = False):
        super().__init__()
        self.use_metadata = use_metadata

        # Project embeddings
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism (per-class)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

        # Frame-level classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Metadata embedding (optional)
        if use_metadata:
            self.hour_embed = nn.Embedding(24, 32)
            self.meta_proj = nn.Linear(32, hidden_dim)

    def forward(self, frame_emb: torch.Tensor,
                hour: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            frame_emb: [B, N_frames, input_dim]
            hour: [B] int, hour of day (0-23)
        Returns:
            logits: [B, num_classes]
        """
        x = self.proj(frame_emb)  # [B, N, hidden]

        # Inject metadata
        if self.use_metadata and hour is not None:
            h_emb = self.meta_proj(self.hour_embed(hour))  # [B, hidden]
            x = x + h_emb.unsqueeze(1)  # broadcast over frames

        # Attention-weighted prediction
        att = F.softmax(self.attention(x), dim=1)  # [B, N, C]
        frame_pred = self.classifier(x)  # [B, N, C]
        logits = (att * frame_pred).sum(dim=1)  # [B, C]

        return logits


class EmbeddingDataset(Dataset):
    """Dataset over precomputed embeddings."""

    def __init__(self, frame_embeddings: np.ndarray, targets: np.ndarray,
                 hours: np.ndarray = None, is_train: bool = True,
                 aug_cfg: dict = None):
        self.frame_emb = torch.from_numpy(frame_embeddings).float()
        self.targets = torch.from_numpy(targets).float()
        self.hours = torch.from_numpy(hours).long() if hours is not None else None
        self.is_train = is_train
        self.aug_cfg = aug_cfg or {}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        emb = self.frame_emb[idx].clone()
        target = self.targets[idx]

        if self.is_train:
            # Embedding dropout
            drop_p = self.aug_cfg.get("embedding_dropout", 0)
            if drop_p > 0:
                mask = torch.bernoulli(torch.full(emb.shape[-1:], 1 - drop_p))
                emb = emb * mask / (1 - drop_p)

            # Gaussian noise
            noise_std = self.aug_cfg.get("gaussian_noise_std", 0)
            if noise_std > 0:
                emb = emb + torch.randn_like(emb) * noise_std

        hour = self.hours[idx] if self.hours is not None else torch.tensor(0)
        return emb, target, hour


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def mixup_data(emb, targets, alpha=0.4):
    if alpha <= 0:
        return emb, targets
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(emb.size(0), device=emb.device)
    return lam * emb + (1 - lam) * emb[idx], lam * targets + (1 - lam) * targets[idx]


def train_sed_head(cfg: dict, frame_embeddings: np.ndarray,
                   train_df: pd.DataFrame, labels: list):
    """Train lightweight SED head on precomputed embeddings."""
    import wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    cv_cfg = cfg["cv"]
    aug_cfg = cfg.get("augmentation", {})

    num_classes = model_cfg["num_classes"]
    input_dim = frame_embeddings.shape[-1]  # 768

    # Optional PCA
    if model_cfg.get("use_pca", False):
        pca_dim = model_cfg["pca_dim"]
        print(f"Applying PCA: {input_dim} -> {pca_dim}")
        N, T, D = frame_embeddings.shape
        flat = frame_embeddings.reshape(-1, D)
        scaler = StandardScaler()
        flat = scaler.fit_transform(flat)
        pca = PCA(n_components=pca_dim)
        flat = pca.fit_transform(flat)
        frame_embeddings = flat.reshape(N, T, pca_dim)
        input_dim = pca_dim
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Prepare targets
    label2idx = {l: i for i, l in enumerate(labels)}
    targets = np.zeros((len(train_df), num_classes), dtype=np.float32)
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

    # Extract hour from URL (if available) or set to 12
    hours = np.full(len(train_df), 12, dtype=np.int64)
    # TODO: extract actual recording hour from metadata if available

    # CV split
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    skf = StratifiedKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=seed)
    fold = cv_cfg["fold"]
    train_idx, val_idx = list(skf.split(train_df, train_df["primary_label"]))[fold]

    train_ds = EmbeddingDataset(
        frame_embeddings[train_idx], targets[train_idx],
        hours[train_idx], is_train=True, aug_cfg=aug_cfg)
    val_ds = EmbeddingDataset(
        frame_embeddings[val_idx], targets[val_idx],
        hours[val_idx], is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"] * 2,
        shuffle=False, num_workers=2, pin_memory=True)

    print(f"Fold {fold}: train={len(train_ds)}, val={len(val_ds)}")
    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")

    # Model
    model = AttentionSEDHead(
        input_dim=input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=num_classes,
        dropout=model_cfg["dropout"],
        use_metadata=model_cfg.get("use_metadata", False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SED Head params: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    total_epochs = train_cfg["epochs"]
    warmup = train_cfg["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = FocalLoss(
        alpha=train_cfg["focal_alpha"],
        gamma=train_cfg["focal_gamma"],
        label_smoothing=train_cfg["label_smoothing"],
    )

    # W&B
    wandb.init(project="birdclef-2026", name=f"EXP001-c000-f{fold}", config=cfg)

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 10)

    for epoch in range(total_epochs):
        # Train
        model.train()
        train_losses = []
        for emb, target, hour in train_loader:
            emb, target = emb.to(device), target.to(device)
            hour = hour.to(device)

            # Mixup
            if aug_cfg.get("mixup_alpha", 0) > 0 and random.random() < aug_cfg.get("mixup_prob", 0):
                emb, target = mixup_data(emb, target, aug_cfg["mixup_alpha"])

            optimizer.zero_grad()
            logits = model(emb, hour)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for emb, target, hour in val_loader:
                emb, hour = emb.to(device), hour.to(device)
                logits = model(emb, hour)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(target.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        aucs = []
        for i in range(all_targets.shape[1]):
            if all_targets[:, i].sum() > 0:
                try:
                    aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
                except ValueError:
                    pass
        val_auc = np.mean(aucs) if aucs else 0.0
        train_loss = np.mean(train_losses)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss,
                    "val_auc": val_auc, "n_classes_evaluated": len(aucs)})

        if epoch % 5 == 0 or val_auc > best_auc:
            print(f"Epoch {epoch+1}/{total_epochs}: loss={train_loss:.4f}, "
                  f"auc={val_auc:.4f} ({len(aucs)} classes)")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_sed_head.pth")
            oof_df = pd.DataFrame(all_preds, columns=labels)
            oof_df.to_csv("oof.csv", index=False)
            print(f"  -> New best: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save result
    import json
    result = {"best_auc": best_auc, "fold": fold, "epochs_run": epoch + 1,
              "n_classes": len(aucs), "total_params": total_params}
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    wandb.log({"best_auc": best_auc})
    wandb.finish()
    print(f"\nDone. Best AUC: {best_auc:.4f}")
    return best_auc


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip embedding extraction (use cached)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Data paths
    import glob as _glob
    _slug = "birdclef-2026"
    _matches = _glob.glob(f"/kaggle/input/**/{_slug}", recursive=True)
    DATA_DIR = Path(_matches[0]) if _matches else Path(f"/kaggle/input/{_slug}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    labels = sorted(taxonomy["primary_label"].unique().tolist())
    print(f"Data: {len(train_df)} samples, {len(labels)} classes")

    # Stage 1: Extract embeddings
    emb_file = cfg["embedding"]["output_file"]
    if args.skip_extraction and Path(emb_file).exists():
        print(f"Loading cached embeddings from {emb_file}")
        data = np.load(emb_file)
        frame_embeddings = data["frame_embeddings"]
    else:
        _, frame_embeddings = extract_embeddings(cfg, train_df, DATA_DIR)

    # Stage 2: Train SED head
    best_auc = train_sed_head(cfg, frame_embeddings, train_df, labels)

    return best_auc


if __name__ == "__main__":
    main()

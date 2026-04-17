# %% [markdown]
# # BirdCLEF+ 2026 — W2 Training: eca_nfnet_l0 mel baseline
#
# - **Backbone**: eca_nfnet_l0 (24M params, NFNet系 — Perch v2と異なる特徴抽出で多様性確保)
# - **Mel**: sr=32000, n_mels=128, 5s chunks → 224×224
# - **Loss**: Focal loss (γ=2.0) — 極端な不均衡(1〜499件/クラス)対策
# - **Fold**: StratifiedGroupKFold (primary_label stratified, author grouped)
# - **Augmentations**: Mixup (α=0.5) + SpecAugment (freq+time masking)
# - **Output**: fold別 best checkpoint → Kaggle dataset化 → submit kernel で mount

# %%
# P100 (SM 6.0) compatibility: PyTorch cu128 dropped SM 6.0 support.
# Reinstall cu121 build which supports SM 5.0+.
# IMPORTANT: Do NOT import torch here — C extensions can't be unloaded,
# causing "already has a docstring" RuntimeError in the next cell.
import subprocess, sys
def _ensure_gpu_compat():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=gpu_name,compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("nvidia-smi failed — skipping GPU compat check")
            return
        line = result.stdout.strip().split('\n')[0]
        name, cap = line.rsplit(',', 1)
        name, cap = name.strip(), cap.strip()
        major = int(cap.split('.')[0])
        print(f"GPU: {name}, SM {cap}")
        if major < 7:
            print(f"SM {cap} < 7.0 — reinstalling PyTorch with cu121...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-q',
                '--force-reinstall', '--no-cache-dir',
                'torch', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu121',
            ], check=True, timeout=600)
            # Remove cu128 torchvision — incompatible with cu121 torch,
            # and timm handles missing torchvision via try/except ImportError
            subprocess.run([
                sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision',
            ], capture_output=True, timeout=60)
            # Verify cu121 was actually installed
            ver_check = subprocess.run(
                [sys.executable, '-c', 'import torch; print(torch.__version__, torch.version.cuda)'],
                capture_output=True, text=True, timeout=30)
            print(f"PyTorch cu121 installed: {ver_check.stdout.strip()}")
        else:
            print(f"SM {cap} >= 7.0 — no reinstall needed")
    except Exception as e:
        print(f"GPU compat check failed: {e}")
_ensure_gpu_compat()
del _ensure_gpu_compat

# %%
import numpy as np
import pandas as pd
import os, sys, time, gc, json, random, warnings
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchaudio.transforms as T
import timm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import librosa

warnings.filterwarnings("ignore")
START = time.time()
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

# %%
# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Audio
    sr: int = 32_000
    chunk_duration: float = 5.0
    # Mel spectrogram
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 16_000
    top_db: float = 80.0
    power: float = 2.0
    target_size: tuple = (224, 224)
    in_channels: int = 3
    # Model
    backbone: str = "eca_nfnet_l0"
    num_classes: int = 234
    dropout: float = 0.2
    drop_path_rate: float = 0.2
    # Training
    n_folds: int = 5
    train_folds: list = field(default_factory=lambda: [0])  # fold0 first (~6h), add more later
    epochs: int = 30
    patience: int = 5  # early stopping patience
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    # Augmentation
    mixup_alpha: float = 0.5
    freq_mask_param: int = 12
    time_mask_param: int = 48
    # Focal loss
    focal_gamma: float = 2.0
    # Misc
    seed: int = 42
    num_workers: int = 2

    @property
    def chunk_samples(self) -> int:
        return int(self.sr * self.chunk_duration)

cfg = Config()
print(f"Config: backbone={cfg.backbone}, epochs={cfg.epochs}, folds={cfg.train_folds}")

# %%
# ══════════════════════════════════════════════════════════════
# SEED & DEVICE
# ══════════════════════════════════════════════════════════════
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, SM {torch.cuda.get_device_capability()}")

# %%
# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
TRAIN_DIR = DATA_ROOT / "train_audio"
OUTPUT_DIR = Path("/kaggle/working")

meta = pd.read_csv(DATA_ROOT / "train.csv")
print(f"Metadata: {meta.shape}")

# Build label mapping (234 classes from sample_submission)
sub_df = pd.read_csv(DATA_ROOT / "sample_submission.csv", nrows=1)
SPECIES = list(sub_df.columns[1:])
label2idx = {sp: i for i, sp in enumerate(SPECIES)}
idx2label = {i: sp for sp, i in label2idx.items()}
print(f"Species in submission: {len(SPECIES)}")

# Map primary_label to index (species not in submission get -1, will be skipped)
meta["label_idx"] = meta["primary_label"].map(label2idx).fillna(-1).astype(int)
meta = meta[meta["label_idx"] >= 0].reset_index(drop=True)
print(f"Metadata after filtering unknown labels: {meta.shape}")

# File paths
meta["filepath"] = meta["filename"].apply(lambda f: str(TRAIN_DIR / f))

# %%
# ══════════════════════════════════════════════════════════════
# FOLD SPLIT: StratifiedGroupKFold (stratify=label, group=author)
# ══════════════════════════════════════════════════════════════
sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
meta["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(
    sgkf.split(meta, meta["label_idx"], groups=meta["author"])
):
    meta.loc[val_idx, "fold"] = fold_idx

print("Fold distribution:")
print(meta["fold"].value_counts().sort_index())
print(f"\nFold 0 val species coverage: {meta[meta['fold']==0]['label_idx'].nunique()}/{len(SPECIES)}")

# %%
# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, is_train=True):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.is_train = is_train
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=cfg.power,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.target_size = cfg.target_size
        if is_train:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
            self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load audio
        try:
            y, _ = librosa.load(row["filepath"], sr=self.cfg.sr, mono=True)
        except Exception:
            y = np.zeros(self.cfg.chunk_samples, dtype=np.float32)

        chunk = self.cfg.chunk_samples

        # Random crop (train) or center crop (val)
        if len(y) > chunk:
            if self.is_train:
                start = random.randint(0, len(y) - chunk)
            else:
                start = (len(y) - chunk) // 2
            y = y[start : start + chunk]
        else:
            # Pad short audio
            y = np.pad(y, (0, max(0, chunk - len(y))))

        # Normalize
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

        waveform = torch.from_numpy(y).float()

        # Mel spectrogram
        mel = self.db(self.mel_spec(waveform.unsqueeze(0)))  # (1, n_mels, T)
        mel = F.interpolate(mel.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)  # (1, 224, 224)

        # Normalize to [0, 1]
        mel_min = mel.min()
        mel_max = mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)

        # SpecAugment (train only)
        if self.is_train:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        # 3-channel
        mel = mel.repeat(3, 1, 1)  # (3, 224, 224)

        # Label (one-hot for focal loss)
        label = torch.zeros(self.cfg.num_classes, dtype=torch.float32)
        label[row["label_idx"]] = 1.0

        return mel, label

# %%
# ══════════════════════════════════════════════════════════════
# MIXUP
# ══════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

# %%
# ══════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()

# %%
# ══════════════════════════════════════════════════════════════
# MODEL: eca_nfnet_l0 + GEM + Attention SED Head
# ══════════════════════════════════════════════════════════════
class GEMPool(nn.Module):
    """Generalized Mean Pooling over frequency axis."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, F, T) → pool over F → (B, C, T)
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class AttSEDHead(nn.Module):
    """Attention-based Sound Event Detection head."""
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
        # x: (B, C, T)
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        logit = (att * cls).sum(dim=-1)  # (B, num_classes)
        return logit


class BirdModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True,
            in_chans=cfg.in_channels, features_only=False,
            global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.gem = GEMPool(p_init=3.0)
        self.head = AttSEDHead(feat_dim, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        feat = self.backbone(x)  # (B, C, F', T')
        pooled = self.gem(feat)  # (B, C, T')
        logit = self.head(pooled)  # (B, num_classes)
        return logit

# %%
# ══════════════════════════════════════════════════════════════
# SCHEDULER: Cosine with linear warmup
# ══════════════════════════════════════════════════════════════
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

# %%
# ══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg):
    model.train()
    losses = []
    for batch_idx, (mel, labels) in enumerate(loader):
        mel, labels = mel.to(device), labels.to(device)

        # Mixup
        mel, labels = mixup_data(mel, labels, alpha=cfg.mixup_alpha)

        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(mel)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - START
            print(f"    step {batch_idx+1}/{len(loader)}, loss={np.mean(losses[-100:]):.4f}, elapsed={elapsed/60:.1f}min")

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        with autocast("cuda"):
            logits = model(mel)
            loss = criterion(logits, labels)
        losses.append(loss.item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # Macro AUC (skip classes with no positive samples in val)
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
            try:
                aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
            except ValueError:
                pass
    macro_auc = np.mean(aucs) if aucs else 0.0

    return np.mean(losses), macro_auc, preds


def train_fold(fold, meta, cfg):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")

    train_df = meta[meta["fold"] != fold]
    val_df = meta[meta["fold"] == fold]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = BirdCLEFDataset(train_df, cfg, is_train=True)
    val_ds = BirdCLEFDataset(val_df, cfg, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = BirdModel(cfg).to(device)
    criterion = FocalLoss(gamma=cfg.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda")
    scheduler = CosineWarmupScheduler(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.lr)

    best_auc = 0.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(cfg.epochs):
        lr = scheduler.step(epoch)
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg)
        val_loss, val_auc, _ = validate(model, val_loader, criterion)
        epoch_time = time.time() - t0

        print(f"  epoch {epoch+1}/{cfg.epochs} | lr={lr:.6f} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_auc={val_auc:.4f} | time={epoch_time:.0f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            no_improve = 0
            ckpt_path = OUTPUT_DIR / f"fold{fold}_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "config": {
                    "backbone": cfg.backbone,
                    "num_classes": cfg.num_classes,
                    "dropout": cfg.dropout,
                    "drop_path_rate": cfg.drop_path_rate,
                    "in_channels": cfg.in_channels,
                    "n_mels": cfg.n_mels,
                    "n_fft": cfg.n_fft,
                    "hop_length": cfg.hop_length,
                    "fmin": cfg.fmin,
                    "fmax": cfg.fmax,
                    "target_size": list(cfg.target_size),
                    "sr": cfg.sr,
                    "chunk_duration": cfg.chunk_duration,
                },
            }, ckpt_path)
            print(f"  ★ New best: AUC={val_auc:.4f} @ epoch {epoch+1}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping: no improvement for {cfg.patience} epochs")
                break

    print(f"\nFold {fold} done: best_auc={best_auc:.4f} @ epoch {best_epoch}")

    # Cleanup
    del model, optimizer, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return best_auc, best_epoch

# %%
# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
results = {}
for fold in cfg.train_folds:
    best_auc, best_epoch = train_fold(fold, meta, cfg)
    results[fold] = {"best_auc": best_auc, "best_epoch": best_epoch}

print("\n" + "=" * 60)
print("ALL FOLDS COMPLETE")
print("=" * 60)
aucs = [r["best_auc"] for r in results.values()]
print(f"CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
for fold, r in results.items():
    print(f"  fold {fold}: AUC={r['best_auc']:.4f} @ epoch {r['best_epoch']}")

# Save results summary
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

total = time.time() - START
print(f"\nTotal time: {total/60:.1f} min ({total/3600:.1f} h)")

# List output files
for p in sorted(OUTPUT_DIR.glob("fold*_best.pt")):
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"  {p.name}: {size_mb:.1f} MB")

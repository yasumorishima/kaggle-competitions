"""
Generate the ROGII TCN-HYBRID notebook.

Idea (verified-finding driven):
  A raw 7-channel geometry TCN scored CV toe RMSE 16.3 and lost to the GBT
  baseline (LB 10.224). The gap is the GBT's engineered physics features
  (beam-search / DTW / particle-filter / multi-scale-NCC / formation-plane-KNN
  per-row TVT estimates). This notebook feeds those SAME ~200 features into a
  sequence model: a dilated-residual 1D-CNN (TCN) that convolves along the
  ordered toe rows of each well. The flat per-row GBT ignores row-to-row
  continuity (it patches it post-hoc with Savitzky-Golay smoothing); the TCN
  learns it directly.

How it is built (patcher, like rogii-wellbore-baseline):
  Take the public pub_dwt notebook (9.251) and KEEP cells 0-5 = the feature
  pipeline (imports, CFG, build_well/build_dataset, train_df[cache]/test_df
  [computed in-kernel]/features). DROP cells 6-23 = GBT/Climber/Optuna/postproc/
  submission. APPEND a TCN head (this file's TCN_* cells).

  train_df comes from the cached 7.4GB artifacts/data/train.csv (per-row engineered
  features + target=delta). test_df is computed in-kernel by build_dataset over the
  (hidden-at-scoring) test wells -- features are available for test because
  build_well runs the identical path for train/test (only `target` is train-only).

SMOKE is hardcoded (Kaggle injects no env). SMOKE=True caps train.csv to nrows and
runs a tiny TCN / 2 folds / 2 epochs to prove load->seq->train->submission end to
end. FLIP to False, regenerate, commit, push for the full run.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "rogii-wellbore-work", "pub_dwt",
                   "9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb")
OUT = os.path.join(HERE, "rogii-tcn-hybrid.ipynb")

# Flip to False (regenerate + commit + push) for the full run.
SMOKE = True

# train.csv row cap for SMOKE (first ~8 wells of toe rows).
SMOKE_NROWS = 40000

# ---------------------------------------------------------------------------
# Appended TCN-head cells. Raw strings (NOT f-strings: bodies contain many { }).
# `__SMOKE__` is the only injected token.
# ---------------------------------------------------------------------------

TCN_CONFIG = r'''
# === TCN-hybrid head: sequence model over pub_dwt per-row features ===
import torch, torch.nn as nn
import gc
SMOKE = __SMOKE__
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
N_SPLITS = 5
EPOCHS   = 2   if SMOKE else 30
PATIENCE = 1   if SMOKE else 6
CH       = 32  if SMOKE else 128
NB       = 2   if SMOKE else 7
LR = 1e-3
WD = 1e-4
print(f"[TCN] SMOKE={SMOKE} EPOCHS={EPOCHS} CH={CH} NB={NB} n_features={len(features)}")

DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        _t = torch.zeros(8, device="cuda"); _ = float((_t + 1).sum().item())
        DEVICE = "cuda"; print("GPU OK:", torch.cuda.get_device_name(0))
    else:
        print("cuda not available")
except Exception as e:
    print("GPU unusable -> CPU:", str(e)[:140])
print("DEVICE =", DEVICE)
'''

TCN_BUILD = r'''
# === standardize features (fit on train) + build per-well ordered toe sequences ===
feat = list(features)
# feature mean/std on train toe rows (inf -> nan, nan-skipping stats)
Xtr_all = train_df[feat].to_numpy(dtype=np.float32)
Xtr_all[~np.isfinite(Xtr_all)] = np.nan
mu = np.nanmean(Xtr_all, axis=0).astype(np.float32)
sd = np.nanstd(Xtr_all, axis=0).astype(np.float32); sd[sd < 1e-6] = 1.0
del Xtr_all; gc.collect()

ytr = train_df['target'].to_numpy(dtype=np.float32)
ymu = float(np.nanmean(ytr)); ysd = float(np.nanstd(ytr)) or 1.0
print(f"target delta mean={ymu:.3f} std={ysd:.3f} feat={len(feat)}")

def _norm(M):
    M = M.astype(np.float32, copy=True)
    M[~np.isfinite(M)] = np.nan
    M = (M - mu) / sd
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

def _row_idx(idser):
    return idser.str.rsplit('_', n=1).str[-1].astype(int)

def build_seqs(df, with_target):
    df = df.copy()
    df['_ri'] = _row_idx(df['id'])
    out = []
    for wid, gdf in df.groupby('well', sort=False):
        gdf = gdf.sort_values('_ri')
        item = {
            'wid': wid,
            'X': _norm(gdf[feat].to_numpy()),
            'ids': gdf['id'].to_numpy(),
            'last': float(gdf['last_known_tvt'].iloc[0]),
        }
        if with_target:
            t = gdf['target'].to_numpy(dtype=np.float32)
            item['t'] = (t - ymu) / ysd
        out.append(item)
    return out

train_seqs = build_seqs(train_df, True)
test_seqs  = build_seqs(test_df, False)
groups = np.array([s['wid'] for s in train_seqs])
print("train wells", len(train_seqs), "test wells", len(test_seqs))
try:
    del train_df, X, y, g, X_test
except Exception:
    pass
gc.collect()
'''

TCN_MODEL = r'''
# === TCN model ===
class TCNBlock(nn.Module):
    def __init__(s, c, d):
        super().__init__()
        s.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.b1 = nn.BatchNorm1d(c)
        s.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.b2 = nn.BatchNorm1d(c)
        s.act = nn.ReLU(); s.do = nn.Dropout(0.1)
    def forward(s, x):
        y = s.do(s.act(s.b1(s.c1(x)))); y = s.b2(s.c2(y)); return s.act(x + y)

class TCN(nn.Module):
    def __init__(s, cin, c, nb):
        super().__init__()
        s.inp = nn.Conv1d(cin, c, 1)
        s.blocks = nn.ModuleList([TCNBlock(c, 2 ** i) for i in range(nb)])
        s.head = nn.Conv1d(c, 1, 1)
    def forward(s, x):
        x = s.inp(x)
        for b in s.blocks:
            x = b(x)
        return s.head(x).squeeze(1)

def huber(p, t, d=1.0):
    e = p - t; a = e.abs()
    return torch.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d)).mean()

def to_x(s):
    return torch.tensor(s['X'].T[None], dtype=torch.float32, device=DEVICE)  # (1,C,L)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
'''

TCN_TRAIN = r'''
# === GroupKFold train with early stopping (valid toe RMSE in feet) ===
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=(2 if SMOKE else N_SPLITS))
idx = np.arange(len(train_seqs))
fold_states, fold_best = [], []
for fold, (tr, va) in enumerate(gkf.split(idx, groups=groups)):
    model = TCN(len(feat), CH, NB).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best = 1e9; best_state = None; bad = 0
    tr = np.array(tr)
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(tr); tl = 0.0
        for j in tr:
            s = train_seqs[j]
            x = to_x(s); t = torch.tensor(s['t'][None], dtype=torch.float32, device=DEVICE)
            opt.zero_grad(); loss = huber(model(x), t); loss.backward(); opt.step()
            tl += float(loss)
        model.eval(); P, T = [], []
        with torch.no_grad():
            for j in va:
                s = train_seqs[j]
                P.append(model(to_x(s)).cpu().numpy()[0] * ysd + ymu)
                T.append(s['t'] * ysd + ymu)
        vr = rmse(np.concatenate(P), np.concatenate(T))
        if vr < best - 1e-4:
            best = vr; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  fold{fold} ep{ep} tl={tl/max(len(tr),1):.4f} vRMSE={vr:.3f} best={best:.3f}")
        if bad >= PATIENCE:
            print("   early stop"); break
    fold_states.append(best_state); fold_best.append(best)
    print(f"fold{fold} best toe RMSE = {best:.4f}")
print("CV toe RMSE mean =", float(np.mean(fold_best)), "| folds", [round(b,3) for b in fold_best])
'''

TCN_INFER = r'''
# === fold-averaged inference + submission ===
models = []
for st in fold_states:
    m = TCN(len(feat), CH, NB).to(DEVICE); m.load_state_dict(st); m.eval(); models.append(m)

rows = []
with torch.no_grad():
    for s in test_seqs:
        acc = np.zeros(s['X'].shape[0], dtype=np.float64)
        x = to_x(s)
        for m in models:
            acc += m(x).cpu().numpy()[0]
        acc /= max(len(models), 1)
        tvt = s['last'] + (acc * ysd + ymu)
        for i, _id in enumerate(s['ids']):
            rows.append((_id, float(tvt[i])))

pred_df = pd.DataFrame(rows, columns=['id', 'tvt'])
samp = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')
sub = samp[['id']].merge(pred_df, on='id', how='left')
fill = float(pd.Series([s['last'] for s in test_seqs]).mean()) if test_seqs else 0.0
nmiss = int(sub['tvt'].isna().sum())
sub['tvt'] = sub['tvt'].fillna(fill)
sub.to_csv('submission.csv', index=False)
print("submission", sub.shape, "| missing(filled):", nmiss, "| NaN:", int(sub['tvt'].isna().sum()))
print(sub.head())
'''

APPEND_CELLS = [TCN_CONFIG, TCN_BUILD, TCN_MODEL, TCN_TRAIN, TCN_INFER]


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]

    # Keep feature-pipeline cells 0..5; drop GBT/postproc/submission 6..23.
    keep = cells[:6]

    # Patch cell 1: drop the custom `from hill_climbing import Climber` import
    # (module absent on Kaggle; we don't use the GBT/Climber path).
    src1 = "".join(keep[1].get("source", []))
    assert "from hill_climbing import Climber" in src1, "hill_climbing import not found in cell 1"
    src1 = src1.replace("from hill_climbing import Climber\n", "")
    keep[1]["source"] = src1

    # Patch cell 5: SMOKE caps the 7.4GB train.csv read with nrows.
    src5 = "".join(keep[5].get("source", []))
    read_anchor = ('train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", '
                   'low_memory=False)')
    assert read_anchor in src5, "train.csv read anchor not found in cell 5"
    if SMOKE:
        src5 = src5.replace(
            read_anchor,
            'train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", '
            f'low_memory=False, nrows={SMOKE_NROWS})')
    keep[5]["source"] = src5

    # Append the TCN head.
    smoke_lit = "True" if SMOKE else "False"
    for body in APPEND_CELLS:
        keep.append(code_cell(body.replace("__SMOKE__", smoke_lit)))

    nb["cells"] = keep
    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    nb.setdefault("metadata", {})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT} with {len(keep)} cells (SMOKE={SMOKE})")


if __name__ == "__main__":
    main()

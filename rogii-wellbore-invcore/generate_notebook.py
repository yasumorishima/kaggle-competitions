#!/usr/bin/env python3
"""
ROGII inversion-core GO/NO-GO diagnostic notebook generator (self-contained).

Builds a Kaggle T4 notebook that tests ONE question before committing to the
full MDN+MTP build:

  Does a learned GR->stratigraphy inversion net (K=1 degenerate MDN with
  typewell cross-attention) beat the hand-tuned L2 emission beam aligner that
  is the signal source of the current champion (CV 9.978)?

Success gate (advisor design, agentId af338789ded609a30):
  (i)  R_mdn_synth   < 0.70 * R_align_synth          (learned core > L2 aligner on synthetic pairs)
  (ii) R_mdn_realkn  < R_align_realkn   AND  R_mdn_realkn / R_mdn_synth < 2.0
       (transfers to real known-zone; synth->real gap is bounded)

Both green -> proceed to full MTP K=3..5 + fine-tune.
gap-ratio (ii) > 2 -> stop, improve the forward simulator physics first.

Self-contained: no GBT pipeline, no internet. Reuses only the verified data
schema (TRAIN_DIR/<wid>__horizontal_well.csv + <wid>__typewell.csv; columns
TVT_input/X/Y/Z/GR ; typewell TVT/GR) and the L2 beam aligner emission
(gv-tw_gr[ni])**2/es (de=0 == pure L2, identical to champion's signal source).

FLIP SMOKE=False below, regenerate, commit, push for the full diagnostic run.
Push via GHA:
  gh workflow run kaggle-push.yml -f notebook_dir=rogii-wellbore-invcore \
     -f kernel_id=yasunorim/rogii-invcore-diag -f memo="invcore go/no-go"
"""
import json
from pathlib import Path

SMOKE = True  # True: 8 wells, 2 epochs, fast pipeline check. False: full diagnostic.

BASE = Path(__file__).resolve().parent
OUT = BASE / "rogii-invcore-diag.ipynb"

# ---------------------------------------------------------------- cell sources
CELL_CONFIG = r'''
# === ROGII inversion-core GO/NO-GO diagnostic ===
SMOKE = __SMOKE__
import os, math, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from numba import njit

SEED = 42
np.random.seed(SEED)

def _find():
    for p in [Path("/kaggle/input/rogii-wellbore-geology-prediction"),
              Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")]:
        if (p/"train").exists(): return p
    for p in Path("/kaggle/input").glob("*/sample_submission.csv"): return p.parent
    raise FileNotFoundError("Data not found")

DATA = _find(); TRAIN_DIR = DATA/"train"
HW = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
if SMOKE: HW = HW[:8]
print(f"SMOKE={SMOKE}  train wells={len(HW)}")

L      = 96      # toe-window length (resampled rows)
M      = 256     # typewell profile length (resampled)
N_SYN  = (200 if SMOKE else 6000)   # synthetic training pairs
EPOCHS = (2 if SMOKE else 40)
'''

CELL_FORWARD = r'''
# === forward simulator (self-built; ROGII forward = project typewell GR along bed model) ===
def _smooth(a, fill, k):
    a = np.where(np.isfinite(a), a, fill).astype(np.float64)
    if k <= 0 or len(a) < 2*k+1: return a
    ker = np.ones(2*k+1)/(2*k+1)
    return np.convolve(a, ker, mode="same")

def _nn(arr, v):
    return int(np.argmin(np.abs(arr - v)))

def _load(hw_path):
    wid = hw_path.stem.replace("__horizontal_well","")
    twp = hw_path.parent/(wid+"__typewell.csv")
    if not twp.exists(): return None
    try:
        hw = pd.read_csv(hw_path); tw = pd.read_csv(twp).sort_values("TVT")
    except Exception: return None
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 60 or len(tw) < 8: return None
    tw_tvt = tw["TVT"].to_numpy(np.float64); tw_gr = tw["GR"].to_numpy(np.float64)
    dtvt = float(np.median(np.diff(tw_tvt)))
    if dtvt <= 0: return None
    X = kn["X"].to_numpy(np.float64); Y = kn["Y"].to_numpy(np.float64); Z = kn["Z"].to_numpy(np.float64)
    ttvt = kn["TVT_input"].to_numpy(np.float64)
    gr = (kn["GR"].astype(float).interpolate(limit_direction="both")
          .fillna(float(np.nanmean(tw_gr))).to_numpy(np.float64))
    return dict(wid=wid, X=X, Y=Y, Z=Z, ttvt=ttvt, gr=gr,
                tw_tvt=tw_tvt, tw_gr=tw_gr, dtvt=dtvt)

WELLS = [w for w in (_load(p) for p in HW) if w is not None]
print(f"loaded {len(WELLS)} wells")

def _resample(x, n):
    if len(x) == n: return x.astype(np.float64)
    xi = np.linspace(0, 1, len(x)); xo = np.linspace(0, 1, n)
    return np.interp(xo, xi, x).astype(np.float64)

def synth_pair(w, rng):
    """Generate one synthetic toe window from a real well's geometry + its typewell.
       anchor convention = last-KNOWN row before the toe (matches real_known()).
       Mismatch (nonlinear bed-thickness warp + missing-bed dropout + GR shift) is
       injected so the horizontal GR does NOT trivially equal typewell GR -> the L2
       aligner cannot solve the generative inverse for free (fixes degenerate sim).
       Returns gr_w[L], tvt_w[L], anchor, twg_r[M], twt_r[M], tw_gr_native, tw_tvt_native."""
    n = len(w["X"]); split = int(rng.integers(int(0.4*n), int(0.7*n)))
    j1 = min(n, split + int(rng.integers(L, 2*L)))
    if j1 - split < 12 or split < 2: return None
    # geometry from the last-known row (split-1) into the toe -> first step is genuine
    X, Y, Z = w["X"][split-1:j1], w["Y"][split-1:j1], w["Z"][split-1:j1]
    dZt = np.diff(Z); dht = np.hypot(np.diff(X), np.diff(Y))          # len = j1-split (toe steps)
    # --- sampled geology ---
    dip      = rng.uniform(-0.5, 0.5)
    thick_s  = float(np.exp(rng.normal(0, 0.18)))
    gr_shift = rng.normal(0, 8.0)                                    # beyond es=144 absorption
    sigma_n  = rng.uniform(3.0, 10.0)
    warp_amp = rng.uniform(0.0, 0.6); warp_f = rng.uniform(0.5, 2.0) # nonlinear thickness mismatch
    tw_tvt = w["tw_tvt"].astype(np.float64); tw_gr = w["tw_gr"].astype(np.float64)
    lo, hi = float(tw_tvt[0]), float(tw_tvt[-1]); span = hi - lo + 1e-9
    anchor = float(rng.uniform(np.percentile(tw_tvt, 20), np.percentile(tw_tvt, 65)))
    # --- true subsurface TVT path: anchor + dip-driven geometry (BENCH dexp form) ---
    dexp = -dZt + dip*dht
    tvt_true = anchor + np.cumsum(dexp)
    if tvt_true.min() < lo or tvt_true.max() > hi: return None       # stay inside typewell support
    # --- nonlinear monotone warp: where each subsurface depth maps in the typewell ---
    u = (tvt_true - lo)/span
    u2 = u + warp_amp*np.sin(2*np.pi*warp_f*u)/(2*np.pi*warp_f)      # monotone for amp<1
    u2 = np.clip(u2*thick_s, 0.0, 1.0)
    map_tvt = lo + u2*span
    gr = np.interp(map_tvt, tw_tvt, tw_gr) + gr_shift
    # --- missing-bed dropout: a depth band reads a flat anomalous value (extra/absent bed) ---
    if rng.random() < 0.4 and len(gr) > 20:
        a = int(rng.integers(0, len(gr)-10)); b = a + int(rng.integers(5, 15))
        gr[a:b] = float(np.median(tw_gr)) + rng.normal(0, sigma_n)
    gr = _smooth(gr, float(np.nanmean(tw_gr)), 1) + rng.normal(0, sigma_n, gr.shape)
    # resample window to L; typewell to M (MDN) + keep native (aligner baseline)
    gr_w = _resample(gr, L); tvt_w = _resample(tvt_true, L)
    twg = _resample(tw_gr, M); twt = _resample(tw_tvt, M)
    return gr_w, tvt_w, anchor, twg, twt, tw_gr, tw_tvt
'''

CELL_BASELINE = r'''
# === baseline: pure L2 emission beam aligner (de=0 == champion 9.978 signal source) ===
@njit(cache=True)
def _beam_l2(sgr, tw_gr, si, BS, mc, es, W):
    n=len(sgr); nt=len(tw_gr); MAX=BS*(2*W+6)
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30); bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-W,W+1):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        if nc==0:
            bp=np.int64(0)
            for bi in range(1,bn):
                if bcost[bi]<bcost[bp]: bp=bi
            ci0=bidx[bp]
            if ci0<0: ci0=np.int64(0)
            if ci0>=nt: ci0=np.int64(nt-1)
            cI[0]=ci0; cC[0]=bcost[bp]; cP[0]=bp; nc=np.int64(1)
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]; cC[i],cC[mi]=cC[mi],cC[i]; cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

def align_l2(gr_w, anchor, tw_gr_native, tw_tvt_native, true_for_w=None):
    """Pure L2 emission beam aligner on NATIVE typewell (champion-strength baseline).
       W is sized generously to cover the per-step TVT change so L2 is not weakened."""
    twg = tw_gr_native.astype(np.float64); twt = tw_tvt_native.astype(np.float64)
    si = _nn(twt, anchor)
    ndt = float(np.median(np.diff(twt))); ndt = ndt if ndt > 1e-9 else 1.0
    if true_for_w is not None and len(true_for_w) > 1:
        step = float(np.percentile(np.abs(np.diff(true_for_w)), 95))
        W = int(np.clip(np.ceil(step/ndt) + 4, 4, 40))
    else:
        W = 8
    p = _beam_l2(gr_w.astype(np.float64), twg, si, 12, 20.0, 144.0, W)
    return twt[p]
'''

CELL_MODEL = r'''
# === K=1 degenerate MDN: TCN encoder -> typewell cross-attention -> per-row TVT ===
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV)

class TCNBlock(nn.Module):
    def __init__(s, c, k=5, d=1):
        super().__init__()
        p = (k-1)//2*d
        s.c1 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        s.c2 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        s.n  = nn.GroupNorm(8, c)
    def forward(s, x):
        h = F.gelu(s.c1(x)); h = s.c2(h); return F.gelu(s.n(x + h))

class InvCore(nn.Module):
    """Horizontal GR window (query) attends to typewell GR profile (key/value).
       Outputs per-row TVT increment; TVT = anchor + cumsum(dtvt_pred)."""
    def __init__(s, C=64):
        super().__init__()
        s.in_h = nn.Conv1d(4, C, 1)          # [GR,dGR,pos,one]
        s.tcn  = nn.Sequential(TCNBlock(C,5,1), TCNBlock(C,5,2), TCNBlock(C,5,4))
        s.in_t = nn.Conv1d(2, C, 1)          # typewell [GR,dGR]
        s.tw_enc = nn.Sequential(TCNBlock(C,5,1), TCNBlock(C,5,2))
        s.attn = nn.MultiheadAttention(C, 4, batch_first=True)
        s.head = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))
    def forward(s, gh, gt):
        # gh: (B,4,L)  gt: (B,2,M)
        q = s.tcn(s.in_h(gh)).transpose(1,2)         # (B,L,C)
        kv = s.tw_enc(s.in_t(gt)).transpose(1,2)     # (B,M,C)
        a,_ = s.attn(q, kv, kv)                       # (B,L,C)
        dtvt = s.head(q + a).squeeze(-1)              # (B,L)
        return dtvt

def make_feats(gr_w, twg, twt):
    gr = (gr_w - gr_w.mean())/max(gr_w.std(), 1e-3)      # clamp: avoid blow-up on flat windows
    dgr = np.gradient(gr)
    pos = np.linspace(0,1,len(gr))
    gh = np.stack([gr, dgr, pos, np.ones_like(gr)], 0)
    tg = (twg - twg.mean())/max(twg.std(), 1e-3)
    gt = np.stack([tg, np.gradient(tg)], 0)
    return gh.astype(np.float32), gt.astype(np.float32)
'''

CELL_RUN = r'''
# === REAL known-zone training (no synthetic): learned core vs L2 aligner on real data ===
# Decisive question, zero synth->real gap: a K=1 cross-attention inversion net trained on
# REAL (GR window -> TVT) known-zone pairs -- does it beat the L2 emission beam aligner on
# HELD-OUT wells' real known-zone? anchor = last-known row TVT; toe = split..split+toe_len.
rng = np.random.default_rng(SEED)
nw = len(WELLS); n_hold = max(2, nw//5)
hold = set(rng.choice(nw, n_hold, replace=False).tolist())
tr_wells = [w for i,w in enumerate(WELLS) if i not in hold]
ho_wells = [w for i,w in enumerate(WELLS) if i in hold]
print(f"train wells={len(tr_wells)} holdout wells={len(ho_wells)}")

def _window(w, split, toe_len, gr_aug=0.0):
    n=len(w["X"]); ttvt=w["ttvt"]; gr=w["gr"]; tw_tvt=w["tw_tvt"]; tw_gr=w["tw_gr"]
    j1=min(n, split+toe_len)
    if split<5 or j1-split<12: return None
    anchor=float(ttvt[split-1])
    gr_t=_smooth(gr[split:j1], float(np.nanmean(tw_gr)), 1)
    if gr_aug>0: gr_t = gr_t + rng.normal(0, gr_aug, gr_t.shape)   # GR-noise augmentation
    true=ttvt[split:j1]
    gr_w=_resample(gr_t, L); true_w=_resample(true, L)
    twg=_resample(tw_gr, M); twt=_resample(tw_tvt, M)
    gh,gt=make_feats(gr_w, twg, twt)
    return gh, gt, true_w, anchor, gr_w, tw_gr, tw_tvt

def build_real(wells, multi):
    GH=[];GT=[];Y=[];A=[]
    for w in wells:
        n=len(w["X"])
        fracs=[0.30,0.40,0.50,0.60,0.68] if multi else [0.5]
        for fr in fracs:
            split=int(fr*n); toe=int(rng.integers(L, 2*L))
            r=_window(w, split, toe, gr_aug=4.0)
            if r is None: continue
            gh,gt,true_w,anchor,_,_,_=r
            GH.append(gh);GT.append(gt);Y.append(true_w);A.append(anchor)
    return np.array(GH),np.array(GT),np.array(Y),np.array(A)

GH,GT,Y,A = build_real(tr_wells, True)
assert len(GH)>0, "no real training windows"
print(f"real train windows={len(GH)}")

# --- train K=1 InvCore on real known-zone windows ---
model = InvCore().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
ghT=torch.tensor(GH,device=DEV); gtT=torch.tensor(GT,device=DEV)
yT=torch.tensor(Y.astype(np.float32),device=DEV); aT=torch.tensor(A.astype(np.float32),device=DEV)
dyT = yT - aT[:,None]   # predict TVT offset from the last-known anchor
bs=64
for ep in range(EPOCHS):
    perm=torch.randperm(len(ghT),device=DEV); tot=0.
    model.train()
    for i in range(0,len(ghT),bs):
        idx=perm[i:i+bs]
        dtvt=model(ghT[idx], gtT[idx])
        pred=torch.cumsum(dtvt,1)
        loss=F.smooth_l1_loss(pred, dyT[idx])
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
    sched.step()
    if ep%max(1,EPOCHS//6)==0 or ep==EPOCHS-1:
        print(f"  ep{ep} loss {tot/len(ghT):.4f}")

def mdn_pred(gh, gt, anchor):
    model.eval()
    with torch.no_grad():
        d=model(torch.tensor(gh[None],device=DEV), torch.tensor(gt[None],device=DEV))
        off=torch.cumsum(d,1).cpu().numpy()[0]
    return anchor + off

# --- eval on holdout wells' real known-zone (capped toe; same window for L2 and MDN) ---
rma=[]; rmm=[]; wins=0
for w in ho_wells:
    n=len(w["X"]); split=int(0.7*n); toe=int(1.5*L)
    r=_window(w, split, toe)
    if r is None: continue
    gh,gt,true_w,anchor,gr_w,twgn,twtn=r
    pa=align_l2(gr_w, anchor, twgn, twtn, true_for_w=true_w)
    pm=mdn_pred(gh, gt, anchor)
    ra=float(np.sqrt(np.mean((pa-true_w)**2))); rm=float(np.sqrt(np.mean((pm-true_w)**2)))
    rma.append(ra); rmm.append(rm); wins += (rm < ra)
R_align_realkn=float(np.mean(rma)); R_mdn_realkn=float(np.mean(rmm))
nwh=len(rma)
print(f"[REAL] wells={nwh}  R_align_realkn={R_align_realkn:.4f}  R_mdn_realkn={R_mdn_realkn:.4f}")
print(f"[REAL] MDN beats L2 on {wins}/{nwh} wells")

# --- GO/NO-GO: single decisive gate, zero synth confound ---
beat = R_mdn_realkn < R_align_realkn
margin = (R_align_realkn - R_mdn_realkn)/R_align_realkn*100
print("\n================ GO/NO-GO (real-trained) ================")
print(f"R_mdn_realkn {R_mdn_realkn:.3f} vs R_align_realkn {R_align_realkn:.3f}  ->  "
      f"{'MDN wins' if beat else 'L2 wins'} ({margin:+.1f}% vs L2)")
print(f"VERDICT: {'GO -> learned core beats L2 on real; build full MTP (synth pretrain + real finetune)' if beat else 'NO-GO -> learned core does not beat L2 on real known-zone'}")
print("=========================================================")
'''

CELLS = [CELL_CONFIG, CELL_FORWARD, CELL_BASELINE, CELL_MODEL, CELL_RUN]


def main():
    cells = []
    for src in CELLS:
        s = src.replace("__SMOKE__", "True" if SMOKE else "False").strip("\n") + "\n"
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": s.splitlines(keepends=True)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": False,
                       "language": "python", "sourceType": "notebook", "isGpuEnabled": True},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  (SMOKE={SMOKE}, {len(cells)} cells)")


if __name__ == "__main__":
    main()

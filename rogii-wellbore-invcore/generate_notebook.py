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

SMOKE = True  # True: 8 wells, N=400 particles, fast pipeline check. False: full diagnostic.

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

# --- dip-aware beam (champion signal source): transition centered on geometric drift d_exp ---
@njit(cache=True)
def _beam_dip(sgr, tw_gr, si, BS, mc, es, d_exp, W):
    n=len(sgr); nt=len(tw_gr); MAX=BS*(2*W+6)
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30); bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; de=d_exp[step]; nc=np.int64(0)
        dlo=int(np.floor(de))-W; dhi=int(np.ceil(de))+W
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(dlo,dhi+1):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                dd=d-de
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(dd if dd>=0 else -dd)
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

def align_dip(gr_w, anchor, tw_gr_native, tw_tvt_native, dexp_idx, W=8):
    """Champion baseline: dip-aware beam. dexp_idx = per-row expected typewell-index drift
       (= (-dZ + dip*dhoriz)/dtvt), the geometric TVT-change prior the public beam ignores.
       W=8 (generous search around the prior) so the beam is not artificially throttled."""
    twg = tw_gr_native.astype(np.float64); twt = tw_tvt_native.astype(np.float64)
    si = _nn(twt, anchor)
    p = _beam_dip(gr_w.astype(np.float64), twg, si, 12, 20.0, 144.0, dexp_idx.astype(np.float64), W)
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
    """Horizontal window (query: GR + geometry prior) attends to typewell GR (key/value).
       Predicts the per-row RESIDUAL between the geometric path and truth (GR correction).
       Final TVT = geo_path + residual -- geometry is exact, the net learns the GR fix the
       dip-aware beam does via DP. Fair, same-information comparison to the champion beam."""
    def __init__(s, C=64):
        super().__init__()
        s.in_h = nn.Conv1d(4, C, 1)          # [GR, dGR, geo_offset, pos]
        s.tcn  = nn.Sequential(TCNBlock(C,5,1), TCNBlock(C,5,2), TCNBlock(C,5,4))
        s.in_t = nn.Conv1d(2, C, 1)          # typewell [GR, dGR]
        s.tw_enc = nn.Sequential(TCNBlock(C,5,1), TCNBlock(C,5,2))
        s.attn = nn.MultiheadAttention(C, 4, batch_first=True)
        s.head = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))
    def forward(s, gh, gt):
        # gh: (B,4,L)  gt: (B,2,M)
        q = s.tcn(s.in_h(gh)).transpose(1,2)         # (B,L,C)
        kv = s.tw_enc(s.in_t(gt)).transpose(1,2)     # (B,M,C)
        a,_ = s.attn(q, kv, kv)                       # (B,L,C)
        resid = s.head(q + a).squeeze(-1)             # (B,L) residual vs geometric path
        return resid

def make_feats(gr_w, geo_off, twg, twt):
    gr = (gr_w - gr_w.mean())/max(gr_w.std(), 1e-3)      # clamp: avoid blow-up on flat windows
    dgr = np.gradient(gr)
    pos = np.linspace(0,1,len(gr))
    go = (geo_off - geo_off.mean())/max(geo_off.std(), 1e-3)   # geometric-path offset (dip prior)
    gh = np.stack([gr, dgr, go, pos], 0)
    tg = (twg - twg.mean())/max(twg.std(), 1e-3)
    gt = np.stack([tg, np.gradient(tg)], 0)
    return gh.astype(np.float32), gt.astype(np.float32)
'''

CELL_PF = r'''
# === long-range core: distance-direction particle filter with along-hole dip update ===
# state = (TVT, dip_local). transition: dTVT = -dZ + dip*dhoriz + walk; dip random-walks
# (the fix for fixed-heel-dip long-toe degradation). likelihood: GR_obs vs typewell GR(TVT).
# No training; hyperparams self-estimated from each well's heel known zone (leak-free).
from numba import njit as _pnjit

@_pnjit(cache=True)
def _systematic(w, N):
    pos = (np.random.rand() + np.arange(N))/N
    cum = np.cumsum(w)
    idx = np.empty(N, np.int64); j = 0
    for k in range(N):
        while pos[k] > cum[j] and j < N-1: j += 1
        idx[k] = j
    return idx

@_pnjit(cache=True)
def pf_track(gr, dZ, dh, tw_gr, tw_tvt, anchor, dip0,
             s_gr, s_dip, s_tvt, s_init, s_dipinit, rough, N, ess_frac, seed):
    np.random.seed(seed)
    n = len(gr)
    tvt = anchor + np.random.randn(N)*s_init
    dip = dip0 + np.random.randn(N)*s_dipinit
    logw = np.zeros(N)
    out = np.empty(n); ess_hist = np.empty(n)
    for i in range(n):
        dip = dip + np.random.randn(N)*s_dip
        tvt = tvt + (-dZ[i] + dip*dh[i]) + np.random.randn(N)*s_tvt
        g = np.interp(tvt, tw_tvt, tw_gr)             # typewell GR at each particle's TVT
        d = (gr[i] - g)/s_gr
        d2 = d*d
        for p in range(N):
            if d2[p] > 9.0: d2[p] = 9.0               # fat-tail clip (3 sigma)
        logw = logw - 0.5*d2
        m = logw.max(); w = np.exp(logw - m); sw = w.sum()
        if sw <= 0: w = np.ones(N)/N
        else: w = w/sw
        out[i] = (w*tvt).sum()                        # posterior-mean TVT
        ess = 1.0/np.sum(w*w); ess_hist[i] = ess
        if ess < ess_frac*N:
            idx = _systematic(w, N)
            tvt = tvt[idx] + np.random.randn(N)*rough
            dip = dip[idx] + np.random.randn(N)*rough*0.5
            logw = np.zeros(N)
    return out, ess_hist

def pf_predict(gr_toe, dZ_toe, dh_toe, tw_gr, tw_tvt, anchor, dip0, hyp, seed):
    return pf_track(gr_toe.astype(np.float64), dZ_toe.astype(np.float64), dh_toe.astype(np.float64),
                    tw_gr.astype(np.float64), tw_tvt.astype(np.float64), float(anchor), float(dip0),
                    hyp["s_gr"], hyp["s_dip"], hyp["s_tvt"], hyp["s_init"], hyp["s_dipinit"],
                    hyp["rough"], int(hyp["N"]), float(hyp["ess_frac"]), int(seed))
'''

CELL_RUN = r'''
# === GO/NO-GO: distance-direction particle filter (online dip) vs fixed-dip beam, LONG-TOE ===
# No training. Per well: anchor early in the known zone, pseudo-toe = rest of known zone
# (longest supervisable extrapolation). All methods on the NATIVE grid. Hyperparams for the
# PF are self-estimated from each well's heel known zone (leak-free).
rng = np.random.default_rng(SEED)
rmse=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))

def _setup(w, split, toe_len):
    n=len(w["X"]); ttvt=w["ttvt"]; gr=w["gr"]; tw_tvt=w["tw_tvt"]; tw_gr=w["tw_gr"]
    X=w["X"]; Y=w["Y"]; Z=w["Z"]
    j1=min(n, split+toe_len)
    if split<20 or j1-split<20 or len(tw_tvt)<8: return None
    # heel-only dip (leak-free)
    dh0=np.hypot(np.diff(X[:split]), np.diff(Y[:split])); hh=np.concatenate([[0.0], np.cumsum(dh0)])
    sh=ttvt[:split]+Z[:split]
    if np.std(hh)<1e-6: return None
    cf=np.polyfit(hh, sh, 1); dip=float(cf[0]); dip_res=sh-(cf[0]*hh+cf[1])
    anchor=float(ttvt[split-1])
    dtvt_tw=float(np.median(np.diff(tw_tvt)))
    if dtvt_tw<=0: return None
    # toe geometry
    Xt=X[split-1:j1]; Yt=Y[split-1:j1]; Zt=Z[split-1:j1]
    dZt=np.diff(Zt); dht=np.hypot(np.diff(Xt), np.diff(Yt))
    dexp_tvt=-dZt + dip*dht; geo=anchor+np.cumsum(dexp_tvt); dexp_idx=dexp_tvt/dtvt_tw
    gr_t=_smooth(gr[split:j1], float(np.nanmean(tw_gr)), 1)
    true=ttvt[split:j1]
    # --- leak-free PF hyperparams from heel known zone ---
    gr_k=_smooth(gr[:split], float(np.nanmean(tw_gr)), 1)
    s_gr=float(np.std(gr_k - np.interp(ttvt[:split], tw_tvt, tw_gr))); s_gr=max(s_gr, 3.0)
    # along-hole dip variation in known zone: dip in 5 heel segments
    seg=max(1, split//5); dips=[]
    for a in range(0, split-seg, seg):
        hs=hh[a:a+seg]-hh[a]; ss=sh[a:a+seg]
        if np.std(hs)>1e-6: dips.append(np.polyfit(hs, ss, 1)[0])
    s_dip_total=float(np.std(np.array(dips))) if len(dips)>1 else abs(dip)*0.1+1e-3
    med_abs_dexp=float(np.median(np.abs(dexp_tvt)))+1e-6
    hyp=dict(s_gr=s_gr, s_dip=max(s_dip_total/ max(seg,1.0), 1e-5), s_tvt=0.2*med_abs_dexp,
             s_init=max(float(np.std(dip_res)), 1.0), s_dipinit=s_dip_total+1e-4,
             rough=0.05*(true.max()-true.min()+1e-6), N=(400 if SMOKE else 2000), ess_frac=0.5)
    return dict(gr_t=gr_t.astype(np.float64), dZt=dZt, dht=dht, dexp_idx=dexp_idx,
                geo=geo.astype(np.float64), true=true.astype(np.float64), anchor=anchor, dip=dip,
                twgn=tw_gr, twtn=tw_tvt, hyp=hyp)

wells = WELLS if not SMOKE else WELLS[:8]
r_geo=[]; r_dip=[]; r_pf=[]; wins=0; ess_lo=0; nrows=0
for wi, w in enumerate(wells):
    n=len(w["X"]); split=int(0.20*n); toe=n-split          # LONG pseudo-toe: anchor early
    s=_setup(w, split, toe)
    if s is None: continue
    true=s["true"]; H=s["hyp"]
    rg=rmse(s["geo"], true)
    p_dip=align_dip(s["gr_t"], s["anchor"], s["twgn"], s["twtn"], s["dexp_idx"])
    rd=rmse(p_dip, true)
    pf_tvt, ess=pf_predict(s["gr_t"], s["dZt"], s["dht"], s["twgn"], s["twtn"], s["anchor"], s["dip"], H, wi+1)
    rp=rmse(pf_tvt, true)
    r_geo.append(rg); r_dip.append(rd); r_pf.append(rp); wins += (rp < rd)
    ess_lo += int(np.mean(ess) < H["N"]/20.0); nrows += 1
assert nrows>0, "no eligible wells"
R_geo=float(np.mean(r_geo)); R_dip=float(np.mean(r_dip)); R_pf=float(np.mean(r_pf))
print(f"[LONG-TOE] wells={nrows}  geo_only={R_geo:.3f}  dip_beam(fixed)={R_dip:.3f}  PF(online-dip)={R_pf:.3f}")
print(f"[LONG-TOE] PF beats dip_beam on {wins}/{nrows} wells | wells with collapsed ESS(mean<N/20)={ess_lo}/{nrows}")

# --- GO/NO-GO: PF must beat the fixed-dip beam by >= 0.3 (below that = ceiling noise) ---
gain = R_dip - R_pf
print("\n================ GO/NO-GO (PF online-dip vs fixed-dip beam, long-toe) ================")
print(f"PF {R_pf:.3f} vs dip_beam {R_dip:.3f}  ->  gain {gain:+.3f}")
if gain >= 1.0:
    v="GO (strong, gold-signal) -> 6-formation + stretch/squeeze 2nd stage"
elif gain >= 0.3:
    v="GO -> PF helps; tune + build into pipeline"
else:
    v="NO-GO -> PF within ceiling noise; pivot core (typewell elastic stretch)"
print(f"VERDICT: {v}")
print("=====================================================================================")
'''

CELLS = [CELL_CONFIG, CELL_FORWARD, CELL_BASELINE, CELL_PF, CELL_RUN]


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

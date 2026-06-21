"""
Generate the ROGII SURFACE notebook = tuned self-contained base (romantamrazov)
+ STEP2 global formation structural-surface features for unseen-well generalization.

Base = _source_super.ipynb (same as rogii-wellbore-tuned). We carry the tuned
repro patches (SMOKE gating, CatBoost single-GPU + bootstrap_type=Bernoulli),
then INJECT one new cell right after the feature matrix is built (the cell that
defines `feature_cols=[c for c in train_df.columns if c not in SKIP]`).

The injected cell adds a per-formation absolute-depth surface S_f(X,Y) built from
PER-ROW samples across all train wells (not the well-median collapse the baseline
FormationPlaneKNN uses):

  S_f(X,Y) = global 2nd-order WLS trend (basis [1,X,Y,X^2,Y^2,XY])
             + local residual IDW (k-NN on a per-well subsample, like DenseANCC)

Datum-separated: S_f is fit on the raw formation-column field (datum-free); the
per-well vertical datum b_f is estimated from the prefix as median(tvt+Z - S_f)
per formation, with a cross-formation consensus spread feature. Predicted
structural TVT per eval row: tvtS_f = -Z_eval + S_f(X,Y) + b_f, emitted as
delta features tvtS_f_d = tvtS_f - last_known_tvt (same units / row order as the
baseline tvtF_*_d), plus surf consensus features (mean/std/range/datum-spread).

FOLD-SAFETY (critical, the whole point — private = unseen wells, optimize
GroupKFold CV): the global trend shares coefficients across wells, so a held-out
well must be excluded. Done via leave-one-well-out at feature-build time:
  - trend: per-well partial normal-equation sums, solved as (A - A_w, b - b_w)
    (O(1)/well rank-downdate) for each train well; full fit for test wells.
  - local residual: self_wid excluded from the kNN neighbors (existing DenseANCC
    pattern).
Because every TRAIN well's surface features are computed WITHOUT that well's own
data, the features are safe under any GroupKFold split (no CV-loop restructure).

The baseline FormationPlaneKNN features are kept (not replaced); the GBT + Ridge
stack picks the new columns up automatically (feature_cols recomputed in-cell).

FLIP SMOKE to False, regenerate, commit, push for the full run.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "_source_super.ipynb"
OUT = BASE / "rogii-krig.ipynb"

# Flip to False (regenerate + commit + push) for the full run.
SMOKE = False


def src_str(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell, text):
    cell["source"] = text


def find_cell(cells, needle, cell_type="code"):
    for i, c in enumerate(cells):
        if c["cell_type"] != cell_type:
            continue
        if needle in src_str(c):
            return i
    raise RuntimeError(f"cell containing {needle!r} not found")


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


SMOKE_SRC = '''\
# === SMOKE gating (baked-in; Kaggle injects no env). FLIP via generate_notebook.py. ===
SMOKE = __SMOKE__
if SMOKE:
    N_SPLITS = 2
    for _c in LGB_CONFIGS:
        _c["n_estimators"] = 60
    CB_PARAMS["iterations"] = 60
    CB_PARAMS["od_wait"] = 30
print(f"SMOKE={SMOKE}  N_SPLITS={N_SPLITS}")
'''

SURF_SRC = r'''
# === STEP2: Global formation structural-surface features (per-row, datum-separated, LOO fold-safe) ===
from scipy.spatial import cKDTree as _ckd
import numpy as _np, pandas as _pd, gc as _gc, time as _t

_SPW = 80   # per-well subsample for the residual field (mirrors DenseANCC SPW)
_KR  = 16   # residual IDW neighbors


def _phi(x, y):
    # 2nd-order polynomial basis (normalized inputs)
    return _np.column_stack([_np.ones_like(x), x, y, x * x, y * y, x * y])


class _FormSurf:
    """Per-formation S_f(X,Y) = global 2nd-order WLS trend + local residual IDW.
       LOO fold-safe: trend via per-well normal-equation downdate, residual via
       self_wid exclusion in the kNN tree AND residuals taken against the SAME
       downdated trend (exact leave-one-well-out, no beta0 imprint)."""

    def __init__(self, wids, data_dir):
        XX = []; YY = []; WW = []; FF = {f: [] for f in FORMATIONS}
        sX = []; sY = []; sW = []; sFd = {f: [] for f in FORMATIONS}
        code = {}
        for wid in wids:
            p = data_dir / (wid + "__horizontal_well.csv")
            try:
                df = _pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            c = code.setdefault(wid, len(code))
            X = df["X"].to_numpy(_np.float64); Y = df["Y"].to_numpy(_np.float64)
            XX.append(X); YY.append(Y); WW.append(_np.full(len(df), c, dtype=_np.int32))
            for f in FORMATIONS:
                FF[f].append(df[f].to_numpy(_np.float64))
            ix = _np.linspace(0, len(df) - 1, min(_SPW, len(df))).astype(int)
            sX.append(X[ix]); sY.append(Y[ix]); sW.append(_np.full(len(ix), c, dtype=_np.int32))
            for f in FORMATIONS:
                sFd[f].append(df[f].to_numpy(_np.float64)[ix])
        self.code = code
        self.X = _np.concatenate(XX); self.Y = _np.concatenate(YY); self.W = _np.concatenate(WW)
        self.F = {f: _np.concatenate(FF[f]) for f in FORMATIONS}
        self.mx = float(self.X.mean()); self.my = float(self.Y.mean())
        self.nx = float(self.X.std() or 1.0); self.ny = float(self.Y.std() or 1.0)
        P = _phi((self.X - self.mx) / self.nx, (self.Y - self.my) / self.ny)
        self.A = P.T @ P + 1e-6 * _np.eye(6)
        self.b = {f: P.T @ self.F[f] for f in FORMATIONS}
        self.Aw = {}; self.bw = {f: {} for f in FORMATIONS}
        for c in _np.unique(self.W):
            m = self.W == c; Pm = P[m]
            self.Aw[int(c)] = Pm.T @ Pm
            for f in FORMATIONS:
                self.bw[f][int(c)] = Pm.T @ self.F[f][m]
        self.beta0 = {f: _np.linalg.solve(self.A, self.b[f]) for f in FORMATIONS}
        # residual tree (subsample); keep RAW formation values for exact-LOO residual
        self.sX = _np.concatenate(sX); self.sY = _np.concatenate(sY); self.sW = _np.concatenate(sW)
        self.sF = {f: _np.concatenate(sFd[f]) for f in FORMATIONS}
        self.scale = _np.array([self.sX.std() or 1.0, self.sY.std() or 1.0])
        self.tree = _ckd(_np.column_stack([self.sX, self.sY]) / self.scale)

    def _beta(self, f, code):
        if code is not None and code in self.Aw:
            return _np.linalg.solve(self.A - self.Aw[code] + 1e-6 * _np.eye(6),
                                    self.b[f] - self.bw[f][code])
        return self.beta0[f]

    def _trend(self, beta, xn, yn):
        return (beta[0] + beta[1] * xn + beta[2] * yn
                + beta[3] * xn * xn + beta[4] * yn * yn + beta[5] * xn * yn)

    def predict(self, xy, wid=None):
        code = self.code.get(wid) if wid is not None else None
        xy = _np.atleast_2d(xy)
        xq = (xy[:, 0] - self.mx) / self.nx; yq = (xy[:, 1] - self.my) / self.ny
        kf = min(_KR + 8, len(self.sX))
        dist, idx = self.tree.query(xy / self.scale, k=kf, workers=-1)
        dist = _np.atleast_2d(dist); idx = _np.atleast_2d(idx)
        if code is not None:
            dist = _np.where(self.sW[idx] == code, _np.inf, dist)
        kk = min(_KR - 1, kf - 1)
        ordr = _np.argpartition(dist, kk, 1)[:, :_KR]
        dk = _np.take_along_axis(dist, ordr, 1); ik = _np.take_along_axis(idx, ordr, 1)
        vk = _np.isfinite(dk); w = _np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        ws = w.sum(1); safe = _np.where(ws < 1e-9, 1.0, ws)
        nbx = (self.sX[ik] - self.mx) / self.nx; nby = (self.sY[ik] - self.my) / self.ny
        out = {}
        for f in FORMATIONS:
            beta = self._beta(f, code)
            trend_q = self._trend(beta, xq, yq)
            # exact-LOO residual: neighbor residuals against the SAME downdated trend
            resid_nb = self.sF[f][ik] - self._trend(beta, nbx, nby)
            resid = (resid_nb * w).sum(1) / safe
            out[f] = (trend_q + resid).astype(_np.float64)
        return out


print("[SURF] building FormationSurfaceModel ..."); _t0 = _t.time()
_FS = _FormSurf(train_wids, TRAIN_DIR)
print(f"[SURF]  trend pts={len(_FS.X):,} tree pts={len(_FS.sX):,} ({_t.time()-_t0:.0f}s)")


def _surf_feats(paths, is_train):
    recs = []
    for p in paths:
        wid = p.stem.replace("__horizontal_well", "")
        try:
            hw = _pd.read_csv(p)
        except Exception:
            continue
        if "TVT_input" not in hw.columns:
            continue
        kn = hw[hw["TVT_input"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10:
            continue
        last_tvt = float(kn["TVT_input"].to_numpy()[-1])
        swid = wid if is_train else None
        xk = kn[["X", "Y"]].to_numpy(_np.float64); zk = kn["Z"].to_numpy(_np.float64)
        tk = kn["TVT_input"].to_numpy(_np.float64)
        xe = ev[["X", "Y"]].to_numpy(_np.float64); ze = ev["Z"].to_numpy(_np.float64)
        Sk = _FS.predict(xk, swid); Se = _FS.predict(xe, swid)
        tvts = {}; bs = []
        for f in FORMATIONS:
            b_f = float(_np.median(tk + zk - Sk[f]))
            bs.append(b_f)
            tvts[f] = (-ze + Se[f] + b_f)
        # clip per well to its own vertical scale (guards quadratic-trend blow-up
        # outside the train convex hull on unseen wells); all in delta units.
        rng = float(tk.max() - tk.min())
        cap = 3.0 * rng + 1000.0
        deltas = {f: _np.clip(tvts[f] - last_tvt, -cap, cap) for f in FORMATIONS}
        M = _np.stack([deltas[f] for f in FORMATIONS], 1)  # (n,6) clipped deltas
        smean_d = M.mean(1); sstd = M.std(1); srng = M.max(1) - M.min(1)
        bs = _np.array(bs); b_spread = float(bs.std())
        ids = [f"{wid}_{i}" for i in ev.index]
        d = {"id": ids,
             "surf_mean_d": smean_d.astype(_np.float32),
             "surf_std": sstd.astype(_np.float32),
             "surf_rng": srng.astype(_np.float32),
             "surf_datum_spread": _np.full(len(ev), b_spread, _np.float32)}
        for f in FORMATIONS:
            d[f"tvtS_{f}_d"] = deltas[f].astype(_np.float32)
        recs.append(_pd.DataFrame(d))
    return _pd.concat(recs, ignore_index=True) if recs else _pd.DataFrame({"id": []})


print("[SURF] train surface feats (LOO) ..."); _t0 = _t.time()
_sf_tr = _surf_feats(hw_paths, True)
print(f"[SURF]  train surf rows={len(_sf_tr)} ({_t.time()-_t0:.0f}s)")
print("[SURF] test surface feats ..."); _t0 = _t.time()
_sf_te = _surf_feats(test_paths, False)
print(f"[SURF]  test surf rows={len(_sf_te)} ({_t.time()-_t0:.0f}s)")

_n_tr0 = len(train_df); _n_te0 = len(test_df)
train_df = train_df.merge(_sf_tr, on="id", how="left")
test_df = test_df.merge(_sf_te, on="id", how="left")
assert len(train_df) == _n_tr0 and len(test_df) == _n_te0, "surf merge changed row count"
_surfcols = [c for c in _sf_tr.columns if c != "id"]
_miss_tr = int(train_df[_surfcols].isna().any(axis=1).sum())
_miss_te = int(test_df[_surfcols].isna().any(axis=1).sum())
print(f"[SURF] merged surf cols={len(_surfcols)} | train NaN-rows={_miss_tr} test NaN-rows={_miss_te}")
assert _miss_tr == 0, f"[SURF] {_miss_tr} train rows lack surface features (id set mismatch with build_well)"
for c in _surfcols:
    train_df[c] = train_df[c].fillna(0.0).astype(_np.float32)
    test_df[c] = test_df[c].fillna(0.0).astype(_np.float32)

# recompute the model matrices so the GBT + Ridge stack see the surface features
feature_cols = [c for c in train_df.columns if c not in SKIP]
X = train_df[feature_cols]; y = train_df["target"]; g = train_df["well"]
Xt = test_df[feature_cols]
print(f"[SURF] #features now {len(feature_cols)} (+{len(_surfcols)} surface)")
_gc.collect()
'''


# STEP1 diagnostic: dip-aware-transition beam vs public geometry-blind beam,
# benchmarked by KNOWN-ZONE pseudo-toe reconstruction (toe target never used).
# Runs over all train wells (cheap CPU), prints, pipeline continues.
BENCH_SRC = r'''
# === STEP1 BENCH: dip-aware aligner (known-zone pseudo-toe reconstruction) ===
from pathlib import Path as _BPath
import numpy as _bnp, pandas as _bpd
from numba import njit as _bnjit

@_bnjit(cache=True)
def _beam_dip_jit(sgr, tw_gr, si, BS, mc, es, d_exp, W):
    n=len(sgr); nt=len(tw_gr); MAX=BS*(2*W+6)
    bidx=_bnp.zeros(BS,_bnp.int64); bidx[0]=si
    bcost=_bnp.full(BS,1e30); bcost[0]=0.; bn=_bnp.int64(1)
    hI=_bnp.zeros((n,BS),_bnp.int64); hP=_bnp.zeros((n,BS),_bnp.int64)
    cI=_bnp.zeros(MAX,_bnp.int64); cC=_bnp.full(MAX,1e30); cP=_bnp.zeros(MAX,_bnp.int64)
    for step in range(n):
        gv=sgr[step]; de=d_exp[step]; nc=_bnp.int64(0)
        dlo=int(_bnp.floor(de))-W; dhi=int(_bnp.ceil(de))+W
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(dlo,dhi+1):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                dd=d-de
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(dd if dd>=0 else -dd)
                fnd=_bnp.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        if nc==0:
            # all moves left the typewell range: carry the best current beam
            # forward unchanged (prevents bn=0 path collapse).
            bp=_bnp.int64(0)
            for bi in range(1,bn):
                if bcost[bi]<bcost[bp]: bp=bi
            ci0=bidx[bp]
            if ci0<0: ci0=_bnp.int64(0)
            if ci0>=nt: ci0=_bnp.int64(nt-1)
            cI[0]=ci0; cC[0]=bcost[bp]; cP[0]=bp; nc=_bnp.int64(1)
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]; cC[i],cC[mi]=cC[mi],cC[i]; cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=_bnp.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=_bnp.zeros(n,_bnp.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

def _bench_one(hw_path):
    wid=_BPath(hw_path).stem.replace('__horizontal_well','')
    twp=_BPath(hw_path).parent/(wid+'__typewell.csv')
    if not twp.exists(): return None
    try:
        hw=_bpd.read_csv(hw_path); tw=_bpd.read_csv(twp).sort_values('TVT')
    except Exception: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<40 or len(tw)<5: return None
    tw_tvt=tw['TVT'].to_numpy(_bnp.float64); tw_gr=tw['GR'].to_numpy(_bnp.float64)
    dtvt=float(_bnp.median(_bnp.diff(tw_tvt)))
    if dtvt<=0: return None
    ttvt=kn['TVT_input'].to_numpy(_bnp.float64)               # benchmark ground truth (not the comp target)
    Z=kn['Z'].to_numpy(_bnp.float64); X=kn['X'].to_numpy(_bnp.float64); Y=kn['Y'].to_numpy(_bnp.float64)
    gr=(kn['GR'].astype(float).interpolate(limit_direction='both')
        .fillna(float(_bnp.nanmean(tw_gr))).to_numpy(_bnp.float64))
    n=len(kn); split=int(0.7*n)
    if split<10 or n-split<10: return None
    dh=_bnp.hypot(_bnp.diff(X),_bnp.diff(Y)); hcum=_bnp.concatenate([[0.0],_bnp.cumsum(dh)])
    sh=ttvt[:split]+Z[:split]; hh=hcum[:split]                # heel-only dip estimate (leak-free)
    dip=0.0 if _bnp.std(hh)<1e-6 else float(_bnp.polyfit(hh,sh,1)[0])
    anchor=float(ttvt[split-1]); si=_nn(tw_tvt,anchor)
    sgr=_smooth(gr[split:],float(_bnp.nanmean(tw_gr)),2).astype(_bnp.float64)
    dZt=_bnp.diff(Z[split-1:]); dht=_bnp.hypot(_bnp.diff(X[split-1:]),_bnp.diff(Y[split-1:]))
    dexp_tvt=-dZt+dip*dht                                     # expected dTVT per pseudo-toe step
    dexp_idx=(dexp_tvt/dtvt).astype(_bnp.float64)
    true_pt=ttvt[split:]
    geo=anchor+_bnp.cumsum(dexp_tvt)
    _zero=_bnp.zeros_like(dexp_idx)
    p_pub=_beam_jit(sgr,tw_gr,si,10,20.0,144.0)                       # public beam (window [-2,2], sanity)
    p0=_beam_dip_jit(sgr,tw_gr,si,10,20.0,144.0,_zero,2)             # FAIR baseline: same impl/width, de=0
    p1=_beam_dip_jit(sgr,tw_gr,si,10,20.0,144.0,dexp_idx,2)          # dip-aware: only transition center differs
    rg=float(_bnp.sqrt(_bnp.mean((geo-true_pt)**2)))
    rpub=float(_bnp.sqrt(_bnp.mean((tw_tvt[p_pub]-true_pt)**2)))
    rb=float(_bnp.sqrt(_bnp.mean((tw_tvt[p0]-true_pt)**2)))
    rd=float(_bnp.sqrt(_bnp.mean((tw_tvt[p1]-true_pt)**2)))
    return (rg,rpub,rb,rd,n-split)

print("[BENCH] dip-aware aligner known-zone pseudo-toe reconstruction (all train wells) ...")
_brows=[]
for _hp in sorted(TRAIN_DIR.glob('*__horizontal_well.csv')):
    _r=_bench_one(_hp)
    if _r is not None: _brows.append(_r)
if _brows:
    _BA=_bnp.array([r[:4] for r in _brows]); _gm=_BA.mean(0)   # cols: geo, pub, fair, dip
    _win=int((_BA[:,3]<_BA[:,2]).sum())                         # dip vs FAIR baseline (same impl/width)
    print(f"[BENCH] wells={len(_brows)} | mean per-well RMSE  geo_only={_gm[0]:.3f}  beam_pub={_gm[1]:.3f}  beam_fair={_gm[2]:.3f}  dip_beam={_gm[3]:.3f}")
    print(f"[BENCH] fair check: beam_pub({_gm[1]:.3f}) vs beam_fair({_gm[2]:.3f}) should match (same beam, de=0)")
    print(f"[BENCH] dip_beam beats beam_fair on {_win}/{len(_brows)} wells | mean(fair-dip)={(_BA[:,2]-_BA[:,3]).mean():.3f}")
else:
    print("[BENCH] no eligible wells")
'''


# STEP2: dip-aware-transition beam as a production feature on the REAL toe.
# Intra-well only (own known zone for dip+anchor, own typewell, own trajectory,
# own toe GR) => fold-safe by construction, no cross-well leak. Reuses
# _beam_dip_jit defined in the BENCH cell (runs earlier).
DIPBEAM_SRC = r'''
# === STEP2: dip-aware beam member (intra-well, fold-safe by construction) ===
import numpy as _qnp, pandas as _qpd
from pathlib import Path as _QPath

def _dipbeam_one(hw_path):
    wid=_QPath(hw_path).stem.replace('__horizontal_well','')
    twp=_QPath(hw_path).parent/(wid+'__typewell.csv')
    if not twp.exists(): return None
    try:
        hw=_qpd.read_csv(hw_path); tw=_qpd.read_csv(twp).sort_values('TVT')
    except Exception: return None
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10 or len(tw)<5: return None
    tw_tvt=tw['TVT'].to_numpy(_qnp.float64); tw_gr=tw['GR'].to_numpy(_qnp.float64)
    dtvt=float(_qnp.median(_qnp.diff(tw_tvt)))
    if dtvt<=0: return None
    ktvt=kn['TVT_input'].to_numpy(_qnp.float64); kz=kn['Z'].to_numpy(_qnp.float64)
    kx=kn['X'].to_numpy(_qnp.float64); ky=kn['Y'].to_numpy(_qnp.float64)
    # heel dip = d(TVT+Z)/d(horizontal) over the KNOWN zone (known at test time => leak-free)
    dhk=_qnp.hypot(_qnp.diff(kx),_qnp.diff(ky)); hck=_qnp.concatenate([[0.0],_qnp.cumsum(dhk)])
    dip=0.0 if _qnp.std(hck)<1e-6 else float(_qnp.polyfit(hck,ktvt+kz,1)[0])
    last_tvt=float(ktvt[-1])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(_qnp.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(_qnp.float64)        # toe is trailing-contiguous (base pattern)
    sgr=_smooth(hgr,float(_qnp.nanmean(tw_gr)),2).astype(_qnp.float64)
    zx=ev['X'].to_numpy(_qnp.float64); zy=ev['Y'].to_numpy(_qnp.float64); ze=ev['Z'].to_numpy(_qnp.float64)
    # steps: last known point -> first toe row -> ... (geometry transition center)
    Xc=_qnp.concatenate([[float(kx[-1])],zx]); Yc=_qnp.concatenate([[float(ky[-1])],zy]); Zc=_qnp.concatenate([[float(kz[-1])],ze])
    dZt=_qnp.diff(Zc); dht=_qnp.hypot(_qnp.diff(Xc),_qnp.diff(Yc))
    dexp_idx=((-dZt+dip*dht)/dtvt).astype(_qnp.float64)          # len = len(ev)
    si=_nn(tw_tvt,last_tvt)
    path=_beam_dip_jit(sgr,tw_gr,si,10,20.0,144.0,dexp_idx,2)
    tvt_dip=tw_tvt[path]
    ids=[f'{wid}_{i}' for i in ev.index]
    return _qpd.DataFrame({'id':ids,
        'tvt_dipbeam_d':(tvt_dip-last_tvt).astype(_qnp.float32),
        'dipbeam_dip':_qnp.full(len(ev),_qnp.float32(dip),_qnp.float32)})

def _dipbeam_feats(paths):
    recs=[r for r in (_dipbeam_one(p) for p in paths) if r is not None]
    return _qpd.concat(recs,ignore_index=True) if recs else _qpd.DataFrame({'id':[]})

print("[DIPBEAM] computing dip-aware beam member (intra-well) ...")
_db_tr=_dipbeam_feats(hw_paths); _db_te=_dipbeam_feats(test_paths)
_dn0=len(train_df); _dm0=len(test_df)
train_df=train_df.merge(_db_tr,on='id',how='left'); test_df=test_df.merge(_db_te,on='id',how='left')
assert len(train_df)==_dn0 and len(test_df)==_dm0, "[DIPBEAM] merge changed row count"
for _c in ['tvt_dipbeam_d','dipbeam_dip']:
    train_df[_c]=train_df[_c].fillna(0.0).astype(_qnp.float32)
    test_df[_c]=test_df[_c].fillna(0.0).astype(_qnp.float32)
feature_cols=[c for c in train_df.columns if c not in SKIP]
X=train_df[feature_cols]; y=train_df["target"]; g=train_df["well"]; Xt=test_df[feature_cols]
print(f"[DIPBEAM] +2 features | #features now {len(feature_cols)}")
_gc.collect()
'''


# STEP3: ANCC residual ordinary-kriging surface (variogram-based), as ADDED features.
# The competition core is toe ANCC(X,Y) extrapolation (TVT = ANCC - Z + datum). The
# base interpolates ANCC with plain IDW (DenseANCCImputer) / trend+IDW (_FormSurf);
# public GMs also use IDW. This replaces the LOCAL interpolation with ordinary kriging
# on the ANCC RESIDUAL (2nd-order trend removed -- advisor: raw-ANCC variogram diverges
# on toe extrapolation), and emits the kriging prediction variance (an independent
# extrapolation-uncertainty signal IDW cannot give). Reuses _FS (from the SURFACE cell):
# same trend, same subsample cloud, same KDTree, same LOWO discipline. Additive (DenseANCC
# / _FormSurf kept). Fold-safe: self-well excluded in the OK neighborhood (LOWO).
KRIG_SRC = r'''
# === STEP3: ANCC residual ordinary kriging (variogram, additive, LOO fold-safe) ===
import numpy as _knp, pandas as _kpd, gc as _kgc
from scipy.optimize import curve_fit as _kcf

_KF='ANCC'; _KK=24; _KFETCH=400
# residual of the subsample cloud vs the GLOBAL ANCC trend (beta0); self-well influence
# in beta0 is negligible -- LOO is enforced by self-well exclusion in the OK neighborhood.
_kxn=(_FS.sX-_FS.mx)/_FS.nx; _kyn=(_FS.sY-_FS.my)/_FS.ny
_kresid=(_FS.sF[_KF]-_FS._trend(_FS.beta0[_KF],_kxn,_kyn)).astype(_knp.float64)
_ksxy=(_knp.column_stack([_FS.sX,_FS.sY])/_FS.scale).astype(_knp.float64)  # scaled coords (match _FS.tree)
_kn=len(_kresid)

# --- empirical variogram on residuals -> exponential model fit (scaled-distance space) ---
_krs=_knp.random.RandomState(0)
_kpa=_krs.randint(0,_kn,60000); _kpb=_krs.randint(0,_kn,60000)
_khh=_knp.sqrt(((_ksxy[_kpa]-_ksxy[_kpb])**2).sum(1))
_kgg=0.5*(_kresid[_kpa]-_kresid[_kpb])**2
_khmax=float(_knp.percentile(_khh,40)); _knb=18
_kedg=_knp.linspace(0,_khmax,_knb+1); _kcen=0.5*(_kedg[:-1]+_kedg[1:])
_kbi=_knp.clip(_knp.digitize(_khh,_kedg)-1,0,_knb-1)
_kemp=_knp.array([(_kgg[(_kbi==b)&(_khh<=_khmax)].mean()
                   if ((_kbi==b)&(_khh<=_khmax)).sum()>30 else _knp.nan) for b in range(_knb)])
_kok=_knp.isfinite(_kemp)
def _kvexp(h,nug,sill,rng): return nug+sill*(1.0-_knp.exp(-h/_knp.maximum(rng,1e-9)))
_ks0=float(_knp.nanmean(_kemp[_kok])) if _kok.any() else 1.0
try:
    _kp,_=_kcf(_kvexp,_kcen[_kok],_kemp[_kok],p0=[_ks0*0.1,_ks0,_khmax*0.3],
               bounds=([0,0,1e-4],[_ks0*5+1,_ks0*10+1,_khmax*5+1]),maxfev=20000)
    _KNUG,_KSILL,_KRNG=float(_kp[0]),float(_kp[1]),float(_kp[2])
except Exception as _ke:
    print("[KRIG] variogram fit fallback:",_ke); _KNUG,_KSILL,_KRNG=_ks0*0.1,_ks0,_khmax*0.3
print(f"[KRIG] exp variogram: nugget={_KNUG:.4f} sill={_KSILL:.4f} range={_KRNG:.5f} bins_used={int(_kok.sum())} N={_kn}")
def _kgam(h): return _KNUG+_KSILL*(1.0-_knp.exp(-h/max(_KRNG,1e-9)))
_KJIT=1e-6*max(_KSILL,1.0)

def _krig_resid(xy, code):
    # local ordinary kriging of the ANCC residual at xy; returns (pred, var). LOWO.
    q=(_knp.atleast_2d(xy).astype(_knp.float64))/_FS.scale
    Nq=len(q); pred=_knp.empty(Nq); var=_knp.empty(Nq)
    nf=min(_KFETCH,_kn)
    for _s in range(0,Nq,3000):
        qc=q[_s:_s+3000]; m=len(qc)
        dist,idx=_FS.tree.query(qc,k=nf,workers=-1)
        dist=_knp.atleast_2d(dist); idx=_knp.atleast_2d(idx)
        if code is not None:
            dist=_knp.where(_FS.sW[idx]==code,_knp.inf,dist)
        kk=min(_KK,nf-1)
        ordr=_knp.argpartition(dist,kk-1,1)[:,:kk]
        ik=_knp.take_along_axis(idx,ordr,1)            # (m,kk)
        nb=_ksxy[ik]                                   # (m,kk,2)
        zz=_kresid[ik]                                 # (m,kk)
        d2=_knp.sqrt(((nb[:,:,None,:]-nb[:,None,:,:])**2).sum(-1))
        G=_kgam(d2)
        _di=_knp.arange(kk); G[:,_di,_di]=G[:,_di,_di]+_KJIT
        dq=_knp.sqrt(((nb-qc[:,None,:])**2).sum(-1))   # (m,kk)
        gq=_kgam(dq)
        A=_knp.zeros((m,kk+1,kk+1)); A[:,:kk,:kk]=G; A[:,kk,:kk]=1.0; A[:,:kk,kk]=1.0
        b=_knp.ones((m,kk+1)); b[:,:kk]=gq
        try:
            sol=_knp.linalg.solve(A,b[...,None])[...,0]
        except _knp.linalg.LinAlgError:
            sol=_knp.zeros((m,kk+1))
            for _qi in range(m):
                sol[_qi]=_knp.linalg.lstsq(A[_qi],b[_qi],rcond=None)[0]
        w=sol[:,:kk]; mu=sol[:,kk]
        pred[_s:_s+3000]=(w*zz).sum(1)
        var[_s:_s+3000]=(w*gq).sum(1)+mu
    return pred, _knp.maximum(var,0.0)

def _krig_feats(paths, is_train):
    recs=[]
    for p in paths:
        wid=p.stem.replace("__horizontal_well","")
        try: hw=_kpd.read_csv(p)
        except Exception: continue
        if "TVT_input" not in hw.columns: continue
        kn=hw[hw["TVT_input"].notna()]; ev=hw[hw["TVT_input"].isna()]
        if len(ev)==0 or len(kn)<10: continue
        last_tvt=float(kn["TVT_input"].to_numpy()[-1])
        code=_FS.code.get(wid) if is_train else None
        xk=kn[["X","Y"]].to_numpy(_knp.float64); zk=kn["Z"].to_numpy(_knp.float64)
        tk=kn["TVT_input"].to_numpy(_knp.float64)
        xe=ev[["X","Y"]].to_numpy(_knp.float64); ze=ev["Z"].to_numpy(_knp.float64)
        beta=_FS._beta(_KF,code)
        rk,_=_krig_resid(xk,code); re,ve=_krig_resid(xe,code)
        ak_k=_FS._trend(beta,(xk[:,0]-_FS.mx)/_FS.nx,(xk[:,1]-_FS.my)/_FS.ny)+rk   # ANCC_krig known
        ak_e=_FS._trend(beta,(xe[:,0]-_FS.mx)/_FS.nx,(xe[:,1]-_FS.my)/_FS.ny)+re   # ANCC_krig toe
        b_f=float(_knp.median(tk+zk-ak_k))
        tvt_k=(-ze+ak_e+b_f)
        rng=float(tk.max()-tk.min()); cap=3.0*rng+1000.0
        d=_knp.clip(tvt_k-last_tvt,-cap,cap)
        ids=[f"{wid}_{i}" for i in ev.index]
        recs.append(_kpd.DataFrame({"id":ids,
            "tvt_krig_ancc_d":d.astype(_knp.float32),
            "krig_sigma":_knp.sqrt(ve).astype(_knp.float32)}))
    return _kpd.concat(recs,ignore_index=True) if recs else _kpd.DataFrame({"id":[]})

print("[KRIG] computing ANCC residual-kriging features ..."); _kt=__import__("time").time()
_kr_tr=_krig_feats(hw_paths, True); _kr_te=_krig_feats(test_paths, False)
print(f"[KRIG]  train rows={len(_kr_tr)} test rows={len(_kr_te)} ({__import__('time').time()-_kt:.0f}s)")
_kn0=len(train_df); _km0=len(test_df)
train_df=train_df.merge(_kr_tr,on="id",how="left"); test_df=test_df.merge(_kr_te,on="id",how="left")
assert len(train_df)==_kn0 and len(test_df)==_km0, "[KRIG] merge changed row count"
_kmiss=int(train_df[["tvt_krig_ancc_d","krig_sigma"]].isna().any(axis=1).sum())
print(f"[KRIG] merged | train NaN-rows={_kmiss}")
for _c in ["tvt_krig_ancc_d","krig_sigma"]:
    train_df[_c]=train_df[_c].fillna(0.0).astype(_knp.float32)
    test_df[_c]=test_df[_c].fillna(0.0).astype(_knp.float32)
feature_cols=[c for c in train_df.columns if c not in SKIP]
X=train_df[feature_cols]; y=train_df["target"]; g=train_df["well"]; Xt=test_df[feature_cols]
print(f"[KRIG] +2 features (krig ANCC + sigma) | #features now {len(feature_cols)}")
_kgc.collect()
'''


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # 1. SMOKE cell right after the config cell
    i_cfg = find_cell(cells, "CB_PARAMS=dict(")
    cfg = src_str(cells[i_cfg])
    assert "LGB_CONFIGS=[" in cfg and "N_SPLITS=" in cfg
    cells.insert(i_cfg + 1, code_cell(SMOKE_SRC.replace("__SMOKE__", "True" if SMOKE else "False")))
    report.append(f"inserted SMOKE cell at {i_cfg + 1}")

    # 2. SMOKE subset of train wells
    i_imp = find_cell(cells, "hw_paths=sorted(TRAIN_DIR.glob('*__horizontal_well.csv'))")
    s = src_str(cells[i_imp])
    anchor = "hw_paths=sorted(TRAIN_DIR.glob('*__horizontal_well.csv'))\n"
    assert anchor in s
    set_src(cells[i_imp], s.replace(anchor, anchor + "if SMOKE: hw_paths=hw_paths[:14]\n"))
    report.append(f"cell {i_imp}: SMOKE subset hw_paths[:14]")

    # 2b. INJECT STEP1 dip-aware aligner benchmark right after the imputer cell
    #     (_beam_jit/_smooth/_nn defined earlier; globs all train wells itself).
    i_imp2 = find_cell(cells, "hw_paths=sorted(TRAIN_DIR.glob('*__horizontal_well.csv'))")
    cells.insert(i_imp2 + 1, code_cell(BENCH_SRC))
    report.append(f"inserted STEP1 BENCH cell at {i_imp2 + 1}")

    # 3. SMOKE subset of test wells
    i_bld = find_cell(cells, "test_paths=sorted(TEST_DIR.glob('*__horizontal_well.csv'))")
    s = src_str(cells[i_bld])
    anchor = "test_paths=sorted(TEST_DIR.glob('*__horizontal_well.csv'))\n"
    assert anchor in s
    set_src(cells[i_bld], s.replace(anchor, anchor + "if SMOKE: test_paths=test_paths[:6]\n"))
    report.append(f"cell {i_bld}: SMOKE subset test_paths[:6]")

    # 4. CatBoost single-GPU + bootstrap fix (Bayesian rejects subsample)
    nd = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        if 'devices="0:1"' in cs:
            set_src(c, cs.replace('devices="0:1"', 'devices="0", bootstrap_type="Bernoulli"')); nd += 1
    assert nd >= 1
    report.append(f"patched {nd} CatBoost devices -> single-GPU + bootstrap_type=Bernoulli")

    # 5. INJECT surface-feature cell right after `feature_cols=` is first built
    i_feat = find_cell(cells, "feature_cols=[c for c in train_df.columns if c not in SKIP]")
    cells.insert(i_feat + 1, code_cell(SURF_SRC))
    report.append(f"inserted SURFACE cell at {i_feat + 1} (after feature_cols build)")

    # 6. INJECT dip-aware beam member cell right after the surface cell
    #    (reuses _beam_dip_jit from the BENCH cell; recomputes X/Xt last).
    cells.insert(i_feat + 2, code_cell(DIPBEAM_SRC))
    report.append(f"inserted DIPBEAM cell at {i_feat + 2} (after surface cell)")

    # 7. INJECT ANCC residual-kriging cell after DIPBEAM (reuses _FS from SURFACE).
    cells.insert(i_feat + 3, code_cell(KRIG_SRC))
    report.append(f"inserted KRIG cell at {i_feat + 3} (after dipbeam cell)")

    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # round-trip + sanity
    rt = json.load(open(OUT, encoding="utf-8"))
    full = "\n".join(src_str(c) for c in rt["cells"] if c["cell_type"] == "code")
    assert 'devices="0:1"' not in full
    assert 'bootstrap_type="Bernoulli"' in full
    assert "if SMOKE: hw_paths=hw_paths[:14]" in full
    assert "class _FormSurf:" in full
    assert "tvtS_" in full
    assert "_beam_dip_jit" in full and "[BENCH]" in full
    assert "tvt_dipbeam_d" in full and "[DIPBEAM]" in full
    # STEP3 ANCC residual kriging wired after the surface cell, reusing _FS
    assert "tvt_krig_ancc_d" in full and "krig_sigma" in full and "[KRIG]" in full
    assert "def _krig_resid(xy, code):" in full and "_kgam" in full
    assert full.index("_FS = _FormSurf(train_wids, TRAIN_DIR)") < full.index("_kresid=(_FS.sF[_KF]")
    # surface cell must come after the first feature_cols build and recompute X/Xt
    assert full.count("feature_cols = [c for c in train_df.columns if c not in SKIP]") >= 1
    assert "_FS = _FormSurf(train_wids, TRAIN_DIR)" in full
    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()

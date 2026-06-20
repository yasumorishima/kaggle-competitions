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
OUT = BASE / "rogii-medal.ipynb"

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


TCN_SRC = r'''
# === STEP2-b: TCN sequence model as a Ridge-stack member (per-row features incl. surface) ===
import torch, torch.nn as nn
torch.manual_seed(42); np.random.seed(42)
_TE = 2 if SMOKE else 40
_TPAT = 1 if SMOKE else 8
_TCH = 32 if SMOKE else 128
_TNB = 2 if SMOKE else 7
_TDROP, _TLR, _TWD, _TCLIP = 0.15, 5e-4, 1e-4, 1.0
_NSEED = 1 if SMOKE else 3
_tfeat = list(feature_cols)
print(f"[TCN] SMOKE={SMOKE} ep={_TE} ch={_TCH} nb={_TNB} nfeat={len(_tfeat)} seeds={_NSEED}")

_dev = "cpu"
try:
    if torch.cuda.is_available():
        _t = torch.zeros(8, device="cuda"); _ = float((_t + 1).sum().item()); _dev = "cuda"
        print("[TCN] GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("[TCN] GPU unusable -> CPU:", str(e)[:100])

_Xall = train_df[_tfeat].to_numpy(np.float32); _Xall[~np.isfinite(_Xall)] = np.nan
_mu = np.nanmean(_Xall, 0).astype(np.float32); _sd = np.nanstd(_Xall, 0).astype(np.float32); _sd[_sd < 1e-6] = 1.0
del _Xall; gc.collect()
_yt = train_df['target'].to_numpy(np.float32); _ymu = float(np.nanmean(_yt)); _ysd = float(np.nanstd(_yt)) or 1.0


def _tnorm(M):
    M = M.astype(np.float32, copy=True); M[~np.isfinite(M)] = np.nan; M = (M - _mu) / _sd
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


def _seqs(df, has_t):
    df = df.copy(); df['_ri'] = df['id'].str.rsplit('_', n=1).str[-1].astype(int)
    out = []
    for wid, gd in df.groupby('well', sort=False):
        gd = gd.sort_values('_ri')
        it = {'wid': wid, 'X': _tnorm(gd[_tfeat].to_numpy()), 'ids': gd['id'].to_numpy()}
        if has_t:
            it['t'] = (gd['target'].to_numpy(np.float32) - _ymu) / _ysd
        out.append(it)
    return out


_tr = _seqs(train_df, True); _te = _seqs(test_df, False)
_grp = np.array([s['wid'] for s in _tr])


def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class _Blk(nn.Module):
    def __init__(s, c, d, dr):
        super().__init__()
        s.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.n1 = _gn(c)
        s.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.n2 = _gn(c)
        s.a = nn.ReLU(); s.do = nn.Dropout(dr)

    def forward(s, x):
        y = s.do(s.a(s.n1(s.c1(x)))); y = s.n2(s.c2(y)); return s.a(x + y)


class _TCN(nn.Module):
    def __init__(s, ci, c, nb, dr):
        super().__init__()
        s.inp = nn.Conv1d(ci, c, 1)
        s.bl = nn.ModuleList([_Blk(c, 2 ** i, dr) for i in range(nb)])
        s.h = nn.Conv1d(c, 1, 1)

    def forward(s, x):
        x = s.inp(x)
        for b in s.bl:
            x = b(x)
        return s.h(x).squeeze(1)


def _hub(p, t, d=1.0):
    e = p - t; a = e.abs(); return torch.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d)).mean()


def _tx(s):
    return torch.tensor(s['X'].T[None], dtype=torch.float32, device=_dev)


def _train_one(_tri, _vai, _seed):
    torch.manual_seed(_seed); np.random.seed(_seed)
    _m = _TCN(len(_tfeat), _TCH, _TNB, _TDROP).to(_dev)
    _opt = torch.optim.Adam(_m.parameters(), lr=_TLR, weight_decay=_TWD)
    _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_TE)
    _best = 1e9; _bs = None; _bad = 0; _trl = np.array(_tri)
    for _ep in range(_TE):
        _m.train(); np.random.shuffle(_trl)
        for _j in _trl:
            s = _tr[_j]; x = _tx(s); t = torch.tensor(s['t'][None], dtype=torch.float32, device=_dev)
            _opt.zero_grad(); _l = _hub(_m(x), t); _l.backward()
            torch.nn.utils.clip_grad_norm_(_m.parameters(), _TCLIP); _opt.step()
        _sch.step()
        _m.eval(); _P = []; _T = []
        with torch.no_grad():
            for _j in _vai:
                s = _tr[_j]; _P.append(_m(_tx(s)).cpu().numpy()[0] * _ysd + _ymu); _T.append(s['t'] * _ysd + _ymu)
        _vr = float(np.sqrt(np.mean((np.concatenate(_P) - np.concatenate(_T)) ** 2)))
        if _vr < _best - 1e-4:
            _best = _vr; _bad = 0
            _bs = {k: v.detach().cpu().clone() for k, v in _m.state_dict().items()}
        else:
            _bad += 1
        if _bad >= _TPAT:
            break
    _m.load_state_dict(_bs); _m.eval(); return _m


_oof_id = {}; _test_sum = {}; _nf = 0; _fb = []
_idx = np.arange(len(_tr))
for _f, (_tri, _vai) in enumerate(GroupKFold(n_splits=N_SPLITS).split(_idx, groups=_grp)):
    _va_sum = {}; _te_fold = {}
    for _seed in range(_NSEED):
        _m = _train_one(_tri, _vai, 1000 * _seed + _f)
        with torch.no_grad():
            for _j in _vai:
                s = _tr[_j]; pr = _m(_tx(s)).cpu().numpy()[0] * _ysd + _ymu
                for _i, _id in enumerate(s['ids']):
                    _va_sum[_id] = _va_sum.get(_id, 0.0) + float(pr[_i]) / _NSEED
            for s in _te:
                pr = _m(_tx(s)).cpu().numpy()[0] * _ysd + _ymu
                for _i, _id in enumerate(s['ids']):
                    _te_fold[_id] = _te_fold.get(_id, 0.0) + float(pr[_i]) / _NSEED
    for _id, v in _va_sum.items():
        _oof_id[_id] = v
    for _id, v in _te_fold.items():
        _test_sum[_id] = _test_sum.get(_id, 0.0) + v
    _nf += 1
    _vp = []; _vt = []
    for _j in _vai:
        s = _tr[_j]; _vp.extend([_va_sum[i] for i in s['ids']]); _vt.extend(list(s['t'] * _ysd + _ymu))
    _fr = float(np.sqrt(np.mean((np.array(_vp) - np.array(_vt)) ** 2))); _fb.append(_fr)
    print(f"[TCN] fold{_f} toe RMSE={_fr:.4f}")
print("[TCN] CV toe RMSE mean =", float(np.mean(_fb)), "| folds", [round(b, 3) for b in _fb])

_tcn_oof = train_df['id'].map(_oof_id).to_numpy(np.float32)
_te_mean = {k: v / _nf for k, v in _test_sum.items()}
_tcn_test = test_df['id'].map(_te_mean).to_numpy(np.float32)
assert not np.isnan(_tcn_oof).any(), "TCN OOF unmapped"
assert not np.isnan(_tcn_test).any(), "TCN test unmapped"
results['tcn'] = {'oof': _tcn_oof, 'test': _tcn_test,
                  'rmse': float(np.sqrt(np.mean((_tcn_oof - y.values) ** 2)))}
print(f"[TCN] member added: OOF residual RMSE={results['tcn']['rmse']:.4f} | results keys={list(results.keys())}")
del _tr, _te; gc.collect()
'''


# STEP2-c: dip-aware-transition beam as an intra-well production feature on the
# REAL toe (own known zone -> dip+anchor, own typewell, own trajectory, own toe
# GR) => fold-safe by construction, no cross-well leak. Self-contained: bundles
# its own _beam_dip_jit (medal base has no BENCH cell); reuses _smooth/_nn from
# the base build_well. Inserted right after the SURFACE cell so feature_cols
# (and the later TCN's _tfeat) pick the 2 new columns up automatically.
DIPBEAM_SRC = r'''
# === STEP2-c: dip-aware beam member (intra-well, fold-safe by construction) ===
import numpy as _qnp, pandas as _qpd
from pathlib import Path as _QPath
from numba import njit as _qnjit

@_qnjit(cache=True)
def _beam_dip_jit(sgr, tw_gr, si, BS, mc, es, d_exp, W):
    n=len(sgr); nt=len(tw_gr); MAX=BS*(2*W+6)
    bidx=_qnp.zeros(BS,_qnp.int64); bidx[0]=si
    bcost=_qnp.full(BS,1e30); bcost[0]=0.; bn=_qnp.int64(1)
    hI=_qnp.zeros((n,BS),_qnp.int64); hP=_qnp.zeros((n,BS),_qnp.int64)
    cI=_qnp.zeros(MAX,_qnp.int64); cC=_qnp.full(MAX,1e30); cP=_qnp.zeros(MAX,_qnp.int64)
    for step in range(n):
        gv=sgr[step]; de=d_exp[step]; nc=_qnp.int64(0)
        dlo=int(_qnp.floor(de))-W; dhi=int(_qnp.ceil(de))+W
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(dlo,dhi+1):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                dd=d-de
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(dd if dd>=0 else -dd)
                fnd=_qnp.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        if nc==0:
            bp=_qnp.int64(0)
            for bi in range(1,bn):
                if bcost[bi]<bcost[bp]: bp=bi
            ci0=bidx[bp]
            if ci0<0: ci0=_qnp.int64(0)
            if ci0>=nt: ci0=_qnp.int64(nt-1)
            cI[0]=ci0; cC[0]=bcost[bp]; cP[0]=bp; nc=_qnp.int64(1)
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]; cC[i],cC[mi]=cC[mi],cC[i]; cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=_qnp.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=_qnp.zeros(n,_qnp.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

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
gc.collect()
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

    # 5b. INJECT dip-aware beam member cell right after the surface cell. Recomputes
    #     feature_cols/X/Xt so both GBT models and the later TCN (_tfeat) see it.
    cells.insert(i_feat + 2, code_cell(DIPBEAM_SRC))
    report.append(f"inserted DIPBEAM cell at {i_feat + 2} (after surface cell)")

    # 6. INJECT TCN as a Ridge-stack member: train it right before the stack is built
    i_stk = find_cell(cells, "Sx=np.column_stack([v['oof'] for v in results.values()])")
    st = src_str(cells[i_stk])
    anchor = "Sx=np.column_stack([v['oof'] for v in results.values()])"
    assert anchor in st
    set_src(cells[i_stk], st.replace(anchor, TCN_SRC.strip("\n") + "\n\n" + anchor))
    report.append(f"injected TCN member into cell {i_stk} (before Ridge stack)")

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
    # surface cell must come after the first feature_cols build and recompute X/Xt
    assert full.count("feature_cols = [c for c in train_df.columns if c not in SKIP]") >= 1
    assert "_FS = _FormSurf(train_wids, TRAIN_DIR)" in full
    # TCN member injected before the Ridge stack
    assert "results['tcn'] = {'oof': _tcn_oof" in full
    assert full.index("results['tcn']") < full.index("Sx=np.column_stack([v['oof'] for v in results.values()])")
    assert "class _TCN(nn.Module)" in full
    # DIPBEAM cell present, self-contained, and before the TCN reads feature_cols
    assert "def _dipbeam_one(hw_path):" in full
    assert "def _beam_dip_jit(" in full
    assert "tvt_dipbeam_d" in full and "dipbeam_dip" in full
    assert full.index("[DIPBEAM] +2 features") < full.index("_tfeat = list(feature_cols)")
    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()

"""
Patch the ROGII dualpipe notebook (Part1 pipeline A) to add a TCN sequence model
as a third, decorrelated Ridge-stack member.

WHY: private LB (the medal-deciding score) = base blend (sp45 + fleongg, CV ~9.2).
The same-well override only inflates the public score (a mirage). The gold lever is
to lower the *base* CV by adding a decorrelated member. The strongest proven
decorrelated member is a TCN sequence model (the identical injection took a sibling
kernel from public 10.224 -> 9.905 on the real LB).

WHAT: pipeline A builds a dict-of-arrays `oof_preds`/`test_preds`, converts them to
DataFrames, then a Ridge stack weights the members. We:
  1. insert a SMOKE gating cell right after the CFG config cell, and
  2. insert one TCN cell right BEFORE `oof_preds = pd.DataFrame(oof_preds)`
     (i.e. after the catboost loop, while oof_preds/test_preds are still dicts).
The TCN cell trains a GroupKFold OOF + test prediction and assigns
`oof_preds["tcn"]` / `test_preds["tcn"]`, so the Ridge stack picks it up.

NOTHING ELSE IS TOUCHED: Part2 (fleongg pipeline B), the 0.55/0.45 final blend,
the guarded contact override, and every other existing cell are left byte-identical.

This patcher reads `rogii-dualpipe.base.ipynb` (the pristine source) and writes
`rogii-dualpipe.ipynb` (the generated, TCN-injected notebook). Re-running is
idempotent w.r.t. the source because it always starts from the .base file.

SMOKE=True bakes a 2-epoch / 1-seed TCN for a fast smoke push.
Flip SMOKE=False, regenerate, commit, push for the full run.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "rogii-dualpipe.base.ipynb"   # pristine source (never overwritten)
OUT = BASE / "rogii-dualpipe.ipynb"        # generated notebook (TCN injected)

# Flip to False (regenerate + commit + push) for the full run.
# Generated as SMOKE=True so the coordinator can smoke-verify the surface/dipbeam +
# FORCE_RETRAIN path first, then flip to False for the full run.
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


# SMOKE flag cell (dualpipe has none). TCN_SRC already references SMOKE
# (_TE = 2 if SMOKE else 40, etc.), so this just defines it. GBT models are
# loaded from pretrained artifacts, so they stay fast regardless of SMOKE.
SMOKE_SRC = '''\
# === SMOKE gating (baked-in; Kaggle injects no env). FLIP via generate_notebook.py. ===
SMOKE = __SMOKE__
# Force GBT re-training: the pretrained artifacts were fit on the OLD feature set, so
# they MUST be retrained once we inject the surface + dip-beam columns (else the loaded
# models ignore / mismatch the new features). The train.csv fast-load is untouched; only
# the lightgbm/catboost load-vs-fit branches honor this flag.
FORCE_RETRAIN = True
print(f"SMOKE={SMOKE}  FORCE_RETRAIN={FORCE_RETRAIN}")
'''


# TCN sequence model member. Adapted from rogii-wellbore-medal/generate_notebook.py
# TCN_SRC (the proven 10.224 -> 9.905 injection) to dualpipe variable names:
#   feature_cols  -> features          (dualpipe's feature list, built in the load cell)
#   N_SPLITS      -> CFG.n_splits      (= 5; CFG.cv = GroupKFold(n_splits=n_splits))
#   train_df / test_df / y / g / GroupKFold : same names already exist in dualpipe
#   results['tcn'] = {...}             -> oof_preds["tcn"] / test_preds["tcn"]
#                                         (dict assignment; both are still dicts here)
#   added `import gc` (dualpipe top imports don't include gc; medal base did)
TCN_SRC = r'''
# === TCN sequence model as a Ridge-stack member (3rd, decorrelated; per-row features) ===
import torch, torch.nn as nn
import gc
torch.manual_seed(42); np.random.seed(42)
_TE = 2 if SMOKE else 40
_TPAT = 1 if SMOKE else 8
_TCH = 32 if SMOKE else 128
_TNB = 2 if SMOKE else 7
_TDROP, _TLR, _TWD, _TCLIP = 0.15, 5e-4, 1e-4, 1.0
_NSEED = 1 if SMOKE else 3
_tfeat = list(features)
print(f"[TCN] SMOKE={SMOKE} ep={_TE} ch={_TCH} nb={_TNB} nfeat={len(_tfeat)} seeds={_NSEED}")

_dev = "cpu"
try:
    if torch.cuda.is_available():
        _tg = torch.zeros(8, device="cuda"); _ = float((_tg + 1).sum().item()); _dev = "cuda"
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
for _f, (_tri, _vai) in enumerate(GroupKFold(n_splits=CFG.n_splits).split(_idx, groups=_grp)):
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
assert len(_tcn_oof) == len(train_df), "TCN OOF length != train_df"
assert len(_tcn_test) == len(test_df), "TCN test length != test_df"
assert not np.isnan(_tcn_oof).any(), "TCN OOF unmapped"
assert not np.isnan(_tcn_test).any(), "TCN test unmapped"

# --- standalone ABSOLUTE-TVT submission (consumed by the blend-level 3-way member) ---
# test_df['last_known_tvt'] is the RAW last-known TVT: build_well stores it via the
# constant broadcast `sc(last_tvt)` (np.full, NOT a scaler), so absolute = last_known + delta.
_lk = test_df['last_known_tvt'].to_numpy(np.float32)
assert len(_lk) == len(_tcn_test), "last_known_tvt / tcn_test length mismatch"
_tcn_abs = (_lk + _tcn_test).astype(float)
import os as _os
_wk = '/kaggle/working' if _os.path.exists('/kaggle/working') else '.'
pd.DataFrame({'id': test_df['id'].astype(str), 'tvt': _tcn_abs}).to_csv(f'{_wk}/tcn_submission.csv', index=False)
print(f"[TCN] wrote tcn_submission.csv abs (rows={len(_tcn_abs)})")

# oof_preds / test_preds are still dicts here -> add the member; the Ridge stack weights it.
# (kept independently of the standalone blend member above; A-ridge gain is free.)
oof_preds["tcn"] = _tcn_oof
test_preds["tcn"] = _tcn_test
print(f"[TCN] member added | OOF residual RMSE={float(np.sqrt(np.mean((_tcn_oof - y.values) ** 2))):.4f} | members={list(oof_preds.keys())}")
del _tr, _te; gc.collect()
'''


# === Surface + dip-beam structural features (proven on the standalone surface kernel:
#     SURF -10.3159->10.1929 = -0.123, DIPBEAM 10.1929->9.9780 = -0.215). Adapted from
#     rogii-wellbore-medal/generate_notebook.py SURF_SRC + DIPBEAM_SRC to dualpipe names:
#       feature_cols -> features ; Xt -> X_test ; SKIP -> {'well','id','target'}
#       TRAIN_DIR    -> defined in-cell as CFG.dataset_path / "train"
#     Already-present dualpipe symbols reused as-is (defined in the big helper cell that
#     runs before the load cell): FORMATIONS, train_wids, hw_paths, test_paths, _smooth,
#     _nn. Both cells read the RAW *__horizontal_well.csv (via train_wids/hw_paths/
#     test_paths), so they work even when train_df/test_df came from the train.csv
#     fast-load -- the new cols are id-merged onto train_df/test_df (medal pattern).
SURF_SRC = r'''
# === Global formation structural-surface features (per-row, datum-separated, LOO fold-safe) ===
from scipy.spatial import cKDTree as _ckd
import numpy as _np, pandas as _pd, gc as _gc, time as _t

TRAIN_DIR = CFG.dataset_path / "train"   # dualpipe path (medal used a TRAIN_DIR constant)
SKIP = {'well', 'id', 'target'}          # dualpipe feature-exclusion set

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

# recompute the model matrices so the GBT + Ridge stack (and the later TCN) see the surface features
features = [c for c in train_df.columns if c not in SKIP]
X = train_df[features]; y = train_df["target"]; g = train_df["well"]
X_test = test_df[features]
print(f"[SURF] #features now {len(features)} (+{len(_surfcols)} surface)")
_gc.collect()
'''


# dip-aware-transition beam as an intra-well production feature on the REAL toe (own
# known zone -> dip+anchor, own typewell, own trajectory, own toe GR) => fold-safe by
# construction, no cross-well leak. Self-contained: bundles its own _beam_dip_jit;
# reuses dualpipe's _smooth/_nn (defined in the big helper cell). Inserted right after
# the SURFACE cell so features (and the later TCN's _tfeat) pick the 2 new columns up.
DIPBEAM_SRC = r'''
# === dip-aware beam member (intra-well, fold-safe by construction) ===
import numpy as _qnp, pandas as _qpd, gc as _qgc
from pathlib import Path as _QPath
from numba import njit as _qnjit

SKIP = {'well', 'id', 'target'}

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
features=[c for c in train_df.columns if c not in SKIP]
X=train_df[features]; y=train_df["target"]; g=train_df["well"]; X_test=test_df[features]
print(f"[DIPBEAM] +2 features | #features now {len(features)}")
_qgc.collect()
'''


# DIAG: marginal benefit of the TCN member on pipeline A's Ridge OOF. Re-runs the
# Ridge stack with vs. without the 'tcn' column (same params / CV / groups) and
# prints both OOF RMSEs + the delta. This is the *private proxy* signal: the public
# LB is override-dominated and does NOT reflect base changes, so the submit decision
# rides on whether TCN lowers pipeline A's Ridge OOF (which flows into the blend).
DIAG_SRC = r'''
# === DIAG: does the TCN member lower pipeline A's Ridge OOF? (private proxy) ===
from sklearn.linear_model import Ridge as _RD
from sklearn.model_selection import GroupKFold as _GKF


def _ridge_oof_rmse(_cols):
    _M = oof_preds[_cols].to_numpy(); _yv = y.values; _oof = np.zeros(len(_M))
    for _tri, _vai in _GKF(n_splits=CFG.n_splits).split(_M, groups=g):
        _r = _RD(**ridge_params).fit(_M[_tri], _yv[_tri])
        _oof[_vai] = _r.predict(_M[_vai])
    return float(np.sqrt(np.mean((_oof - _yv) ** 2)))


_cols_all = list(oof_preds.columns)
_cols_notcn = [c for c in _cols_all if c != "tcn"]
_r_notcn = _ridge_oof_rmse(_cols_notcn)
_r_tcn = _ridge_oof_rmse(_cols_all)
print(f"[DIAG] Ridge-A OOF  without_TCN={_r_notcn:.4f}  with_TCN={_r_tcn:.4f}  "
      f"delta={_r_tcn - _r_notcn:+.4f}  (negative = TCN helps the base)")
# surface + dip-beam feature count (compare without_TCN to the prior no-surf baseline)
_sdcols = [c for c in features if c.startswith("tvtS_") or c.startswith("surf_")
           or c in ("tvt_dipbeam_d", "dipbeam_dip")]
print(f"[SURF-DIPBEAM] features added: {len(_sdcols)} ({sorted(_sdcols)}) | total features={len(features)}")
'''


# === GEO-OOF diagnostic cell (read-only, appended LAST so every _gold_*/lik_pf/imputer is
#     defined). The base only OOF-scores the GBT ridge stack; the GEOSTEERING candidates
#     (PF/beam/selector/poly/surface/contact) that DOMINATE the final blend (0.7 in sp45,
#     direct in fleongg) are produced only at TEST time and never scored on train -> the
#     field tunes them on the override-mirage public LB. This builds the missing
#     private-relevant signal: for N train wells, mask the toe EXACTLY as test masks it
#     (TVT_input NaN on the toe), run the candidate pool, and score each candidate on the
#     toe against the TRUE `TVT` column (present only in train). Also tests the
#     quality-conditioning premise: does the PF's own confidence (pf_best_ll/pf_ll_spread/
#     pf_pt_std) predict its toe error? Prints [GEO-OOF] lines only; never writes the
#     submission; fully guarded so it can never break the run. ===
GEO_OOF_SRC = r'''
import numpy as _gnp, pandas as _gpd, os as _gos
from pathlib import Path as _GP
_GEO_N = 40          # subsample of train wells (2 PF ensembles/well is the runtime cost)
_GEO_SEEDS = 24      # candidate-pool PF seeds (matches _GOLD_CAL_SEEDS)
_GEO_QSEEDS = 48     # (B) quality-probe PF seeds (kept < the deployed 128 to bound runtime)


def _geo_find_train():
    try:
        for _root, _dirs, _files in _gos.walk('/kaggle/input'):
            if 'sample_submission.csv' in _files and (_GP(_root) / 'train').is_dir():
                return _GP(_root) / 'train'
    except Exception:
        pass
    return _GP('/kaggle/input/rogii-wellbore-geology-prediction/train')


try:
    _GEO_DIR = _geo_find_train()
    _geo_wids = sorted(p.stem.replace('__horizontal_well', '')
                       for p in _GEO_DIR.glob('*__horizontal_well.csv'))
    if len(_geo_wids) > _GEO_N:                      # deterministic even stride (no RNG)
        _gstep = len(_geo_wids) / float(_GEO_N)
        _geo_wids = [_geo_wids[int(_k * _gstep)] for _k in range(_GEO_N)]
    _geo_variants = _gold_variant_grid() if callable(globals().get('_gold_variant_grid')) else []
    _geo_acc = {}                 # candidate name -> [toe RMSE per well]
    _geo_q = []                   # (pf_best_ll, pf_ll_spread, mean pf_pt_std, pf_toe_rmse)
    _geo_used = 0
    for _wid in _geo_wids:
        try:
            _hw = _gpd.read_csv(_GEO_DIR / (_wid + '__horizontal_well.csv'))
            _tw = _gpd.read_csv(_GEO_DIR / (_wid + '__typewell.csv'))
        except Exception:
            continue
        _tcol = [c for c in _hw.columns if c.lower() == 'tvt']
        if not _tcol or 'TVT_input' not in _hw.columns:
            continue
        _true = _hw[_tcol[0]].to_numpy(float)
        _toe = _hw['TVT_input'].isna().to_numpy()
        if int(_toe.sum()) < 20 or int(_gnp.isfinite(_true[_toe]).sum()) < 20:
            continue
        _geo_used += 1
        try:                                          # (A) candidate-pool toe floor
            _pool = _gold_candidate_pool(_wid, _hw, _tw, _GEO_DIR, _geo_variants,
                                         include_pf=True, n_seeds=_GEO_SEEDS,
                                         n_particles=int(globals().get('_GOLD_PARTICLES', 350)))
            for _nm, _pr in _pool.items():
                # LEAK-SAFE: drop contact_* -- _gold_contact_candidate re-reads the well's
                # TRAIN file and returns its TRUE TVT (a public-overlap override artifact that
                # does NOT generalize to private wells). Keep only candidates built from the
                # heel-masked TVT_input / offset-well structure (pf|*, poly_*, surface_*, dense_*).
                if _nm.startswith('contact'):
                    continue
                _pr = _gnp.asarray(_pr, float)
                if len(_pr) == len(_hw):
                    _e = float(_gold_rmse(_pr[_toe], _true[_toe]))
                    if _gnp.isfinite(_e):
                        _geo_acc.setdefault(_nm, []).append(_e)
        except Exception as _ea:
            print('[GEO-OOF] pool fail', _wid, str(_ea)[:80])
        try:                                          # (B) PF confidence vs PF toe error
            _out, _ev, _q = lik_pf(_hw, _tw, n_seeds=_GEO_QSEEDS, with_quality=True)
            if _q and 'pf_scale_8' in _out and len(_ev):
                _pf_pred = _gnp.asarray(_out['pf_scale_8'], float)
                _yt = _true[_ev]
                _mk = _gnp.isfinite(_pf_pred) & _gnp.isfinite(_yt)
                if int(_mk.sum()) >= 20:
                    _rpf = float(_gnp.sqrt(_gnp.mean((_pf_pred[_mk] - _yt[_mk]) ** 2)))
                    _geo_q.append((float(_q['pf_best_ll']), float(_q['pf_ll_spread']),
                                   float(_gnp.mean(_q['pf_pt_std'])), _rpf))
        except Exception as _eb:
            print('[GEO-OOF] quality fail', _wid, str(_eb)[:80])
    print(f'[GEO-OOF] wells scored={_geo_used}/{len(_geo_wids)}')
    _rank = sorted(((float(_gnp.mean(v)), float(_gnp.median(v)), len(v), k)
                    for k, v in _geo_acc.items() if len(v) >= max(5, _geo_used // 5)))
    # pf|* and poly_* are leak-clean AND self-independent (own GR/TVT_input + own typewell,
    # no cross-well imputer). surface_*/dense_* use the offset-well imputer with self_wid=None
    # here, so they carry a mild self-inclusion OPTIMISM vs production (self_wid=swid) -> flag
    # them so their toe-RMSE is not over-trusted. PF is the dominant final-blend signal, so the
    # leak-clean group is the one that drives the next lever.
    def _geo_clean(k):
        return k.startswith('pf|') or k.startswith('poly')
    print('[GEO-OOF] === candidate toe-RMSE (train, best first) | [C]=leak-clean self-indep, '
          '[o]=self-incl optimistic ===')
    for _gmean, _gmed, _gn, _gk in _rank[:30]:
        _tag = 'C' if _geo_clean(_gk) else 'o'
        print(f'[GEO-OOF]  [{_tag}] {_gk:28s} mean={_gmean:7.4f} median={_gmed:7.4f} n={_gn}')
    _clean_rank = [r for r in _rank if _geo_clean(r[3])]
    if _clean_rank:
        print(f'[GEO-OOF] best leak-clean candidate = {_clean_rank[0][3]} '
              f'mean={_clean_rank[0][0]:.4f} (this is the private-relevant geosteering floor)')
    if len(_geo_q) >= 10:
        _Q = _gnp.asarray(_geo_q, float)
        for _ci, _cn in enumerate(['pf_best_ll', 'pf_ll_spread', 'pf_pt_std_mean']):
            _cc = float(_gnp.corrcoef(_Q[:, _ci], _Q[:, 3])[0, 1])
            print(f'[GEO-OOF] corr({_cn:14s}, pf_toe_rmse) = {_cc:+.3f}  (n={len(_Q)})')
        print(f'[GEO-OOF] pf_toe_rmse mean={_Q[:,3].mean():.4f} '
              f'median={float(_gnp.median(_Q[:,3])):.4f} p90={float(_gnp.percentile(_Q[:,3],90)):.4f}')
except Exception as _eg:
    print('[GEO-OOF] SKIPPED:', str(_eg)[:200])
'''


# Full replacement for the final-blend cell. It is a strict superset of the base:
#  * the base 2-way sp45/fleongg blend (0.55/0.45 internal ratio) is computed exactly
#    as before, all `submission_sp45_fleongg_w*.csv` candidate exports + report CSV
#    are preserved (backward compatible);
#  * if `tcn_submission.csv` exists, a blend-level 3-way layer is added on TOP:
#        out = (1 - w_tcn) * [ w_sp45*sp45 + (1-w_sp45)*fleongg ] + w_tcn*tcn
#    so the sp45/fleongg 0.55/0.45 mix is untouched and TCN rides over it at w_tcn;
#  * DIAG prints true-RMSE (train has the public wells' real TVT) for w_tcn in
#    {0, 0.10, 0.15, 0.20} -- optimistic (TCN saw those wells) but a real-geology
#    direction signal;
#  * submission.csv is written from the 3-way w_tcn=0.15 / w_sp45=0.55 candidate;
#  * if tcn_submission.csv is absent (TCN failed), it AUTOMATICALLY falls back to the
#    base 2-way path -> the blend never dies on a TCN failure.
# The downstream override cell (_ov_tvt_from_contacts) still overwrites public wells.
BLEND3_SRC = r'''
from pathlib import Path as _FinalBlendPath
import numpy as _final_np
import pandas as _final_pd

_WORK = _FinalBlendPath('/kaggle/working') if _FinalBlendPath('/kaggle/working').exists() else _FinalBlendPath('.')
_BLEND_WEIGHTS_SP45 = (0.50, 0.52, 0.55, 0.58, 0.60)
_SELECTED_SP45_WEIGHT = 0.55
# w_tcn=0 ships the verified A-ridge-improved base (the Ridge stack optimally weights
# the TCN member: GroupKFold OOF 10.42->10.22, -0.199). The blend-level fixed-weight
# 3-way is kept dormant: it is NOT validated (the only available signal is the
# public-well true-RMSE, which is IN-SAMPLE and favors the GBT memorization, not the
# private/unseen-well behavior). The 3-way candidates are still exported for study.
_SELECTED_TCN_WEIGHT = 0.0
_TCN_WEIGHTS = (0.10, 0.15, 0.20)
_INPUT_FILES = {
    'fleongg': _WORK / 'submission.csv',
    'sp45': _WORK / 'sp45_projection_submission.csv',
    'tcn': _WORK / 'tcn_submission.csv',
}


def _read_submission_frame(path, label):
    frame = _final_pd.read_csv(path)
    missing = {'id', 'tvt'} - set(frame.columns)
    if missing:
        raise RuntimeError(f'{label} submission is missing columns: {sorted(missing)}')

    frame = frame[['id', 'tvt']].copy()
    frame['id'] = frame['id'].astype(str)
    frame['tvt'] = frame['tvt'].astype(float)

    if not _final_np.isfinite(frame['tvt'].to_numpy(dtype=float)).all():
        raise RuntimeError(f'Non-finite values in {label} tvt')
    return frame


def _merge_blend_inputs(sp45, fleongg):
    merged = sp45.rename(columns={'tvt': 'tvt_sp45'}).merge(
        fleongg.rename(columns={'tvt': 'tvt_fleongg'}),
        on='id',
        how='inner',
    )
    if len(merged) != len(sp45) or len(merged) != len(fleongg):
        raise RuntimeError(
            f'Blend id mismatch: sp45={len(sp45)}, fleongg={len(fleongg)}, merged={len(merged)}'
        )
    return merged


def _merge_blend_inputs3(sp45, fleongg, tcn):
    merged = _merge_blend_inputs(sp45, fleongg).merge(
        tcn.rename(columns={'tvt': 'tvt_tcn'}),
        on='id',
        how='inner',
    )
    if len(merged) != len(sp45) or len(merged) != len(fleongg) or len(merged) != len(tcn):
        raise RuntimeError(
            f'3-way blend id mismatch: sp45={len(sp45)}, fleongg={len(fleongg)}, '
            f'tcn={len(tcn)}, merged={len(merged)}'
        )
    return merged


def _weighted_submission(merged, w_sp45):
    w_fleongg = 1.0 - float(w_sp45)
    out = merged[['id']].copy()
    out['tvt'] = (
        float(w_sp45) * merged['tvt_sp45'].astype(float)
        + w_fleongg * merged['tvt_fleongg'].astype(float)
    )
    return out


def _weighted_submission3(merged, w_sp45, w_tcn):
    # Keep the sp45/fleongg internal ratio (default 0.55/0.45) intact; ride TCN over it.
    _base = (
        float(w_sp45) * merged['tvt_sp45'].astype(float)
        + (1.0 - float(w_sp45)) * merged['tvt_fleongg'].astype(float)
    )
    out = merged[['id']].copy()
    out['tvt'] = (1.0 - float(w_tcn)) * _base + float(w_tcn) * merged['tvt_tcn'].astype(float)
    return out


def _candidate_report_row(candidate, merged, file_name, w_sp45):
    diff = candidate['tvt'].to_numpy(dtype=float) - merged['tvt_sp45'].to_numpy(dtype=float)
    return {
        'file': file_name,
        'w_sp45': float(w_sp45),
        'w_fleongg': float(1.0 - w_sp45),
        'rows': int(len(candidate)),
        'mean_tvt': float(candidate['tvt'].mean()),
        'std_tvt': float(candidate['tvt'].std()),
        'rmse_vs_sp45': float(_final_np.sqrt(_final_np.mean(diff * diff))),
        'p95_abs_vs_sp45': float(_final_np.quantile(_final_np.abs(diff), 0.95)),
    }


# --- base 2-way sp45/fleongg blend (unchanged; candidate exports + report preserved) ---
_fle = _read_submission_frame(_INPUT_FILES['fleongg'], 'fleongg')
_fle.to_csv(_WORK / 'fleongg_pretrained_submission.csv', index=False)
_sp45 = _read_submission_frame(_INPUT_FILES['sp45'], 'sp45')
_merged = _merge_blend_inputs(_sp45, _fle)

_report_rows = []
for _w_sp45 in _BLEND_WEIGHTS_SP45:
    _candidate = _weighted_submission(_merged, _w_sp45)
    _name = f'submission_sp45_fleongg_w{_w_sp45:.2f}.csv'
    _candidate.to_csv(_WORK / _name, index=False)
    _report_rows.append(_candidate_report_row(_candidate, _merged, _name, _w_sp45))

_report = _final_pd.DataFrame(_report_rows)
_report.to_csv(_WORK / 'sp45_fleongg_blend_report.csv', index=False)
print(_report.to_string(index=False), flush=True)

# --- blend-level 3-way layer (TCN as a standalone member, no A-ridge dilution) ---
_tcn_path = _INPUT_FILES['tcn']
_use_3way = _FinalBlendPath(_tcn_path).exists()
if _use_3way:
    try:
        _tcn_sub = _read_submission_frame(_tcn_path, 'tcn')
        _merged3 = _merge_blend_inputs3(_sp45, _fle, _tcn_sub)
    except Exception as _e3:
        print('[BLEND3] 3-way setup failed -> 2-way fallback:', str(_e3)[:160], flush=True)
        _use_3way = False
else:
    print('[BLEND3] tcn_submission.csv absent -> 2-way fallback', flush=True)

if _use_3way:
    # DIAG: true-RMSE on the public wells (train holds their real TVT via
    # true_tvt = last_known_tvt + target). Optimistic (TCN saw these wells in
    # training) but a real-geology direction signal for picking w_tcn.
    _id2true = dict(zip(
        train_df['id'].astype(str),
        (train_df['last_known_tvt'].astype(float) + train_df['target'].astype(float)),
    ))
    _has_true = _merged3['id'].astype(str).isin(_id2true)
    _n_true = int(_has_true.sum())
    if _n_true > 0:
        _td = _merged3[_has_true].copy()
        _true = _td['id'].astype(str).map(_id2true).to_numpy(dtype=float)

        def _rmse_vs_true(_cand):
            _p = _cand.loc[_has_true.values, 'tvt'].to_numpy(dtype=float)
            return float(_final_np.sqrt(_final_np.mean((_p - _true) ** 2)))

        _cand0 = _weighted_submission3(_merged3, _SELECTED_SP45_WEIGHT, 0.0)
        print(f'[BLEND3-DIAG] true-RMSE on {_n_true} public-well rows '
              f'(train-known TVT; optimistic, TCN trained on these):', flush=True)
        print(f'[BLEND3-DIAG]   w_tcn=0.00 (2-way) RMSE_vs_true={_rmse_vs_true(_cand0):.4f}', flush=True)
        for _wt in _TCN_WEIGHTS:
            _candw = _weighted_submission3(_merged3, _SELECTED_SP45_WEIGHT, _wt)
            print(f'[BLEND3-DIAG]   w_tcn={_wt:.2f} (3-way) RMSE_vs_true={_rmse_vs_true(_candw):.4f}', flush=True)
    else:
        print('[BLEND3-DIAG] no public-well overlap with train ids -> skipping true-RMSE', flush=True)

    # export all 3-way candidates (compat-side, alongside the 2-way ones)
    for _wt in _TCN_WEIGHTS:
        _c3 = _weighted_submission3(_merged3, _SELECTED_SP45_WEIGHT, _wt)
        _c3.to_csv(_WORK / f'submission_3way_wtcn{_wt:.2f}.csv', index=False)

    _final = _weighted_submission3(_merged3, _SELECTED_SP45_WEIGHT, _SELECTED_TCN_WEIGHT)
    _final.to_csv(_WORK / 'submission.csv', index=False)
    print(f'wrote final submission.csv via 3-way w_sp45={_SELECTED_SP45_WEIGHT:.2f} '
          f'w_tcn={_SELECTED_TCN_WEIGHT:.2f}', _final.shape, flush=True)
else:
    # base 2-way path (identical to the original blend cell's tail)
    _final_name = f'submission_sp45_fleongg_w{_SELECTED_SP45_WEIGHT:.2f}.csv'
    _final = _final_pd.read_csv(_WORK / _final_name)
    _final.to_csv(_WORK / 'submission.csv', index=False)
    print('wrote final submission.csv from', _final_name, _final.shape, flush=True)
'''


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # 1. SMOKE cell right after the CFG config cell (so it precedes the TCN cell,
    #    which reads SMOKE).
    i_cfg = find_cell(cells, "class CFG:")
    cfg = src_str(cells[i_cfg])
    assert "n_splits = 5" in cfg and "GroupKFold(n_splits=n_splits)" in cfg
    cells.insert(i_cfg + 1, code_cell(SMOKE_SRC.replace("__SMOKE__", "True" if SMOKE else "False")))
    report.append(f"inserted SMOKE cell at {i_cfg + 1} (after CFG)")

    # 1b. INJECT SURFACE then DIPBEAM cells right after the feature-build cell, so the
    #     GBT models (now retrained) and the later TCN see the new structural columns.
    i_feat = find_cell(cells, "features = [c for c in train_df.columns if c not in {'well','id','target'}]")
    fc = src_str(cells[i_feat])
    assert "X = train_df[features]" in fc and "X_test = test_df[features]" in fc, "unexpected feature-build cell"
    cells.insert(i_feat + 1, code_cell(SURF_SRC))
    cells.insert(i_feat + 2, code_cell(DIPBEAM_SRC))
    report.append(f"inserted SURFACE cell at {i_feat + 1} and DIPBEAM cell at {i_feat + 2} (after feature build)")

    # 1c. FORCE GBT re-training: the pretrained lightgbm/catboost artifacts were fit on
    #     the OLD feature set; with surface/dipbeam added they must be refit. Gate the
    #     load branch on `and not FORCE_RETRAIN` (FORCE_RETRAIN defined in the SMOKE cell).
    nr = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        anc = "if (CFG.artifacts_path / save_path).exists():"
        if anc in cs and "and not FORCE_RETRAIN" not in cs:
            set_src(c, cs.replace(anc, "if (CFG.artifacts_path / save_path).exists() and not FORCE_RETRAIN:"))
            nr += 1
    assert nr == 2, f"expected to gate 2 GBT load branches (lightgbm + catboost), gated {nr}"
    report.append(f"gated {nr} GBT load branches on `and not FORCE_RETRAIN`")

    # 2. INJECT the TCN member cell right before `oof_preds = pd.DataFrame(oof_preds)`
    #    (after the catboost loop; oof_preds/test_preds are still dicts here).
    i_df = find_cell(cells, "oof_preds = pd.DataFrame(oof_preds)")
    cells.insert(i_df, code_cell(TCN_SRC))
    report.append(f"inserted TCN cell at {i_df} (before oof_preds = pd.DataFrame(oof_preds))")

    # 3. INJECT the DIAG cell right after the Ridge stack is built (oof_preds is a
    #    DataFrame by then), so the log reports the TCN's marginal OOF benefit.
    i_rd = find_cell(cells, "ridge_oof_preds = ridge_trainer.oof_preds")
    cells.insert(i_rd + 1, code_cell(DIAG_SRC))
    report.append(f"inserted DIAG cell at {i_rd + 1} (after Ridge stack)")

    # 4. REPLACE the final-blend cell with the 3-way version (TCN as a standalone
    #    blend member, with 2-way auto-fallback if tcn_submission.csv is missing).
    i_bl = find_cell(cells, "_SELECTED_SP45_WEIGHT = 0.55")
    base_blend = src_str(cells[i_bl])
    assert "_FinalBlendPath" in base_blend and "_weighted_submission(" in base_blend, \
        "unexpected blend cell shape"
    assert "_ov_tvt_from_contacts" not in base_blend, "blend cell must not be the override cell"
    set_src(cells[i_bl], BLEND3_SRC)
    report.append(f"replaced blend cell at {i_bl} with 3-way (TCN member + 2-way fallback)")

    # 5. APPEND the GEO-OOF diagnostic as the LAST cell, so every _gold_*/lik_pf/imputer it
    #    calls is already defined and the imputers are initialised (main() has run). It is
    #    read-only (prints [GEO-OOF] only, never writes submission) and fully guarded.
    _gold_cell_i = find_cell(cells, "Gold visible-prefix calibration overlay")
    assert _gold_cell_i == len(cells) - 1, \
        f"expected the gold-calibration cell to be last (at {_gold_cell_i}, n={len(cells)})"
    cells.append(code_cell(GEO_OOF_SRC))
    report.append(f"appended GEO-OOF diagnostic cell at {len(cells) - 1} (last)")

    # 6. LOO-honesty for the GEO-OOF surface/dense candidates: _gold_surface_candidates
    #    imputes with self_wid=None, so a TRAIN query well sees its OWN structure in the
    #    Formation-plane / DenseANCC imputer (built on all train wids) -> the dense_ancc_*/
    #    surface_* toe-RMSE is self-inclusion-optimistic. Pass self_wid=wid (the query well)
    #    at the two call sites inside _gold_surface_candidates so the query is excluded
    #    (proper leave-one-well-out). PRODUCTION-SAFE: at test time `wid` is a TEST well,
    #    which is absent from the train-built imputer pool, so self_wid=<test wid> excludes
    #    nothing (identical to None there) -> the deployed submission is unchanged.
    _gsrc = src_str(cells[_gold_cell_i])
    for _old, _new in [("fi.impute(xy, self_wid=None)", "fi.impute(xy, self_wid=wid)"),
                       ("di.impute(xy, self_wid=None)", "di.impute(xy, self_wid=wid)")]:
        assert _gsrc.count(_old) == 1, f"expected exactly one {_old!r} in the gold cell"
        _gsrc = _gsrc.replace(_old, _new)
    set_src(cells[_gold_cell_i], _gsrc)
    report.append("patched _gold_surface_candidates imputes to self_wid=wid (LOO honesty)")

    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # round-trip + sanity asserts
    rt = json.load(open(OUT, encoding="utf-8"))
    code = [c for c in rt["cells"] if c["cell_type"] == "code"]
    full = "\n".join(src_str(c) for c in code)

    # TCN cell exists, is a member, and the dict assignment happens before DataFrame conversion
    assert "class _TCN(nn.Module)" in full, "TCN class missing"
    assert 'oof_preds["tcn"] = _tcn_oof' in full, "oof_preds['tcn'] assignment missing"
    assert 'test_preds["tcn"] = _tcn_test' in full, "test_preds['tcn'] assignment missing"
    assert "results['tcn']" not in full, "stale results['tcn'] survived (should be removed)"
    assert full.index('oof_preds["tcn"] = _tcn_oof') < full.index("oof_preds = pd.DataFrame(oof_preds)"), \
        "TCN dict assignment must precede DataFrame conversion"
    # CV print retained
    assert "[TCN] CV toe RMSE mean" in full, "CV print missing"
    # variable adaptation: dualpipe names, not medal names
    assert "_tfeat = list(features)" in full, "feature_cols->features adaptation missing"
    assert "GroupKFold(n_splits=CFG.n_splits)" in full, "N_SPLITS->CFG.n_splits adaptation missing"
    # SMOKE cell precedes the TCN cell
    assert "SMOKE = True" in full or "SMOKE = False" in full, "SMOKE flag cell missing"
    assert full.index("FORCE_RETRAIN={FORCE_RETRAIN}") < full.index("class _TCN(nn.Module)"), \
        "SMOKE cell must precede TCN cell"
    # GEO-OOF diagnostic present, last, read-only, and scores against the true TVT toe
    assert "[GEO-OOF] === candidate toe-RMSE" in full, "GEO-OOF diagnostic missing"
    assert "_gold_candidate_pool(_wid, _hw, _tw" in full, "GEO-OOF must call the gold candidate pool"
    assert "with_quality=True" in full, "GEO-OOF quality-premise probe missing"
    assert full.rindex("[GEO-OOF]") > full.rindex("Gold visible-prefix calibration overlay"), \
        "GEO-OOF must run after the gold-calibration cell"
    # LOO-honesty patch applied: the two _gold_surface_candidates call sites use self_wid=wid
    assert "fi.impute(xy, self_wid=wid)" in full and "di.impute(xy, self_wid=wid)" in full, \
        "surface-candidate imputes not switched to self_wid=wid (LOO honesty)"
    assert "fi.impute(xy, self_wid=None)" not in full and "di.impute(xy, self_wid=None)" not in full, \
        "stale self_wid=None call site survived"
    # DIAG cell present and placed after the Ridge stack
    assert "[DIAG] Ridge-A OOF" in full, "DIAG cell missing"
    assert full.index("ridge_oof_preds = ridge_trainer.oof_preds") < full.index("[DIAG] Ridge-A OOF"), \
        "DIAG must come after the Ridge stack"
    # standalone TCN absolute submission written in the TCN cell
    assert "tcn_submission.csv" in full, "tcn_submission.csv writer missing"
    assert "_tcn_abs = (_lk + _tcn_test)" in full, "TCN absolute = last_known + delta missing"
    assert full.index("tcn_submission.csv") < full.index("oof_preds = pd.DataFrame(oof_preds)"), \
        "tcn_submission.csv must be written before the dict->DataFrame conversion"

    # 3-way blend cell: tcn input, 3-way weighting fn, true-RMSE DIAG, fallback, 3-way export
    assert "'tcn': _WORK / 'tcn_submission.csv'" in full, "_INPUT_FILES missing 'tcn'"
    assert "def _weighted_submission3(" in full, "_weighted_submission3 missing"
    assert "[BLEND3-DIAG]" in full and "RMSE_vs_true" in full, "true-RMSE DIAG missing"
    assert "submission_3way_wtcn" in full, "3-way candidate export missing"
    assert "2-way fallback" in full, "2-way fallback branch missing"
    # submission.csv is written via the 3-way path (and still via 2-way in the fallback)
    assert "wrote final submission.csv via 3-way" in full, "3-way submission.csv write missing"
    # base 2-way behavior preserved (candidate exports + report)
    assert "submission_sp45_fleongg_w{_w_sp45:.2f}.csv" in full, "2-way candidate exports lost"
    assert "sp45_fleongg_blend_report.csv" in full, "blend report export lost"

    # SURFACE + DIPBEAM cells present, after feature build, before the GBT loops & TCN
    assert "class _FormSurf:" in full, "SURFACE cell (_FormSurf) missing"
    assert "_FS = _FormSurf(train_wids, TRAIN_DIR)" in full, "SURFACE build missing"
    assert "tvtS_" in full, "surface tvtS_ columns missing"
    assert "def _beam_dip_jit(" in full and "tvt_dipbeam_d" in full and "dipbeam_dip" in full, \
        "DIPBEAM cell missing"
    _i_feat = full.index("features = [c for c in train_df.columns if c not in {'well','id','target'}]")
    assert _i_feat < full.index("class _FormSurf:"), "SURFACE must come after the feature build"
    assert full.index("class _FormSurf:") < full.index("def _beam_dip_jit("), "SURFACE must precede DIPBEAM"
    assert full.index("[DIPBEAM] +2 features") < full.index('for i, params in enumerate(lgb_params):'), \
        "SURF/DIPBEAM must be injected before the lightgbm loop"
    assert full.index("[DIPBEAM] +2 features") < full.index("_tfeat = list(features)"), \
        "SURF/DIPBEAM must be injected before the TCN reads features"
    # surface/dipbeam recompute features/X/X_test (dualpipe names, not medal feature_cols/Xt)
    assert "X_test = test_df[features]" in full, "SURFACE feature recompute (X_test) missing"
    assert "X=train_df[features]; y=train_df[\"target\"]; g=train_df[\"well\"]; X_test=test_df[features]" in full, \
        "DIPBEAM feature recompute missing"
    # FORCE_RETRAIN defined and both GBT load branches gated on it
    assert "FORCE_RETRAIN = True" in full, "FORCE_RETRAIN flag missing"
    assert full.count("and not FORCE_RETRAIN") == 2, \
        "both lightgbm + catboost load branches must be gated on `and not FORCE_RETRAIN`"
    # the train.csv fast-load is untouched (surface/dipbeam are merged on top, not recomputed)
    assert 'pd.read_csv(CFG.artifacts_path / "data" / "train.csv"' in full, "train.csv fast-load harmed"
    # SURF-DIPBEAM feature-count DIAG line
    assert "[SURF-DIPBEAM] features added:" in full, "SURF-DIPBEAM feature-count DIAG missing"

    # existing override / final blend markers intact (override cell untouched)
    assert "_ov_tvt_from_contacts" in full, "contact-override marker missing (override cell harmed?)"
    assert "_FinalBlendPath" in full, "final-blend marker missing (blend cell harmed?)"
    assert "features = [c for c in train_df.columns if c not in {'well','id','target'}]" in full, \
        "feature build cell harmed"

    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()

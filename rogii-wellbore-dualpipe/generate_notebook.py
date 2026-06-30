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
SMOKE = True

# The BiLSTM Ridge member is shelved: it ran (CV 11.04) but the kernel OOMs at the
# following TCN cell (two sequence models + full GBT retrain exceed host RAM), and its
# expected marginal is low (weaker than the TCN, same features -> correlated, and any
# gain is diluted ~0.165x to the final blend). Code kept; insertion gated off here.
STACK_LSTM = False


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
SURF_CHILD = r'''import sys, time, os
from pathlib import Path
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
hw_paths = sorted((DATA / "train").glob("*__horizontal_well.csv"))
test_paths = sorted((DATA / "test").glob("*__horizontal_well.csv"))
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
print(f"[SURF-CHILD] train wells={len(train_wids)} test wells={len(test_paths)}", flush=True)

# === Global formation structural-surface features (per-row, datum-separated, LOO fold-safe) ===
from scipy.spatial import cKDTree as _ckd
import numpy as _np, pandas as _pd, gc as _gc, time as _t

TRAIN_DIR = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction") / "train"   # dualpipe path (medal used a TRAIN_DIR constant)
SKIP = {'well', 'id', 'target'}          # dualpipe feature-exclusion set

_LAM = 5.0    # TPS smoothing (tune in smoke: 1, 5, 20). N = W distinct well centroids.


def _tps_U(r2):
    # thin-plate radial basis r^2*log(r) expressed in r^2 (>=0); 0 at r=0
    return _np.where(r2 > 1e-12, 0.5 * r2 * _np.log(_np.maximum(r2, 1e-30)), 0.0)


class _FormSurfTPS:
    """Per-formation S_f(X,Y) via smoothed Thin-Plate-Spline on a FEW representative
       nodes per well (heel/mid/toe). Horizontal wells are near-linear tracks, so
       per-row points are spatially redundant; a few nodes per well capture the
       cross-well global structure (dip/folding between wells). Replaces BOTH the
       2nd-order trend and the IDW residual with one global C1 surface; intra-well
       dip is carried by the per-well datum b_f (_surf_feats) and the dipbeam member.
       Exact well-level LOO: each train well's surface is solved with that well's
       nodes EXCLUDED (cached per-well), so any GroupKFold split is leak-free -- same
       guarantee as the prior _FormSurf downdate. predict(xy, wid) is API-compatible."""

    def __init__(self, wids, data_dir):
        raw = []
        for wid in wids:
            p = data_dir / (wid + "__horizontal_well.csv")
            try:
                df = _pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            raw.append((wid, df))
        W = len(raw)
        rows_x = []; rows_y = []; rows_w = []; rows_F = {f: [] for f in FORMATIONS}
        code = {}
        for wid, df in raw:
            c = code.setdefault(wid, len(code))
            # ONE representative node per well: centroid (X,Y) + median formation top.
            # Horizontal wells have a tiny X,Y footprint, so heel/mid/toe of one well
            # are near-coincident points -> the TPS drift block [1,x,y] goes rank-
            # deficient and the saddle system becomes singular (LAPACK segfault =
            # the earlier DeadKernel, no Python traceback). The formation surface is
            # fundamentally a CROSS-well object (it varies with well LOCATION and is
            # ~constant within a near-linear well); within-well variation is carried
            # by the per-well datum b_f and the Z trajectory. So N = W distinct nodes.
            rows_x.append(_np.array([df["X"].mean()], _np.float64))
            rows_y.append(_np.array([df["Y"].mean()], _np.float64))
            rows_w.append(_np.full(1, c, _np.int32))
            for f in FORMATIONS:
                rows_F[f].append(_np.array([_np.median(df[f].to_numpy(_np.float64))], _np.float64))
        self.code = code
        self.nx = _np.concatenate(rows_x); self.ny = _np.concatenate(rows_y)
        self.nw = _np.concatenate(rows_w)
        self.nF = {f: _np.concatenate(rows_F[f]) for f in FORMATIONS}
        N = len(self.nx)
        self.mx = float(self.nx.mean()); self.my = float(self.ny.mean())
        self.sx = float(self.nx.std() or 1.0); self.sy = float(self.ny.std() or 1.0)
        self._Xn = (self.nx - self.mx) / self.sx
        self._Yn = (self.ny - self.my) / self.sy
        _bytes = (N + 3) ** 2 * 8
        print(f"[SURF-TPS] wells={W} N={N} (1 centroid node/well) "
              f"saddle={(N+3)}x{(N+3)} ({_bytes/1e6:.0f}MB) lam={_LAM}", flush=True)
        _t0s = _t.time()
        self._full = self._solve(_np.ones(N, bool))
        print(f"[SURF-TPS] full solve {_t.time()-_t0s:.2f}s", flush=True)
        _t0s = _t.time()
        self._loo = {}
        for c in _np.unique(self.nw):
            self._loo[int(c)] = self._solve(self.nw != c)
        print(f"[SURF-TPS] {len(self._loo)} per-well LOO solves {_t.time()-_t0s:.1f}s", flush=True)

    def _solve(self, mask):
        Xn = self._Xn[mask]; Yn = self._Yn[mask]; m = len(Xn)
        dx = Xn[:, None] - Xn[None, :]; dy = Yn[:, None] - Yn[None, :]
        K = _tps_U(dx * dx + dy * dy) + _LAM * _np.eye(m)
        P = _np.column_stack([_np.ones(m), Xn, Yn])
        Asys = _np.zeros((m + 3, m + 3))
        Asys[:m, :m] = K; Asys[:m, m:] = P; Asys[m:, :m] = P.T
        Asys += 1e-8 * _np.eye(m + 3)
        RHS = _np.zeros((m + 3, len(FORMATIONS)))
        for j, f in enumerate(FORMATIONS):
            RHS[:m, j] = self.nF[f][mask]
        SOL = _np.linalg.solve(Asys, RHS)   # saddle is symmetric-indefinite -> LU
        coeffs = {f: (SOL[:m, j], SOL[m:, j]) for j, f in enumerate(FORMATIONS)}
        return {"Xn": Xn, "Yn": Yn, "coeffs": coeffs}

    def _eval(self, bundle, xy, f):
        Xq = (xy[:, 0] - self.mx) / self.sx
        Yq = (xy[:, 1] - self.my) / self.sy
        dx = Xq[:, None] - bundle["Xn"][None, :]
        dy = Yq[:, None] - bundle["Yn"][None, :]
        w, a = bundle["coeffs"][f]
        return _tps_U(dx * dx + dy * dy) @ w + a[0] + a[1] * Xq + a[2] * Yq

    def predict(self, xy, wid=None):
        xy = _np.atleast_2d(xy)
        code = self.code.get(wid) if wid is not None else None
        b = self._loo[code] if (code is not None and code in self._loo) else self._full
        return {f: self._eval(b, xy, f).astype(_np.float64) for f in FORMATIONS}


print("[SURF] building FormationSurfaceModel (representative-node TPS, per-well LOO) ...")
_t0 = _t.time()
_FS = _FormSurfTPS(train_wids, TRAIN_DIR)
print(f"[SURF]  nodes={len(_FS.nx):,} wells={len(_FS.code)} ({_t.time()-_t0:.0f}s)")


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

print(f"[SURF-CHILD] surf rows train={len(_sf_tr)} test={len(_sf_te)} cols={list(_sf_tr.columns)}", flush=True)
_sf_tr.to_pickle("/kaggle/working/surf_tps_train.pkl")
_sf_te.to_pickle("/kaggle/working/surf_tps_test.pkl")
print("SURF_CHILD_OK", flush=True)
'''


SURF_SRC = r'''
# === Global formation structural-surface features (TPS), computed in an ISOLATED
#     subprocess (OOM-safe). The in-process build's peak memory coexisted with the
#     resident train.csv inside the T4 kernel -> OOM-kill (DeadKernel, no traceback).
#     The child re-derives paths from the mounted competition data, builds the SAME
#     _FormSurfTPS + _surf_feats (per-well LOO), and pickles surf_tps_{train,test}.pkl.
#     This parent runs it, captures its returncode/stdout/stderr, then reads + merges. ===
import subprocess as _sp, sys as _sys, time as _stime, os as _sos
import numpy as _np, pandas as _pd, gc as _gc

SKIP = {'well', 'id', 'target'}
_FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
_SURF_CHILD = __SURF_CHILD__

_wk = '/kaggle/working' if _sos.path.exists('/kaggle/working') else '.'
with open(_wk + '/surf_child.py', 'w') as _fh:
    _fh.write(_SURF_CHILD)

print("[SURF] building TPS surface in an ISOLATED subprocess (OOM-safe) ...", flush=True)
_t0 = _stime.time(); _rc = None; _o = ''; _e = ''
try:
    _r = _sp.run([_sys.executable, '-u', _wk + '/surf_child.py'],
                 capture_output=True, text=True, timeout=5400)
    _rc, _o, _e = _r.returncode, _r.stdout, _r.stderr
except _sp.TimeoutExpired as _texc:
    _o = _texc.stdout if isinstance(_texc.stdout, str) else (_texc.stdout or b'').decode('utf-8', 'replace')
    _e = _texc.stderr if isinstance(_texc.stderr, str) else (_texc.stderr or b'').decode('utf-8', 'replace')
    print("[SURF] !!! surface child TIMED OUT !!!", flush=True)
print(f"[SURF] child returncode={_rc} elapsed={_stime.time()-_t0:.0f}s", flush=True)
print("[SURF] ---- child stdout (tail) ----", flush=True); print(_o[-4000:], flush=True)
if _e.strip():
    print("[SURF] ---- child stderr (tail) ----", flush=True); print(_e[-4000:], flush=True)

_trp = _wk + '/surf_tps_train.pkl'; _tep = _wk + '/surf_tps_test.pkl'
if _rc == 0 and _sos.path.exists(_trp) and _sos.path.exists(_tep):
    _sf_tr = _pd.read_pickle(_trp); _sf_te = _pd.read_pickle(_tep)
    print(f"[SURF] loaded surf pickles train={len(_sf_tr)} test={len(_sf_te)}", flush=True)
else:
    print(f"[SURF] !!! surface child FAILED (rc={_rc}) -> EMPTY surface fallback (fillna 0.0) !!!", flush=True)
    _cols0 = ['id', 'surf_mean_d', 'surf_std', 'surf_rng', 'surf_datum_spread'] + [f'tvtS_{_f}_d' for _f in _FORMS]
    _sf_tr = _pd.DataFrame({_c: [] for _c in _cols0}); _sf_te = _pd.DataFrame({_c: [] for _c in _cols0})

_n_tr0 = len(train_df); _n_te0 = len(test_df)
train_df = train_df.merge(_sf_tr, on="id", how="left")
test_df = test_df.merge(_sf_te, on="id", how="left")
assert len(train_df) == _n_tr0 and len(test_df) == _n_te0, "surf merge changed row count"
_surfcols = [c for c in _sf_tr.columns if c != "id"]
_miss_tr = int(train_df[_surfcols].isna().any(axis=1).sum())
_miss_te = int(test_df[_surfcols].isna().any(axis=1).sum())
print(f"[SURF] merged surf cols={len(_surfcols)} | train NaN-rows={_miss_tr} test NaN-rows={_miss_te}")
if _miss_tr:
    print(f"[SURF] WARNING: {_miss_tr} train rows lack surface features (filled 0.0)", flush=True)
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
# LSTM member marginal: drop only 'lstm' from the full stack (with_all=_r_tcn includes
# every member). delta_lstm<0 => the BiLSTM decorrelates from GBT+TCN and helps the base.
_cols_nolstm = [c for c in _cols_all if c != "lstm"]
_r_nolstm = _ridge_oof_rmse(_cols_nolstm) if "lstm" in _cols_all else _r_tcn
print(f"[DIAG] Ridge-A OOF  without_LSTM={_r_nolstm:.4f}  with_all={_r_tcn:.4f}  "
      f"delta_lstm={_r_tcn - _r_nolstm:+.4f}  (negative = LSTM helps the base)")
# surface + dip-beam feature count (compare without_TCN to the prior no-surf baseline)
_sdcols = [c for c in features if c.startswith("tvtS_") or c.startswith("surf_")
           or c in ("tvt_dipbeam_d", "dipbeam_dip")]
print(f"[SURF-DIPBEAM] features added: {len(_sdcols)} ({sorted(_sdcols)}) | total features={len(features)}")
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
    'surf': _WORK / 'surf_submission.csv',
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

# --- blend-level C layer: surf+dipbeam standalone member, UN-DILUTED (rides over the
#     2-way base, parallel to the dormant TCN lane). C is the decorrelated structural
#     model whose errors are uncorrelated with the GBT/Ridge base -> the "biggest cheap
#     win" for the PRIVATE base. Shipped at _SELECTED_C_WEIGHT (0.0 = verified base until
#     C's standalone CV is confirmed by the [C] CV log); all C variants are exported so a
#     follow-up can pick w_c. This block runs LAST so it has the final word on
#     submission.csv (the downstream override cell still overwrites public wells only).
_SELECTED_C_WEIGHT = 0.0
_C_WEIGHTS = (0.10, 0.15, 0.20, 0.25)
_surf_path = _INPUT_FILES.get('surf')
_use_C = _surf_path is not None and _FinalBlendPath(_surf_path).exists()
if _use_C:
    try:
        _surf_sub = _read_submission_frame(_surf_path, 'surf')
        _mergedC = _merge_blend_inputs(_sp45, _fle).merge(
            _surf_sub.rename(columns={'tvt': 'tvt_surf'}), on='id', how='inner')
        if len(_mergedC) != len(_sp45):
            raise RuntimeError(f'C blend id mismatch: sp45={len(_sp45)}, merged={len(_mergedC)}')
    except Exception as _eC:
        print('[BLENDC] C setup failed -> C layer skipped:', str(_eC)[:160], flush=True)
        _use_C = False

if _use_C:
    def _weighted_submission_C(merged, w_sp45, w_c):
        # keep sp45/fleongg internal ratio intact; ride C over it (TCN-lane structure)
        _base = (float(w_sp45) * merged['tvt_sp45'].astype(float)
                 + (1.0 - float(w_sp45)) * merged['tvt_fleongg'].astype(float))
        out = merged[['id']].copy()
        out['tvt'] = (1.0 - float(w_c)) * _base + float(w_c) * merged['tvt_surf'].astype(float)
        return out

    # DIAG: public-well true-RMSE (IN-SAMPLE / optimistic; C saw these wells in training).
    # Direction signal only -- the private/unseen-well behavior is NOT measured here.
    _id2trueC = dict(zip(train_df['id'].astype(str),
                         (train_df['last_known_tvt'].astype(float) + train_df['target'].astype(float))))
    _hasC = _mergedC['id'].astype(str).isin(_id2trueC)
    _nC = int(_hasC.sum())
    if _nC > 0:
        _trueC = _mergedC.loc[_hasC, 'id'].astype(str).map(_id2trueC).to_numpy(dtype=float)

        def _rmseC(_cand):
            _p = _cand.loc[_hasC.values, 'tvt'].to_numpy(dtype=float)
            return float(_final_np.sqrt(_final_np.mean((_p - _trueC) ** 2)))

        print(f'[BLENDC-DIAG] true-RMSE on {_nC} public-well rows (IN-SAMPLE, optimistic; direction only):', flush=True)
        for _wc in (0.0,) + _C_WEIGHTS:
            print(f'[BLENDC-DIAG]   w_c={_wc:.2f} RMSE_vs_true='
                  f'{_rmseC(_weighted_submission_C(_mergedC, _SELECTED_SP45_WEIGHT, _wc)):.4f}', flush=True)
    else:
        print('[BLENDC-DIAG] no public-well overlap -> skipping C true-RMSE', flush=True)

    for _wc in _C_WEIGHTS:
        _weighted_submission_C(_mergedC, _SELECTED_SP45_WEIGHT, _wc).to_csv(
            _WORK / f'submission_surf_w{_wc:.2f}.csv', index=False)

    _finalC = _weighted_submission_C(_mergedC, _SELECTED_SP45_WEIGHT, _SELECTED_C_WEIGHT)
    _finalC.to_csv(_WORK / 'submission.csv', index=False)
    print(f'[BLENDC] wrote final submission.csv via C layer w_c={_SELECTED_C_WEIGHT:.2f}', _finalC.shape, flush=True)
else:
    print('[BLENDC] surf_submission.csv absent -> C layer skipped (submission.csv unchanged)', flush=True)
'''


# === Bidirectional LSTM as a 2nd decorrelated series member in pipeline A's Ridge stack.
#     Modeled on TCN_SRC but with independent _l* names (so TCN's `del _tr,_te` cannot
#     touch it) and a recurrent architecture (decorrelated from the dilated-conv TCN).
#     Added to oof_preds/test_preds while they are still dicts -> the Ridge stack weights
#     it optimally (positive-constrained), no fixed-weight dilution. NOT a blend member.
LSTM_SRC = r'''
import torch, torch.nn as nn
import gc
torch.manual_seed(43); np.random.seed(43)
_LE = 2 if SMOKE else 40
_LPAT = 1 if SMOKE else 8
_LHID = 48 if SMOKE else 128      # hidden size per direction
_LNL = 1 if SMOKE else 2          # stacked LSTM layers
_LDROP, _LLR, _LWD, _LCLIP = 0.15, 1e-3, 1e-4, 1.0
_LNSEED = 1 if SMOKE else 3
_lfeat = list(features)
print(f"[LSTM] SMOKE={SMOKE} ep={_LE} hid={_LHID} nl={_LNL} nfeat={len(_lfeat)} seeds={_LNSEED}")

_ldev = "cpu"
try:
    if torch.cuda.is_available():
        _lg = torch.zeros(8, device="cuda"); _ = float((_lg + 1).sum().item()); _ldev = "cuda"
        print("[LSTM] GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("[LSTM] GPU unusable -> CPU:", str(e)[:100])

_lXall = train_df[_lfeat].to_numpy(np.float32); _lXall[~np.isfinite(_lXall)] = np.nan
_lmu = np.nanmean(_lXall, 0).astype(np.float32); _lsd = np.nanstd(_lXall, 0).astype(np.float32); _lsd[_lsd < 1e-6] = 1.0
del _lXall; gc.collect()
_lyt = train_df['target'].to_numpy(np.float32); _lymu = float(np.nanmean(_lyt)); _lysd = float(np.nanstd(_lyt)) or 1.0


def _lnorm(M):
    M = M.astype(np.float32, copy=True); M[~np.isfinite(M)] = np.nan; M = (M - _lmu) / _lsd
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


def _lseqs(df, has_t):
    df = df.copy(); df['_ri'] = df['id'].str.rsplit('_', n=1).str[-1].astype(int)
    out = []
    for wid, gd in df.groupby('well', sort=False):
        gd = gd.sort_values('_ri')
        it = {'wid': wid, 'X': _lnorm(gd[_lfeat].to_numpy()), 'ids': gd['id'].to_numpy()}
        if has_t:
            it['t'] = (gd['target'].to_numpy(np.float32) - _lymu) / _lysd
        out.append(it)
    return out


_ltr = _lseqs(train_df, True); _lte = _lseqs(test_df, False)
_lgrp = np.array([s['wid'] for s in _ltr])


class _BiLSTM(nn.Module):
    def __init__(s, ci, hid, nl, dr):
        super().__init__()
        s.inp = nn.Linear(ci, hid)
        s.act = nn.ReLU()
        s.lstm = nn.LSTM(hid, hid, num_layers=nl, batch_first=True,
                         bidirectional=True, dropout=(dr if nl > 1 else 0.0))
        s.do = nn.Dropout(dr)
        s.h = nn.Linear(2 * hid, 1)   # 2*hid: forward+backward concat

    def forward(s, x):                # x: (1, L, ci) per-well, batch=1, variable length
        z = s.act(s.inp(x))
        z, _ = s.lstm(z)              # (1, L, 2*hid)
        return s.h(s.do(z)).squeeze(-1)   # (1, L)


def _lhub(p, t, d=1.0):
    e = p - t; a = e.abs(); return torch.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d)).mean()


def _ltx(s):
    return torch.tensor(s['X'][None], dtype=torch.float32, device=_ldev)


def _ltrain_one(_tri, _vai, _seed):
    torch.manual_seed(_seed); np.random.seed(_seed)
    _m = _BiLSTM(len(_lfeat), _LHID, _LNL, _LDROP).to(_ldev)
    _opt = torch.optim.Adam(_m.parameters(), lr=_LLR, weight_decay=_LWD)
    _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_LE)
    _best = 1e9; _bs = None; _bad = 0; _trl = np.array(_tri)
    for _ep in range(_LE):
        _m.train(); np.random.shuffle(_trl)
        for _j in _trl:
            s = _ltr[_j]; x = _ltx(s); t = torch.tensor(s['t'][None], dtype=torch.float32, device=_ldev)
            _opt.zero_grad(); _l = _lhub(_m(x), t); _l.backward()
            torch.nn.utils.clip_grad_norm_(_m.parameters(), _LCLIP); _opt.step()
        _sch.step()
        _m.eval(); _P = []; _T = []
        with torch.no_grad():
            for _j in _vai:
                s = _ltr[_j]; _P.append(_m(_ltx(s)).cpu().numpy()[0] * _lysd + _lymu); _T.append(s['t'] * _lysd + _lymu)
        _vr = float(np.sqrt(np.mean((np.concatenate(_P) - np.concatenate(_T)) ** 2)))
        if _vr < _best - 1e-4:
            _best = _vr; _bad = 0
            _bs = {k: v.detach().cpu().clone() for k, v in _m.state_dict().items()}
        else:
            _bad += 1
        if _bad >= _LPAT:
            break
    _m.load_state_dict(_bs); _m.eval(); return _m


_loof_id = {}; _ltest_sum = {}; _lnf = 0; _lfb = []
_lidx = np.arange(len(_ltr))
for _f, (_tri, _vai) in enumerate(GroupKFold(n_splits=CFG.n_splits).split(_lidx, groups=_lgrp)):
    _va_sum = {}; _te_fold = {}
    for _seed in range(_LNSEED):
        _m = _ltrain_one(_tri, _vai, 2000 * _seed + _f)
        with torch.no_grad():
            for _j in _vai:
                s = _ltr[_j]; pr = _m(_ltx(s)).cpu().numpy()[0] * _lysd + _lymu
                for _i, _id in enumerate(s['ids']):
                    _va_sum[_id] = _va_sum.get(_id, 0.0) + float(pr[_i]) / _LNSEED
            for s in _lte:
                pr = _m(_ltx(s)).cpu().numpy()[0] * _lysd + _lymu
                for _i, _id in enumerate(s['ids']):
                    _te_fold[_id] = _te_fold.get(_id, 0.0) + float(pr[_i]) / _LNSEED
    for _id, v in _va_sum.items():
        _loof_id[_id] = v
    for _id, v in _te_fold.items():
        _ltest_sum[_id] = _ltest_sum.get(_id, 0.0) + v
    _lnf += 1
    _vp = []; _vt = []
    for _j in _vai:
        s = _ltr[_j]; _vp.extend([_va_sum[i] for i in s['ids']]); _vt.extend(list(s['t'] * _lysd + _lymu))
    _fr = float(np.sqrt(np.mean((np.array(_vp) - np.array(_vt)) ** 2))); _lfb.append(_fr)
    print(f"[LSTM] fold{_f} toe RMSE={_fr:.4f}")
print("[LSTM] CV toe RMSE mean =", float(np.mean(_lfb)), "| folds", [round(b, 3) for b in _lfb])

_lstm_oof = train_df['id'].map(_loof_id).to_numpy(np.float32)
_lte_mean = {k: v / _lnf for k, v in _ltest_sum.items()}
_lstm_test = test_df['id'].map(_lte_mean).to_numpy(np.float32)
assert len(_lstm_oof) == len(train_df), "LSTM OOF length != train_df"
assert len(_lstm_test) == len(test_df), "LSTM test length != test_df"
assert not np.isnan(_lstm_oof).any(), "LSTM OOF unmapped"
assert not np.isnan(_lstm_test).any(), "LSTM test unmapped"

# oof_preds / test_preds are still dicts here -> add the member; the Ridge stack weights it.
oof_preds["lstm"] = _lstm_oof
test_preds["lstm"] = _lstm_test
print(f"[LSTM] member added | OOF residual RMSE={float(np.sqrt(np.mean((_lstm_oof - y.values) ** 2))):.4f} | members={list(oof_preds.keys())}")
# Free the LSTM's GPU model + sequence arrays before the TCN cell runs. The prior
# DeadKernelError (kernel OOM) hit 25 s into the TCN cell: LSTM left a GPU model and
# seq copy resident, so TCN's own GPU model + seq rebuild tipped it over. Release them.
try:
    del _m
except NameError:
    pass
del _ltr, _lte, _loof_id, _ltest_sum
gc.collect()
try:
    if _ldev == "cuda":
        torch.cuda.empty_cache()
except Exception as _ce:
    print("[LSTM] empty_cache skipped:", str(_ce)[:80])
print("[LSTM] freed GPU + sequence memory before the TCN cell")
'''


# === Member C: surface+dipbeam-ONLY standalone GBT (decorrelated blend-level member). ===
# Reads ONLY the structural columns merged onto train_df/test_df by the SURFACE + DIPBEAM
# cells above. Does NOT touch features/X/oof_preds -> pipeline A is left UN-diluted (the
# A-ridge surf/dipbeam gain is kept separately & for free). Writes surf_submission.csv
# (absolute = last_known_tvt + delta, same id-alignment proven by the TCN standalone cell).
# Decorrelation is at the ERROR level, so overlap of inputs with A is irrelevant.
MEMBER_C_SRC = r'''
import lightgbm as _clgb, numpy as _cnp, pandas as _cpd, os as _cos
from sklearn.model_selection import GroupKFold as _CGKF

_C_COLS = [c for c in train_df.columns
           if c.startswith('tvtS_') or c.startswith('surf_')
           or c in ('tvt_dipbeam_d', 'dipbeam_dip')]
assert len(_C_COLS) >= 8, f"[C] too few structural cols ({_C_COLS}); SURFACE/DIPBEAM didn't run?"
print(f"[C] member-C feature cols={len(_C_COLS)}: {sorted(_C_COLS)}", flush=True)

_Xc  = train_df[_C_COLS].astype(_cnp.float32).to_numpy()
_Xct = test_df[_C_COLS].astype(_cnp.float32).to_numpy()
_yc  = train_df['target'].to_numpy(_cnp.float32)     # delta target (same as pipeline A)
_gc_ = train_df['well'].to_numpy()

_C_PARAMS = dict(objective='regression_l1', n_estimators=(60 if SMOKE else 600),
                 learning_rate=0.03, num_leaves=31, min_child_samples=40,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 reg_lambda=1.0, n_jobs=-1, verbosity=-1)

_oofC = _cnp.zeros(len(train_df), _cnp.float64)
_teC  = _cnp.zeros(len(test_df), _cnp.float64)
_cfb = []
for _cf, (_tri, _vai) in enumerate(_CGKF(n_splits=CFG.n_splits).split(_Xc, groups=_gc_)):
    _mc = _clgb.LGBMRegressor(**_C_PARAMS, random_state=100 + _cf)
    _mc.fit(_Xc[_tri], _yc[_tri])
    _oofC[_vai] = _mc.predict(_Xc[_vai])
    _teC += _mc.predict(_Xct) / CFG.n_splits
    _fr = float(_cnp.sqrt(_cnp.mean((_oofC[_vai] - _yc[_vai]) ** 2)))
    _cfb.append(_fr); print(f"[C] fold{_cf} delta-RMSE={_fr:.4f}", flush=True)
print("[C] CV delta-RMSE mean =", float(_cnp.mean(_cfb)), "| folds", [round(b, 3) for b in _cfb], flush=True)

# absolute standalone submission: abs = last_known_tvt (constant broadcast) + delta
_lkc = test_df['last_known_tvt'].to_numpy(_cnp.float64)
assert len(_lkc) == len(_teC), "[C] last_known_tvt / test length mismatch"
_C_abs = (_lkc + _teC).astype(float)
_wkC = '/kaggle/working' if _cos.path.exists('/kaggle/working') else '.'
_cpd.DataFrame({'id': test_df['id'].astype(str), 'tvt': _C_abs}).to_csv(f'{_wkC}/surf_submission.csv', index=False)
print(f"[C] wrote surf_submission.csv (abs, rows={len(_C_abs)}) | features/oof_preds untouched (no A dilution)", flush=True)
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
    cells.insert(i_feat + 1, code_cell(SURF_SRC.replace("__SURF_CHILD__", repr(SURF_CHILD))))
    cells.insert(i_feat + 2, code_cell(DIPBEAM_SRC))
    cells.insert(i_feat + 3, code_cell(MEMBER_C_SRC))
    report.append(f"inserted SURFACE cell at {i_feat + 1}, DIPBEAM at {i_feat + 2}, MEMBER_C at {i_feat + 3} (after feature build)")

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

    # 2. INJECT the BiLSTM member, then the TCN member, right before
    #    `oof_preds = pd.DataFrame(oof_preds)` (after the catboost loop; both are dict
    #    members here). LSTM goes FIRST so it is self-contained with _l* names and the
    #    TCN cell's `del _tr,_te` cannot break it. The Ridge stack weights both.
    i_df = find_cell(cells, "oof_preds = pd.DataFrame(oof_preds)")
    if STACK_LSTM:
        cells.insert(i_df, code_cell(LSTM_SRC))
        cells.insert(i_df + 1, code_cell(TCN_SRC))
        report.append(f"inserted BiLSTM cell at {i_df} and TCN cell at {i_df + 1} (before oof_preds DataFrame)")
    else:
        cells.insert(i_df, code_cell(TCN_SRC))
        report.append(f"inserted TCN cell at {i_df} (BiLSTM shelved: STACK_LSTM=False)")

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
    assert "class _FormSurfTPS:" in full, "SURFACE cell (_FormSurfTPS) missing"
    assert "_FS = _FormSurfTPS(train_wids, TRAIN_DIR)" in full, "SURFACE build missing"
    assert "tvtS_" in full, "surface tvtS_ columns missing"
    assert "def _beam_dip_jit(" in full and "tvt_dipbeam_d" in full and "dipbeam_dip" in full, \
        "DIPBEAM cell missing"
    _i_feat = full.index("features = [c for c in train_df.columns if c not in {'well','id','target'}]")
    assert _i_feat < full.index("class _FormSurfTPS:"), "SURFACE must come after the feature build"
    assert full.index("class _FormSurfTPS:") < full.index("def _beam_dip_jit("), "SURFACE must precede DIPBEAM"
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

    # BiLSTM member (only when STACK_LSTM): dict member before df conversion, _l* names
    if STACK_LSTM:
        assert "class _BiLSTM(nn.Module)" in full, "BiLSTM class missing"
        assert "nn.LSTM(" in full and "bidirectional=True" in full, "bidirectional LSTM layer missing"
        assert 'oof_preds["lstm"] = _lstm_oof' in full, "oof_preds['lstm'] assignment missing"
        assert 'test_preds["lstm"] = _lstm_test' in full, "test_preds['lstm'] assignment missing"
        assert full.index('oof_preds["lstm"] = _lstm_oof') < full.index("oof_preds = pd.DataFrame(oof_preds)"), \
            "LSTM dict assignment must precede DataFrame conversion"
        assert "[LSTM] CV toe RMSE mean" in full, "LSTM CV print missing"
        assert "_ltr = _lseqs(train_df, True)" in full, "LSTM independent _lseqs build missing"
        assert "lstm_submission.csv" not in full, "LSTM is a Ridge member only (no standalone submission)"
        assert full.index("[DIPBEAM] +2 features") < full.index("_lfeat = list(features)"), \
            "SURF/DIPBEAM must be injected before the LSTM reads features"
        assert full.index('oof_preds["lstm"] = _lstm_oof') < full.index('oof_preds["tcn"] = _tcn_oof'), \
            "LSTM cell must precede the TCN cell (so TCN's del _tr,_te can't touch _ltr,_lte)"
        assert "delta_lstm=" in full, "LSTM marginal DIAG missing"
    else:
        assert "class _BiLSTM" not in full, "BiLSTM should be absent when STACK_LSTM=False"

    # MEMBER_C cell present, after DIPBEAM, before the lightgbm loop; writes surf_submission.csv
    assert "[C] member-C feature cols" in full, "MEMBER_C cell missing"
    assert "surf_submission.csv" in full, "surf_submission.csv writer missing"
    assert "[C] CV delta-RMSE mean" in full, "MEMBER_C CV print missing"
    assert full.index("[DIPBEAM] +2 features") < full.index("[C] member-C feature cols"), \
        "MEMBER_C must come after DIPBEAM"
    assert full.index("[C] member-C feature cols") < full.index('for i, params in enumerate(lgb_params):'), \
        "MEMBER_C must be injected before the lightgbm loop"
    # C blend layer in the final blend cell (un-diluted member, runs last)
    assert "'surf': _WORK / 'surf_submission.csv'" in full, "_INPUT_FILES missing 'surf'"
    assert "def _weighted_submission_C(" in full, "_weighted_submission_C missing"
    assert "[BLENDC-DIAG]" in full, "C true-RMSE DIAG missing"
    assert "submission_surf_w" in full, "C variant export missing"
    assert "_SELECTED_C_WEIGHT = 0.0" in full, "C ships at verified base (w_c=0) until CV confirmed"

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

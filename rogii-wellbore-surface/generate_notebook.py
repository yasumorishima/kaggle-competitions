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
OUT = BASE / "rogii-surface.ipynb"

# Flip to False (regenerate + commit + push) for the full run.
SMOKE = True


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
    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()

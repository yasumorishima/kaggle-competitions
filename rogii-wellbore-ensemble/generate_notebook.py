"""
Generate the ROGII GBT+TCN ENSEMBLE notebook.

This is the baseline GBT fork (3xLGB + 3xCatBoost -> inline Climber hill-climb ->
Optuna post-proc -> submission, self-best LB 10.224) with a 7th ensemble member
injected: a sequence-model TCN over the same per-row features. The TCN standalone
reached CV toe RMSE 10.60 (stable, GroupNorm + grad-clip + cosine); 3/5 folds beat
the GBT. Because the GBT (flat per-row) and TCN (sequence) are different model
classes on overlapping features, their errors are partly decorrelated -> the
hill-climb blend should beat either alone.

Injection (verified by data-flow analysis of pub_dwt):
  oof_preds/test_preds are dicts (pub_dwt cell 8), populated per GBT model with
  arrays aligned to train_df/test_df row order (cells 10/12), then turned into
  DataFrames (cell 14) and fed to Climber.fit(oof_preds, y=train_df.target=delta)
  (cell 15). We insert a TCN cell right BEFORE the `oof_preds = pd.DataFrame(...)`
  cell that trains the TCN with GroupKFold, builds OOF (train_df order, delta) and
  fold-averaged test preds (test_df order, delta), and sets oof_preds['tcn'] /
  test_preds['tcn']. The existing Climber + Optuna + submission then blend it
  with NO further changes. allow_negative_weights=False => a useless TCN gets
  weight 0 (worst case = the GBT's 10.224, never worse).

All baseline patches are replicated verbatim (inline Climber, force re-train guard,
single-GPU CatBoost, SMOKE). Unified SMOKE flag drives both the GBT and the TCN.
FLIP SMOKE to False, regenerate, commit, push for the full run.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = (BASE.parent / "rogii-wellbore-work" / "pub_dwt"
       / "9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb")
OUT = BASE / "rogii-tcn-ensemble.ipynb"

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


# --- baseline inline Climber (verbatim from rogii-wellbore-baseline) ---
CLIMBER_SRC = '''\
# === Inline greedy ensemble selection (replaces external hill_climbing.Climber) ===
import numpy as _np


class Climber:
    def __init__(self, objective="minimize", eval_metric=None,
                 allow_negative_weights=False, precision=0.001,
                 score_decimal_places=3, n_jobs=-1, use_gpu=False, max_steps=100):
        self.objective = objective
        self.eval_metric = eval_metric
        self.allow_negative_weights = allow_negative_weights
        self.precision = precision
        self.score_decimal_places = score_decimal_places
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.max_steps = max_steps
        self.weights_ = None
        self.columns_ = None
        self.best_score = None

    def _score(self, y_true, y_pred):
        return round(float(self.eval_metric(y_true, y_pred)), self.score_decimal_places)

    def _better(self, cand, ref):
        if ref is None:
            return True
        return cand < ref if self.objective == "minimize" else cand > ref

    def fit(self, preds, y):
        self.columns_ = list(preds.columns)
        P = preds.values.astype(_np.float64)
        y = _np.asarray(y).astype(_np.float64)
        n, m = P.shape
        counts = _np.zeros(m); ens_sum = _np.zeros(n); n_sel = 0; best_score = None
        for _ in range(self.max_steps):
            sbs = None; sbj = -1
            for j in range(m):
                sc = self._score(y, (ens_sum + P[:, j]) / (n_sel + 1))
                if self._better(sc, sbs):
                    sbs = sc; sbj = j
            if sbj >= 0 and self._better(sbs, best_score):
                counts[sbj] += 1.0; ens_sum += P[:, sbj]; n_sel += 1; best_score = sbs
            else:
                break
        if n_sel == 0:
            single = [self._score(y, P[:, j]) for j in range(m)]
            j = int(_np.argmin(single)) if self.objective == "minimize" else int(_np.argmax(single))
            counts[j] = 1.0; n_sel = 1; best_score = single[j]
        self.weights_ = counts / counts.sum()
        self.best_score = float(best_score)
        return self

    def predict(self, preds):
        cols = [preds[c].values.astype(_np.float64) for c in self.columns_]
        return (_np.vstack(cols).T * self.weights_[None, :]).sum(axis=1)
'''

SMOKE_SRC = '''\
# Baked-in SMOKE (Kaggle injects no env). FLIP via generate_notebook.py.
SMOKE = __SMOKE__
N_TRIALS = 3 if SMOKE else 60
print(f"SMOKE={SMOKE}  N_TRIALS={N_TRIALS}")
'''

# --- TCN ensemble member, injected before `oof_preds = pd.DataFrame(oof_preds)` ---
TCN_SRC = r'''
# === TCN ensemble member: sequence model over per-row features -> oof_preds['tcn'] / test_preds['tcn'] ===
import torch, torch.nn as nn, gc
torch.manual_seed(42); np.random.seed(42)
T_EPOCHS = 2 if SMOKE else 40
T_PAT = 1 if SMOKE else 8
T_CH  = 32 if SMOKE else 128
T_NB  = 2 if SMOKE else 7
T_DROP, T_LR, T_WD, T_CLIP = 0.15, 5e-4, 1e-4, 1.0
_tfeat = list(features)
print(f"[TCN] SMOKE={SMOKE} ep={T_EPOCHS} ch={T_CH} nb={T_NB} nfeat={len(_tfeat)}")

_dev = "cpu"
try:
    if torch.cuda.is_available():
        _t = torch.zeros(8, device="cuda"); _ = float((_t + 1).sum().item()); _dev = "cuda"
        print("[TCN] GPU OK:", torch.cuda.get_device_name(0))
except Exception as e:
    print("[TCN] GPU unusable -> CPU:", str(e)[:120])
print("[TCN] device", _dev)

# standardize features (fit on train); standardize target delta
_Xall = train_df[_tfeat].to_numpy(dtype=np.float32)
_Xall[~np.isfinite(_Xall)] = np.nan
_mu = np.nanmean(_Xall, axis=0).astype(np.float32)
_sd = np.nanstd(_Xall, axis=0).astype(np.float32); _sd[_sd < 1e-6] = 1.0
del _Xall; gc.collect()
_yt = train_df['target'].to_numpy(dtype=np.float32)
_ymu = float(np.nanmean(_yt)); _ysd = float(np.nanstd(_yt)) or 1.0

def _tnorm(M):
    M = M.astype(np.float32, copy=True); M[~np.isfinite(M)] = np.nan
    M = (M - _mu) / _sd
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

def _seqs(df, has_t):
    df = df.copy(); df['_ri'] = df['id'].str.rsplit('_', n=1).str[-1].astype(int)
    out = []
    for wid, gd in df.groupby('well', sort=False):
        gd = gd.sort_values('_ri')
        it = {'wid': wid, 'X': _tnorm(gd[_tfeat].to_numpy()), 'ids': gd['id'].to_numpy()}
        if has_t:
            it['t'] = (gd['target'].to_numpy(dtype=np.float32) - _ymu) / _ysd
        out.append(it)
    return out

_tr = _seqs(train_df, True)
_te = _seqs(test_df, False)
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
    def __init__(s, cin, c, nb, dr):
        super().__init__()
        s.inp = nn.Conv1d(cin, c, 1)
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

_oof_id = {}; _test_sum = {}; _nf = 0; _fb = []
_idx = np.arange(len(_tr))
for _f, (_tri, _vai) in enumerate(CFG.cv.split(_idx, groups=_grp)):
    _m = _TCN(len(_tfeat), T_CH, T_NB, T_DROP).to(_dev)
    _opt = torch.optim.Adam(_m.parameters(), lr=T_LR, weight_decay=T_WD)
    _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=T_EPOCHS)
    _best = 1e9; _bs = None; _bad = 0; _tri = np.array(_tri)
    for _ep in range(T_EPOCHS):
        _m.train(); np.random.shuffle(_tri)
        for _j in _tri:
            s = _tr[_j]; x = _tx(s); t = torch.tensor(s['t'][None], dtype=torch.float32, device=_dev)
            _opt.zero_grad(); _l = _hub(_m(x), t); _l.backward()
            torch.nn.utils.clip_grad_norm_(_m.parameters(), T_CLIP); _opt.step()
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
        if _bad >= T_PAT:
            break
    _m.load_state_dict(_bs); _m.eval()
    with torch.no_grad():
        for _j in _vai:
            s = _tr[_j]; pr = _m(_tx(s)).cpu().numpy()[0] * _ysd + _ymu
            for _i, _id in enumerate(s['ids']):
                _oof_id[_id] = float(pr[_i])
        for s in _te:
            pr = _m(_tx(s)).cpu().numpy()[0] * _ysd + _ymu
            for _i, _id in enumerate(s['ids']):
                _test_sum[_id] = _test_sum.get(_id, 0.0) + float(pr[_i])
    _nf += 1; _fb.append(_best)
    print(f"[TCN] fold{_f} best toe RMSE = {_best:.4f}")
print("[TCN] CV toe RMSE mean =", float(np.mean(_fb)), "| folds", [round(b, 3) for b in _fb])

# align to train_df / test_df row order (delta units, same as GBT oof)
tcn_oof = train_df['id'].map(_oof_id).to_numpy(dtype=np.float32)
_te_mean = {k: v / _nf for k, v in _test_sum.items()}
tcn_test = test_df['id'].map(_te_mean).to_numpy(dtype=np.float32)
assert not np.isnan(tcn_oof).any(), "TCN OOF has unmapped train rows"
assert not np.isnan(tcn_test).any(), "TCN test has unmapped test rows"
oof_preds['tcn'] = tcn_oof
test_preds['tcn'] = tcn_test
print(f"[TCN] injected -> oof_preds keys = {list(oof_preds.keys())}")
del _tr, _te; gc.collect()
'''


def main():
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    report = []

    # 1. drop hill_climbing import
    i_imp = find_cell(cells, "from hill_climbing import Climber")
    imp = src_str(cells[i_imp])
    assert "from hill_climbing import Climber\n" in imp
    set_src(cells[i_imp], imp.replace("from hill_climbing import Climber\n", ""))
    report.append(f"cell {i_imp}: removed hill_climbing import")

    # 2. inline Climber after imports
    cells.insert(i_imp + 1, code_cell(CLIMBER_SRC))
    report.append(f"inserted inline Climber at {i_imp + 1}")

    # 3. SMOKE/config after CFG
    i_cfg = find_cell(cells, "class CFG:")
    cells.insert(i_cfg + 1, code_cell(SMOKE_SRC.replace("__SMOKE__", "True" if SMOKE else "False")))
    report.append(f"inserted SMOKE cell at {i_cfg + 1}")

    # 4. N_TRIALS
    i_opt = find_cell(cells, "study.optimize(objective, n_trials=500")
    o = src_str(cells[i_opt])
    no = o.replace("study.optimize(objective, n_trials=500, n_jobs=-1)",
                   "study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)")
    assert no != o
    set_src(cells[i_opt], no)
    report.append(f"cell {i_opt}: n_trials -> N_TRIALS")

    # 5a. SMOKE data subsample + train.csv nrows
    i_load = find_cell(cells, "test_paths = sorted((CFG.dataset_path / \"test\")")
    ls = src_str(cells[i_load])
    read_anchor = ('train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", '
                   'low_memory=False)')
    assert read_anchor in ls
    ls = ls.replace(read_anchor,
                    'train_df = pd.read_csv(CFG.artifacts_path / "data" / "train.csv", '
                    'low_memory=False, nrows=(60000 if SMOKE else None))')
    anchor = ('test_paths = sorted((CFG.dataset_path / "test").glob'
              "('*__horizontal_well.csv'))")
    assert anchor in ls
    smoke_load = (
        'if SMOKE:\n'
        "    train_df = train_df.groupby('well', group_keys=False).head(300).reset_index(drop=True)\n"
        '\n' + anchor + '\n'
        'if SMOKE:\n'
        '    test_paths = test_paths[:6]\n'
    )
    ls2 = ls.replace(anchor + "\n", smoke_load)
    assert ls2 != ls
    set_src(cells[i_load], ls2)
    report.append(f"cell {i_load}: SMOKE subsample + nrows")

    # 5b. SMOKE shrink GBT rounds
    i_par = find_cell(cells, "lgb_params_base = dict(")
    ps = src_str(cells[i_par])
    assert "lgb_params = [" in ps and "cb_params = [" in ps
    set_src(cells[i_par], ps.rstrip("\n") + (
        "\n\nif SMOKE:\n"
        "    for _p in lgb_params:\n        _p['n_estimators'] = 50\n"
        "    for _p in cb_params:\n        _p['iterations'] = 50\n"))
    report.append(f"cell {i_par}: SMOKE shrink GBT rounds")

    # 5c. force re-train (models.pkl file guard)
    guard_old = 'if (CFG.artifacts_path / "models" / name).exists():'
    guard_new = 'if (CFG.artifacts_path / "models" / name / "models.pkl").exists():'
    ng = 0
    for ci, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        if guard_old in cs:
            set_src(c, cs.replace(guard_old, guard_new)); ng += cs.count(guard_old)
    assert ng >= 2, f"expected >=2 guards, found {ng}"
    report.append(f"forced re-train ({ng} guards)")

    # 5d. single-GPU CatBoost
    nd = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        cs = src_str(c)
        if 'devices="0:1"' in cs:
            set_src(c, cs.replace('devices="0:1"', 'devices="0"')); nd += 1
    assert nd >= 1
    report.append(f"patched {nd} CatBoost devices -> single-GPU")

    # 6. INJECT TCN cell right before `oof_preds = pd.DataFrame(oof_preds)`
    i_df = find_cell(cells, "oof_preds = pd.DataFrame(oof_preds)")
    cells.insert(i_df, code_cell(TCN_SRC))
    report.append(f"inserted TCN ensemble cell at {i_df} (before oof_preds DataFrame)")

    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # round-trip + sanity
    rt = json.load(open(OUT, encoding="utf-8"))
    full = "\n".join(src_str(c) for c in rt["cells"] if c["cell_type"] == "code")
    assert "\nfrom hill_climbing import Climber" not in ("\n" + full)
    assert "class Climber:" in full
    assert "n_trials=N_TRIALS" in full
    assert "oof_preds['tcn'] = tcn_oof" in full
    assert "test_preds['tcn'] = tcn_test" in full
    assert 'devices="0:1"' not in full
    assert "task_type=\"GPU\"" in full
    # TCN cell must come before the DataFrame conversion
    assert full.index("oof_preds['tcn'] = tcn_oof") < full.index("oof_preds = pd.DataFrame(oof_preds)")
    print("=== PATCH REPORT ===")
    for r in report:
        print(" -", r)
    print(f"=== {OUT.name}: {len(rt['cells'])} cells (SMOKE={SMOKE}) | round-trip + asserts OK ===")


if __name__ == "__main__":
    main()

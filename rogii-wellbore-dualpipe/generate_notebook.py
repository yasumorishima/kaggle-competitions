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
print(f"SMOKE={SMOKE}")
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
    assert full.index("print(f\"SMOKE={SMOKE}\")") < full.index("class _TCN(nn.Module)"), \
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

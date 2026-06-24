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
# oof_preds / test_preds are still dicts here -> add the member; the Ridge stack weights it.
oof_preds["tcn"] = _tcn_oof
test_preds["tcn"] = _tcn_test
print(f"[TCN] member added | OOF residual RMSE={float(np.sqrt(np.mean((_tcn_oof - y.values) ** 2))):.4f} | members={list(oof_preds.keys())}")
del _tr, _te; gc.collect()
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
    # existing override / final blend markers intact (untouched)
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

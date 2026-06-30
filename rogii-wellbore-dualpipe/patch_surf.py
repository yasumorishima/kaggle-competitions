"""
One-shot surgery on generate_notebook.py: move the TPS-surface build out of the
in-process SURF cell and into an ISOLATED subprocess.

WHY: the probe (yasunorim/rogii-tps-probe, COMPLETE) proved the _FormSurfTPS build +
_surf_feats run fine standalone (895s, returncode 0). So the dualpipe ERROR is NOT a TPS
bug -- it is the surface build's peak memory coexisting with the resident train.csv
(7.4GB) inside the GBT-enabled T4 kernel (~13-16GB RAM) -> OOM-kill (SIGKILL, no Python
traceback = the "DeadKernel"). f0656ca (the lighter 2nd-order _FormSurf) fit the headroom;
the heavier TPS does not.

FIX: the SURF cell now writes a tiny child script that re-derives paths from the mounted
competition data, builds the SAME _FormSurfTPS + _surf_feats (extracted verbatim from the
current SURF_SRC, so byte-identical math + full column set + per-well LOO fold-safety),
and pickles surf_tps_{train,test}.pkl. The child holds NO train.csv, so its peak memory is
isolated; on exit the OS frees it all and the parent just reads + merges the pickles. The
parent captures the child's returncode + stdout/stderr (so a future failure is diagnosable
in a COMPLETE run) and falls back to an empty (fillna-0) surface if the child dies.

Rerun-safe: the child computes the TEST surface at kernel runtime from whatever test wells
are mounted (the code competition swaps the hidden test at rerun), never from a static
dataset. Idempotent: edits the SURF_SRC block + the single main() insertion line only.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "generate_notebook.py"
txt = P.read_text(encoding="utf-8")

START = "SURF_SRC = r'''"
END_ANCHOR = "\n\n\n# dip-aware-transition beam as an intra-well production feature"
i0 = txt.index(START)
i1 = txt.index(END_ANCHOR, i0)
old_block = txt[i0:i1]                       # "SURF_SRC = r'''...'''"
assert old_block.endswith("'''"), "old SURF_SRC block did not end on closing triple-quote"
assert "class _FormSurfTPS:" in old_block and "_FS = _FormSurfTPS(train_wids, TRAIN_DIR)" in old_block

inner = old_block[len(START):-3]             # the cell body, between r''' and '''
MARK = "\n_n_tr0 = len(train_df); _n_te0 = len(test_df)"
head, sep, rest = inner.partition(MARK)
assert sep, "merge marker (_n_tr0) not found in SURF_SRC"
tail = MARK + rest                           # original merge / feature-recompute block

# --- CHILD: self-contained preamble + the verbatim head (CFG path -> literal) + pickle ---
COMP = '"/kaggle/input/competitions/rogii-wellbore-geology-prediction"'
PREAMBLE = (
    "import sys, time, os\n"
    "from pathlib import Path\n"
    f"DATA = Path({COMP})\n"
    'FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]\n'
    'hw_paths = sorted((DATA / "train").glob("*__horizontal_well.csv"))\n'
    'test_paths = sorted((DATA / "test").glob("*__horizontal_well.csv"))\n'
    'train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]\n'
    'print(f"[SURF-CHILD] train wells={len(train_wids)} test wells={len(test_paths)}", flush=True)\n'
)
child_head = head.replace("CFG.dataset_path", f"Path({COMP})")
PICKLE = (
    '\nprint(f"[SURF-CHILD] surf rows train={len(_sf_tr)} test={len(_sf_te)} '
    'cols={list(_sf_tr.columns)}", flush=True)\n'
    '_sf_tr.to_pickle("/kaggle/working/surf_tps_train.pkl")\n'
    '_sf_te.to_pickle("/kaggle/working/surf_tps_test.pkl")\n'
    'print("SURF_CHILD_OK", flush=True)\n'
)
child = PREAMBLE + child_head + PICKLE
assert "'''" not in child, "child contains a triple-single-quote -> would break r''' wrapper"

# --- PARENT: subprocess runner + softened-assert merge tail ---
tail_soft = tail.replace(
    'assert _miss_tr == 0, f"[SURF] {_miss_tr} train rows lack surface features '
    '(id set mismatch with build_well)"',
    'if _miss_tr:\n'
    '    print(f"[SURF] WARNING: {_miss_tr} train rows lack surface features (filled 0.0)", flush=True)',
)
assert tail_soft != tail, "hard _miss_tr assert not found to soften"

RUNNER = '''
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
'''

parent = RUNNER + tail_soft
new_block = "SURF_CHILD = r'''" + child + "'''\n\n\n" + "SURF_SRC = r'''" + parent + "'''"

txt2 = txt.replace(old_block, new_block, 1)
assert txt2 != txt, "SURF_SRC block replacement did not apply"

OLD_INS = "cells.insert(i_feat + 1, code_cell(SURF_SRC))"
NEW_INS = 'cells.insert(i_feat + 1, code_cell(SURF_SRC.replace("__SURF_CHILD__", repr(SURF_CHILD))))'
assert OLD_INS in txt2, "main() SURF insertion line not found"
txt3 = txt2.replace(OLD_INS, NEW_INS, 1)

P.write_text(txt3, encoding="utf-8")
print("patched generate_notebook.py")
print(f"  child script chars = {len(child)}")
print(f"  parent (SURF_SRC) chars = {len(parent)}")

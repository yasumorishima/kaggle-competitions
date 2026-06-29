"""
Generate an ISOLATED diagnostic kernel for the ROGII dualpipe TPS-surface crash.

WHY: the dualpipe kernel (yasunorim/rogii-dualpipe) has been in KernelWorkerStatus.ERROR
since the surface-TPS injection. Kaggle does NOT save the output of a FAILED run, and
`kaggle kernels output` returns the last *successful* run's stale log, so the real crash
(suspected a native LAPACK segfault / OOM-kill with no Python traceback, dying ~1 min in,
before the 75-min GBT retrain) has never been observed.

HOW: this probe runs the EXACT `_FormSurfTPS` build + `_surf_feats` (copied from
generate_notebook.py SURF_SRC) inside a CHILD subprocess. A segfault/OOM-kill only kills
the child; the PARENT survives, captures the child's returncode (negative = killed by
signal N) and the last flushed stdout line (localizes where it died), prints them, and
exits 0 -> the kernel reaches COMPLETE -> Kaggle SAVES the log. This is the only way to
get diagnostics out of a natively-crashing kernel.

Pure numpy/scipy/pandas, CPU only, competition data only. Writes rogii-tps-probe.ipynb.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "rogii-tps-probe.ipynb"

# ---- CHILD: exact replica of SURF_SRC's TPS build + surf_feats, with progress prints ----
CHILD = r'''
import sys, time
import numpy as np, pandas as pd
from pathlib import Path

print("[CHILD] start", flush=True)
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
TRAIN_DIR = DATA / "train"
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
hw_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
print(f"[CHILD] train wells={len(train_wids)} numpy={np.__version__}", flush=True)

_LAM = 5.0


def _tps_U(r2):
    return np.where(r2 > 1e-12, 0.5 * r2 * np.log(np.maximum(r2, 1e-30)), 0.0)


class _FormSurfTPS:
    def __init__(self, wids, data_dir):
        raw = []
        for wid in wids:
            p = data_dir / (wid + "__horizontal_well.csv")
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
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
            rows_x.append(np.array([df["X"].mean()], np.float64))
            rows_y.append(np.array([df["Y"].mean()], np.float64))
            rows_w.append(np.full(1, c, np.int32))
            for f in FORMATIONS:
                rows_F[f].append(np.array([np.median(df[f].to_numpy(np.float64))], np.float64))
        self.code = code
        self.nx = np.concatenate(rows_x); self.ny = np.concatenate(rows_y)
        self.nw = np.concatenate(rows_w)
        self.nF = {f: np.concatenate(rows_F[f]) for f in FORMATIONS}
        N = len(self.nx)
        self.mx = float(self.nx.mean()); self.my = float(self.ny.mean())
        self.sx = float(self.nx.std() or 1.0); self.sy = float(self.ny.std() or 1.0)
        self._Xn = (self.nx - self.mx) / self.sx
        self._Yn = (self.ny - self.my) / self.sy
        print(f"[CHILD][SURF-TPS] wells={W} N={N} saddle={(N+3)}x{(N+3)} "
              f"({(N+3)**2*8/1e6:.0f}MB) lam={_LAM}", flush=True)
        t0 = time.time(); self._full = self._solve(np.ones(N, bool))
        print(f"[CHILD][SURF-TPS] full solve {time.time()-t0:.2f}s", flush=True)
        t0 = time.time(); self._loo = {}
        uniq = np.unique(self.nw)
        for ii, c in enumerate(uniq):
            self._loo[int(c)] = self._solve(self.nw != c)
            if ii % 100 == 0:
                print(f"[CHILD][SURF-TPS] LOO {ii}/{len(uniq)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[CHILD][SURF-TPS] {len(self._loo)} per-well LOO solves {time.time()-t0:.1f}s", flush=True)

    def _solve(self, mask):
        Xn = self._Xn[mask]; Yn = self._Yn[mask]; m = len(Xn)
        dx = Xn[:, None] - Xn[None, :]; dy = Yn[:, None] - Yn[None, :]
        K = _tps_U(dx * dx + dy * dy) + _LAM * np.eye(m)
        P = np.column_stack([np.ones(m), Xn, Yn])
        Asys = np.zeros((m + 3, m + 3))
        Asys[:m, :m] = K; Asys[:m, m:] = P; Asys[m:, :m] = P.T
        Asys += 1e-8 * np.eye(m + 3)
        RHS = np.zeros((m + 3, len(FORMATIONS)))
        for j, f in enumerate(FORMATIONS):
            RHS[:m, j] = self.nF[f][mask]
        SOL = np.linalg.solve(Asys, RHS)
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
        xy = np.atleast_2d(xy)
        code = self.code.get(wid) if wid is not None else None
        b = self._loo[code] if (code is not None and code in self._loo) else self._full
        return {f: self._eval(b, xy, f).astype(np.float64) for f in FORMATIONS}


t0 = time.time()
_FS = _FormSurfTPS(train_wids, TRAIN_DIR)
print(f"[CHILD] FS built nodes={len(_FS.nx)} ({time.time()-t0:.0f}s)", flush=True)


def _surf_feats(paths, is_train):
    recs = []
    for k, p in enumerate(paths):
        wid = p.stem.replace("__horizontal_well", "")
        try:
            hw = pd.read_csv(p)
        except Exception:
            continue
        if "TVT_input" not in hw.columns:
            continue
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10:
            continue
        last_tvt = float(kn["TVT_input"].to_numpy()[-1])
        swid = wid if is_train else None
        xk = kn[["X", "Y"]].to_numpy(np.float64); zk = kn["Z"].to_numpy(np.float64)
        tk = kn["TVT_input"].to_numpy(np.float64)
        xe = ev[["X", "Y"]].to_numpy(np.float64); ze = ev["Z"].to_numpy(np.float64)
        Sk = _FS.predict(xk, swid); Se = _FS.predict(xe, swid)
        tvts = {}; bs = []
        for f in FORMATIONS:
            b_f = float(np.median(tk + zk - Sk[f])); bs.append(b_f); tvts[f] = (-ze + Se[f] + b_f)
        rng = float(tk.max() - tk.min()); cap = 3.0 * rng + 1000.0
        deltas = {f: np.clip(tvts[f] - last_tvt, -cap, cap) for f in FORMATIONS}
        ids = [f"{wid}_{i}" for i in ev.index]
        d = {"id": ids}
        for f in FORMATIONS:
            d[f"tvtS_{f}_d"] = deltas[f].astype(np.float32)
        recs.append(pd.DataFrame(d))
        if k % 100 == 0:
            print(f"[CHILD][surf_feats] {k}/{len(paths)}", flush=True)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame({"id": []})


t0 = time.time()
sf = _surf_feats(hw_paths, True)
print(f"[CHILD] surf_feats rows={len(sf)} cols={list(sf.columns)} ({time.time()-t0:.0f}s)", flush=True)
print("PROBE_OK", flush=True)
'''

# ---- PARENT cell: write child, run in subprocess, report, ALWAYS complete ----
PARENT = r'''# === ROGII TPS-surface crash probe (subprocess-isolated; survives segfault/OOM) ===
import subprocess, sys, time

CHILD_SRC = __CHILD__

with open("/kaggle/working/tps_probe.py", "w") as _fh:
    _fh.write(CHILD_SRC)

print("=== running TPS-surface build in an ISOLATED subprocess ===", flush=True)
_t0 = time.time()
rc = None; out = ""; err = ""
try:
    _r = subprocess.run([sys.executable, "-u", "/kaggle/working/tps_probe.py"],
                        capture_output=True, text=True, timeout=2400)
    rc, out, err = _r.returncode, _r.stdout, _r.stderr
except subprocess.TimeoutExpired as _e:
    out = _e.stdout if isinstance(_e.stdout, str) else (_e.stdout or b"").decode("utf-8", "replace")
    err = _e.stderr if isinstance(_e.stderr, str) else (_e.stderr or b"").decode("utf-8", "replace")
    print("!!! CHILD TIMED OUT (2400s) !!!", flush=True)

print(f"=== child finished: returncode={rc}  elapsed={time.time()-_t0:.0f}s ===", flush=True)
print("---------------- CHILD STDOUT (tail) ----------------", flush=True)
print(out[-12000:], flush=True)
print("---------------- CHILD STDERR (tail) ----------------", flush=True)
print(err[-12000:], flush=True)
print("-----------------------------------------------------", flush=True)
if rc is None:
    print(">>> VERDICT: child TIMED OUT (surface build too slow or hung)", flush=True)
elif rc < 0:
    print(f">>> VERDICT: child KILLED BY SIGNAL {-rc} "
          f"(segfault/SIGSEGV=11, OOM-kill/SIGKILL=9). The last CHILD STDOUT line "
          f"localizes the death point.", flush=True)
elif rc != 0:
    print(">>> VERDICT: child raised a PYTHON EXCEPTION (see CHILD STDERR traceback).", flush=True)
else:
    print(">>> VERDICT: child COMPLETED OK -- the TPS-surface build is NOT the crash. "
          "Bisect the next stage (merge / downstream pipeline).", flush=True)
print("DONE_PARENT", flush=True)
'''

PARENT = PARENT.replace("__CHILD__", repr(CHILD))

nb = {
    "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": PARENT},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")

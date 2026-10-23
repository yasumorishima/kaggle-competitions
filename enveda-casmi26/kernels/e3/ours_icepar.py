"""ours_icepar (yasunorim, 10-23): ICEBERG on a CPU kernel in N parallel runner processes.

On the CPU the base's single ICEBERG runner (torch, 4 threads) predicts about 2-3 spectra/s and reaches only about a third
of the hidden test's (candidate, collision energy) jobs inside its 5,400 s budget; molecules whose chunk did not finish get
no ICEBERG score. Measured in a 4-core container (10-23): 1 runner x 4 threads 3.3 pred/s, 2 x 2 threads 4.0, 4 x 1 thread
5.2 (x1.6). This splits the runner input round-robin (the base's priority order is kept inside each shard), runs one
runner per shard with --threads 1 and the same budget (chunks of 500 predictions instead of 3,000, so less
work is lost when the budget stops a shard mid-chunk), and merges the outputs and the stage metadata into the base's
workdir, so every later cell reads them unchanged. The time budget is not raised.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor


def run(run_ice, pkg, items, workdir, device, budget_s, site, extra_args=(), n=4, log=print):
    if n <= 1 or len(items) < 2 * n:
        return run_ice(pkg, items, workdir=workdir, device=device, budget_s=budget_s, site=site, extra_args=extra_args)
    shards = [items[k::n] for k in range(n)]
    old = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
    for k in old:
        os.environ[k] = "1"
    try:
        def one(k):
            return run_ice(pkg, shards[k], workdir=os.path.join(workdir, f"s{k}"), device=device, budget_s=budget_s,
                           site=site, extra_args=list(extra_args) + ["--threads", "1", "--chunk-jobs", "500"])
        with ThreadPoolExecutor(n) as ex:
            outs = list(ex.map(one, range(n)))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    res = {}
    for o in outs:
        res.update(o or {})
    meta = dict(status="ok", n_mols=0, n_mols_covered=0, n_mols_scored=0, n_cands=0, n_scored=0, n_pred=0, errors=[],
                shards=n)
    for k in range(n):
        f = os.path.join(workdir, f"s{k}", "ice_out.json.meta.json")
        if not os.path.exists(f):
            meta["errors"].append(f"shard {k}: no meta")
            continue
        m = json.load(open(f))
        for c in ("n_mols", "n_mols_covered", "n_mols_scored", "n_cands", "n_scored", "n_pred"):
            meta[c] += int(m.get(c) or 0)
        meta["errors"] += [f"shard {k}: {e}" for e in (m.get("errors") or [])][:5]
        if m.get("status") != "ok":
            meta["status"] = m.get("status") if meta["status"] in ("ok", "budget") else meta["status"]
        for c in ("device", "rdkit", "torch"):
            meta.setdefault(c, m.get(c))
        meta["seconds"] = max(meta.get("seconds", 0), m.get("seconds") or 0)
    if meta["n_mols_scored"] == 0 and meta["status"] == "ok":
        meta["status"] = "budget"
    os.makedirs(workdir, exist_ok=True)
    json.dump(res, open(os.path.join(workdir, "ice_out.json"), "w"))
    json.dump(meta, open(os.path.join(workdir, "ice_out.json.meta.json"), "w"))
    log("ours_icepar:", {c: meta.get(c) for c in ("status", "shards", "n_mols", "n_mols_scored", "n_cands", "n_scored",
                                                   "n_pred", "seconds")})
    return res


def n1par(items, budget, work, gl_pkg, device, site, wheels, extra=(), n=4, log=print):
    """The base's n1_run (glb_driver.py, the N1 / challenger GLACIER proxies) split over n parallel driver processes
    (OMP / MKL threads 1 each; input order kept inside each shard). Returns (out, meta, done) like n1_run."""
    import subprocess
    import sys
    import time
    if n <= 1 or len(items) < 2 * n:
        n = 1
    shards = [items[k::n] for k in range(n)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1") if n > 1 else dict(os.environ)
    procs, t0 = [], time.time()
    for k in range(n):
        d = os.path.join(work, f"p{k}")
        os.makedirs(d, exist_ok=True)
        fin, fout = os.path.join(d, "n1_in.json"), os.path.join(d, "n1_out.json")
        for f in (fout, fout + ".meta.json"):
            if os.path.exists(f):
                os.remove(f)
        json.dump(shards[k], open(fin, "w"))
        cmd = [sys.executable, os.path.join(work, "glb_driver.py"), gl_pkg, fin, fout, "--device", device,
               "--budget", str(float(budget)), "--site", site, "--wheels", wheels] + list(extra)
        procs.append((subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                       env=env), fout))
    out, meta = {}, dict(status="ok", done=[], n_pred=0, n_pred_fail=0, n_spec_covered=0, n_spec_merged=0, errors=[],
                         shards=n)
    for k, (p, fout) in enumerate(procs):
        try:
            so, _ = p.communicate(timeout=max(60.0, float(budget) + 600 - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            p.kill(); so, _ = p.communicate()
            meta["errors"].append(f"shard {k}: timeout")
        log(f"[ours n1par] shard {k}:", (so or "")[-800:])
        try:
            m = json.load(open(fout + ".meta.json"))
        except Exception as e:
            meta["errors"].append(f"shard {k}: meta {e!r}"); m = {}
        try:
            out.update(json.load(open(fout)))
        except Exception as e:
            meta["errors"].append(f"shard {k}: out {e!r}")
        meta["done"] += list(m.get("done") or [])
        for c in ("n_pred", "n_pred_fail", "n_spec_covered", "n_spec_merged"):
            meta[c] += int(m.get(c) or 0)
        meta["errors"] += [f"shard {k}: {e}" for e in (m.get("errors") or [])][:3]
        if m.get("status") != "ok":
            meta["status"] = m.get("status") or "error"
        meta["seconds"] = round(time.time() - t0, 2)
    return out, meta, set(meta["done"])

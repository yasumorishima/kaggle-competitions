"""Gemma 4 agent: healthy-task control on CPU (no model, no GPU).

For each of the 129 published tasks, run the official Phase 2 verification (swegemma.harness.verification)
twice: with an empty agent patch (the baseline code is tested) and with the task's
gold patch. A task is healthy when the no-op fails and the gold passes; only healthy tasks count when comparing
agent configs. One RESULT line per task, then a compact HEALTH table at the end (the log API keeps the tail).
"""
import asyncio
import glob
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()
WORKERS = int(os.environ.get("G4_WORKERS", "1"))


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "OTEL_SDK_DISABLED": "true"})
WH = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent-wheelhouse", recursive=True), key=len)[0])
SKIP = ("vllm", "torch", "cutlass", "flash", "xformers", "triton", "nvidia", "cu12", "flashinfer")
wheels = sorted(str(w) for w in WH.glob("*.whl") if not any(s in w.name.lower() for s in SKIP))
log("wheels", len(wheels))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels], check=False)
importlib.invalidate_caches()

from swegemma.models import load_tasks  # noqa: E402
import swegemma.harness.verification as V  # noqa: E402

DATA = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True), key=len)[0]).parent
WORK = Path("/kaggle/working")
# print the harness source to the log (output file downloads are blocked from the cloud container)
if os.environ.get("G4_DUMP", "0") == "1":
    import swegemma
    log("swegemma files", sorted(str(p.relative_to(Path(swegemma.__file__).parent)) for p in Path(swegemma.__file__).parent.rglob("*.py")))
    for mod in ("swegemma.evaluate",):
        print(f"=== SOURCE {mod}\n" + inspect.getsource(importlib.import_module(mod)), flush=True)
    print("=== SOURCE verify_task\n" + inspect.getsource(V.verify_task), flush=True)
from swegemma.config import EvalConfig, build_submission_limits  # noqa: E402
from swegemma.deduplication import resolve_task_snapshot_paths  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402

limits, gen = build_submission_limits()


def make_ev(models):
    return Evaluator(EvalConfig(
        tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots", results_dir=WORK / "results",
        submission_dir=DATA / "sample_submission", models=models, sandbox="subprocess", limits=limits,
        generation_constraints=gen, graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"),
        wheels_dir=DATA / "wheels", verbose=False))


EV = None
for models in ({}, None, []):
    try:
        EV = make_ev(models)
        break
    except Exception as e:
        log("EvalConfig models=", repr(models), "failed:", repr(e)[:300])
if EV is None:
    raise SystemExit("could not build an Evaluator")
NOOP = ""          # verify_task still runs the tests on the baseline when the agent patch is empty


def call(task, patch):
    """the verify_task call Evaluator.evaluate_task makes after Phase 1, with our patch as the agent patch"""
    task = EV._hydrate_task_from_secret(task)
    snap, base, pp = resolve_task_snapshot_paths(EV.config.snapshots_dir, task.instance_id, task.repo)
    if not snap.exists():
        raise FileNotFoundError(str(snap))
    return asyncio.run(V.verify_task(EV.docker, EV.config, task, snap, base_snapshot_path=base, patch_path=pp,
                                     agent_patch=patch, start_time=time.perf_counter()))


tasks = load_tasks(DATA / "tasks.jsonl")
ONLY = [x for x in os.environ.get("G4_ONLY", "fastapi_15030,fastapi_14962,fastapi_15023,fastapi_14964,fastapi_14978,requests_7505,requests_7502,requests_7315,requests_7427,requests_7433,requests_7309,requests_7205,rich_4070,fastapi_14851,requests_7328,fastapi_12942,fastapi_14099,fastapi_14609,fastapi_14349,fastapi_14512,fastapi_11355,fastapi_14485,fastapi_14482,fastapi_14459,fastapi_14455,fastapi_14430,fastapi_14361,fastapi_13207,fastapi_14360,fastapi_14356,fastapi_14262,fastapi_14266,fastapi_14246,fastapi_14186,fastapi_13713,fastapi_14077,httpx_3672,fastapi_14791,fastapi_15785,fastapi_14371,fastapi_14605,fastapi_15763,fastapi_14583,fastapi_14297,fastapi_15745,fastapi_14303,fastapi_14953,fastapi_13786,requests_6644,requests_6629,requests_6589,requests_6757,rich_3486,rich_3296,requests_6592,rich_3469,rich_3130,rich_3064").split(",") if x and x != "__ONLY__"]
if ONLY:
    tasks = [t for t in tasks if t.instance_id in ONLY]
log("tasks", len(tasks), "workers", WORKERS)


def one(t):
    row = dict(id=t.instance_id, repo=t.repo)
    for tag, patch in (("noop", NOOP), ("gold", t.patch)):
        t1 = time.time()
        try:
            r = call(t, patch)
            ok = bool(getattr(r, "resolved", r.get("resolved") if isinstance(r, dict) else r))
            err = getattr(r, "error", None) or (r.get("error") if isinstance(r, dict) else None)
            row[tag] = int(ok)
            if err:
                row[tag + "_err"] = str(err)[:200]
            if tag == "gold" and not ok:     # why the reference fix fails here: exit code and the end of the test output
                row["gold_exit"] = getattr(r, "test_exit_code", None)
                row["gold_out"] = (getattr(r, "test_output", "") or "")[-1500:]
        except Exception as e:
            row[tag] = -1
            row[tag + "_err"] = repr(e)[:300]
        row[tag + "_s"] = round(time.time() - t1)
    row["healthy"] = int(row["noop"] == 0 and row["gold"] == 1)
    print("RESULT " + json.dumps(row), flush=True)
    return row


with ThreadPoolExecutor(WORKERS) as pool:
    rows = list(pool.map(one, tasks))
json.dump(rows, open(WORK / "health.json", "w"))
by = {}
for r in rows:
    b = by.setdefault(r["repo"], [0, 0])
    b[0] += r["healthy"]
    b[1] += 1
print("SUMMARY " + json.dumps(dict(healthy=sum(r["healthy"] for r in rows), tasks=len(rows),
                                    noop_pass=sum(r["noop"] == 1 for r in rows), gold_fail=sum(r["gold"] == 0 for r in rows),
                                    errors=sum(-1 in (r["noop"], r["gold"]) for r in rows), by_repo=by)), flush=True)
print("HEALTH " + " ".join(f"{r['id']}:{r['noop']}{r['gold']}" for r in rows), flush=True)
log("done")

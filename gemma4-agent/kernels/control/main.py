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
WORKERS = int(os.environ.get("G4_WORKERS", "3"))


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

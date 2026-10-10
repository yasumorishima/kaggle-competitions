"""Gemma 4 agent: local paired evaluation on the published tasks (built by localeval/build.py).

Serves gemma-4-31b-it-qat-w4a16-ct with the competition's vLLM wheel and runs the official swegemma
Evaluator (Phase 1 agent + Phase 2 pytest verification, subprocess sandbox) for each agent config in
CONFIGS on the same tasks, so configs are compared task by task. Setup follows the official
getting-started notebook. One JSON line per task goes to stdout (RESULT ...), a summary at the end.
"""
import asyncio
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import logging
logging.getLogger("asyncio").setLevel(logging.CRITICAL)   # aiohttp "Unclosed client session" noise
T0 = time.time()
CONFIGS = __CONFIGS__          # name -> {relative path: text}; "@sample" = the official sample_submission
LIMIT = int(os.environ.get("G4_LIMIT", "__LIMIT__"))
WORKERS = int(os.environ.get("G4_WORKERS", "__WORKERS__"))
SHARD = "__SHARD__"            # "k/M": tasks with index % M == k


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


os.environ.update({
    "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_NO_USAGE_STATS": "1",
    "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
WHEELHOUSE = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent-wheelhouse", recursive=True), key=len)[0])
for pat in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth", "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    for p in glob.glob(pat):
        try:
            os.unlink(p)
        except OSError:
            pass
tmp = Path("/tmp/wheelhouse")
tmp.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    if not (tmp / name).exists():
        os.symlink(w, tmp / name)
log("python", sys.version.split()[0], "wheels", len(list(tmp.glob("*.whl"))))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                *sorted(str(w) for w in tmp.glob("*.whl"))], check=True)
importlib.invalidate_caches()
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

import litellm  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from adk_submission import VllmConfig, VllmServer, discover_adapters  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
def _find_attr(name, roots=("swegemma", "adk_submission", "adk_eval_core")):
    """the wheelhouse moves helpers between modules (swegemma.models.discovery is gone after the 10-10 update)"""
    import pkgutil
    for root in roots:
        try:
            pkg = importlib.import_module(root)
        except Exception:
            continue
        if hasattr(pkg, name):
            return getattr(pkg, name)
        for mi in pkgutil.walk_packages(pkg.__path__, root + "."):
            try:
                mod = importlib.import_module(mi.name)
            except Exception:
                continue
            if hasattr(mod, name):
                print("found", name, "in", mi.name, flush=True)
                return getattr(mod, name)
    raise ImportError(name)


ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits = (
    _find_attr(n) for n in ("ALLOWED_ADAPTER_EXTENSIONS", "EvalConfig", "build_submission_limits"))
Evaluator = _find_attr("Evaluator")

# diagnostics (ours, no behaviour change): when ADK rejects a tool call for a missing mandatory argument, print the
# argument names and short values that did arrive (localeval-6c: 625 of 775 edit_file calls lost old_string)
try:
    from google.adk.tools.function_tool import FunctionTool as _FT
    _ft_run0 = _FT.run_async

    async def _ft_run(self, *, args, tool_context):
        out = await _ft_run0(self, args=args, tool_context=tool_context)
        if isinstance(out, dict) and "mandatory input parameters are not present" in str(out.get("error", "")):
            shown = {k: (repr(v)[:80] if not isinstance(v, str) else f"str[{len(v)}] {v[:60]!r}") for k, v in (args or {}).items()}
            print("MISSINGARG", self.name, json.dumps(shown)[:600], flush=True)
        return out
    _FT.run_async = _ft_run
except Exception as _e:
    print("MISSINGARG hook failed", repr(_e))
load_tasks = _find_attr("load_tasks")

try:
    validate_single_declared_model = _find_attr("validate_single_declared_model")
except ImportError:                      # gone from the wheelhouse: read the one model the agent YAML files declare
    def validate_single_declared_model(d):
        found = set()

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == "model" and isinstance(v, str):
                        found.add(v)
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        for f in Path(d).rglob("*.yaml"):
            try:
                walk(yaml.safe_load(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        print("declared models (own reader)", found, flush=True)
        return sorted(found)[0] if found else "gemma-4-31b-it-qat-w4a16-ct"

litellm.drop_params = True
DATA = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True), key=len)[0]).parent
MODEL = Path(sorted(glob.glob("/kaggle/input/models/**/gemma-4-31b-it-qat-w4a16-ct/*/config.json", recursive=True)
                    + glob.glob("/kaggle/input/**/gemma-4-31b-it-qat-w4a16-ct/**/config.json", recursive=True), key=len)[0]).parent
WORK = Path("/kaggle/working")
log("data", DATA, "model", MODEL)

dirs = {}
for name, files in CONFIGS.items():
    d = WORK / "agents" / name
    if d.exists():
        shutil.rmtree(d)
    if files == "@sample":
        shutil.copytree(DATA / "sample_submission", d)
    else:
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text, encoding="utf-8")
    dirs[name] = d

first = next(iter(dirs.values()))
declared = validate_single_declared_model(first)
adapters = {n: discover_adapters(str(d), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) for n, d in dirs.items()}
any_lora = any(bool(a) for a in adapters.values())
ngpu = torch.cuda.device_count()
server = VllmServer(VllmConfig(
    model=str(MODEL), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
    max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=any_lora, max_loras=8, max_lora_rank=128, tensor_parallel_size=4 if ngpu >= 4 else (2 if ngpu >= 2 else 1),
    startup_timeout=60 * 25), adapter_manifest=next((a for a in adapters.values() if a), {}))
server.start()
log("vLLM up", server.base_url, "gpus", ngpu, "lora", any_lora)
models = server.create_model_registry(aliases=[declared, "gemma-4-31b-it-qat-w4a16-ct"], model_prefix="openai/", api_key="EMPTY")

tasks = load_tasks(DATA / "tasks.jsonl")
k, m = (int(x) for x in SHARD.split("/"))
ONLY = [x for x in "__ONLY__".split(",") if x and x != "__ONLY__"]   # e.g. the healthy tasks from kernels/control
if ONLY:
    tasks = [t for t in tasks if t.instance_id in set(ONLY)]
tasks = [t for i, t in enumerate(tasks) if i % m == k][:LIMIT]
log("tasks", len(tasks), "configs", list(dirs), "workers", WORKERS)
limits, gen = build_submission_limits()


def evaluator_for(name, d):
    raw = yaml.safe_load((d / "eval_config.yaml").read_text(encoding="utf-8"))
    ev = raw.get("evaluation", raw)
    turns = ev.get("max_turns", ev.get("max_llm_calls"))
    return Evaluator(EvalConfig(
        tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots", results_dir=WORK / "results" / name,
        submission_dir=d, models=models, sandbox="subprocess",
        timeout_seconds=int(ev.get("timeout_seconds", 300)), max_time_minutes=float(ev.get("max_time_minutes", 60.0)),
        max_tool_calls=int(ev.get("max_tool_calls", 100)), max_turns=int(turns) if turns is not None else None,
        limits=limits, generation_constraints=gen, adapter_manifest=adapters[name],
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336,
                                                        event_retention_size=5),
        graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"), wheels_dir=DATA / "wheels", verbose=False))


summary = {}
for name, d in dirs.items():
    ev = evaluator_for(name, d)
    rows = []

    def one(arg):
        i, t = arg
        t1 = time.time()
        try:
            r = asyncio.run(ev.evaluate_task(task=t, task_index=i, total_tasks=len(tasks)))
            row = dict(cfg=name, id=t.instance_id, repo=t.repo, resolved=bool(r.resolved), exit=r.test_exit_code,
                       patch=len(r.agent_patch or ""), tools=r.tool_calls, secs=round(r.duration_seconds, 1))
        except Exception as e:  # an error counts unresolved
            row = dict(cfg=name, id=t.instance_id, repo=t.repo, resolved=False, error=repr(e)[:300],
                       secs=round(time.time() - t1, 1))
        print("RESULT " + json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, enumerate(tasks, start=1)))
    n = sum(r["resolved"] for r in rows)
    summary[name] = dict(resolved=n, tasks=len(rows), mean_secs=round(sum(r["secs"] for r in rows) / max(1, len(rows)), 1),
                         by_repo={rp: sum(r["resolved"] for r in rows if r["repo"] == rp) for rp in sorted({r["repo"] for r in rows})})
    log("CONFIG", name, json.dumps(summary[name]))
    json.dump(rows, open(WORK / f"rows_{name}.json", "w"))

print("SUMMARY " + json.dumps(summary), flush=True)
# compact per-task table at the very end (the Kaggle log API keeps only the tail): id resolved secs per config
for name in dirs:
    rows = json.load(open(WORK / f"rows_{name}.json"))
    print(f"ROWS {name} " + " ".join(f"{r['id']}:{int(r['resolved'])}:{int(r['secs'])}" for r in rows), flush=True)
server.stop() if hasattr(server, "stop") else None
log("done")

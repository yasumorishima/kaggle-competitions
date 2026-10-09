"""Gemma 4 agent: local paired evaluation on a Kaggle TPU v5e-8 (built by tpueval/build.py).

Same evaluation as kernels/localeval (official swegemma Evaluator, Phase 1 agent + Phase 2 pytest, subprocess
sandbox, configs compared task by task), but the model is served by vLLM's TPU build (vllm-tpu, from PyPI; the
competition's vLLM wheel is CUDA only) so it runs without the weekly GPU quota. Two Python 3.12 venvs made with
uv: /tmp/vt (vllm-tpu server) and /tmp/ev (the competition's adk / swegemma wheels + their deps). The script
bootstraps them and re-runs itself inside /tmp/ev. Model: the competition's w4a16 QAT checkpoint if the TPU build
loads it, else the bf16 QAT-unquantized checkpoint of the same model.
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
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
T0 = float(os.environ.get("G4_T0", time.time()))
CONFIGS = __CONFIGS__
LIMIT = int(os.environ.get("G4_LIMIT", "__LIMIT__"))
WORKERS = int(os.environ.get("G4_WORKERS", "__WORKERS__"))
SHARD = "__SHARD__"


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def sh(cmd):
    log("$", cmd[:200])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print("\n".join(out[-25:]), flush=True)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd[:200]}")


if not os.environ.get("G4_INNER"):
    sh("nproc; free -g | head -2; python3 --version")
    sh(f"{sys.executable} -m pip install -q uv")
    UV = f"{sys.executable} -m uv"
    WH = sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent-wheelhouse", recursive=True), key=len)[0]
    sh(f"{UV} python install 3.12")
    sh(f"{UV} venv -q -p 3.12 /tmp/vt && {UV} pip install -q -p /tmp/vt/bin/python vllm-tpu")
    sh("/tmp/vt/bin/python -c 'import vllm, jax; print(\"vllm\", vllm.__version__, \"jax\", jax.__version__, jax.devices()[:1])'")
    wheels = " ".join(f"{WH}/{w}" for w in ("adk_eval_core-0.1.0-py3-none-any.whl", "adk_submission-0.2.12-py3-none-any.whl",
                                            "google_adk-1.36.1-py3-none-any.whl", "google_genai-2.11.0-py3-none-any.whl",
                                            "swegemma-0.2.7-py3-none-any.whl", "anthropic-1.4.0-py3-none-any.whl"))
    sh(f"{UV} venv -q -p 3.12 /tmp/ev && {UV} pip install -q -p /tmp/ev/bin/python {wheels} litellm pyyaml")
    env = dict(os.environ, G4_INNER="1", G4_T0=str(T0))
    sys.exit(subprocess.run(["/tmp/ev/bin/python", os.path.abspath(__file__)], env=env).returncode)

os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_NO_USAGE_STATS": "1",
                   "OTEL_SDK_DISABLED": "true", "VLLM_ENGINE_READY_TIMEOUT_S": "3000"})
import litellm  # noqa: E402
import yaml  # noqa: E402
from adk_submission import VllmConfig, VllmServer, discover_adapters  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.models.discovery import validate_single_declared_model  # noqa: E402

litellm.drop_params = True
DATA = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True), key=len)[0]).parent
def model_dir(tag):
    c = sorted(glob.glob(f"/kaggle/input/**/{tag}/**/config.json", recursive=True), key=len)
    return Path(c[0]).parent if c else None


MODELS = [m for m in (model_dir("gemma-4-31b-it-qat-w4a16-ct"), model_dir("gemma-4-31b-it-qat-q4_0-unquantized")) if m]
WORK = Path("/kaggle/working")
log("data", DATA, "models", MODELS)

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


class TpuVllmServer(VllmServer):
    """the adk_submission vLLM server, launched with the vllm-tpu venv's interpreter"""
    def build_cmd(self):
        cmd = super().build_cmd()
        cmd[0] = "/tmp/vt/bin/python"
        return cmd


sh("/tmp/vt/bin/python -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -iA3 -- '--tool-call-parser\\|--reasoning-parser' | head -30 || true")
server = None
for MODEL in MODELS:
    try:
        server = TpuVllmServer(VllmConfig(
            model=str(MODEL), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
            max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
            enable_lora=any_lora, max_loras=8, max_lora_rank=128, tensor_parallel_size=8,
            startup_timeout=60 * 50), adapter_manifest=next((a for a in adapters.values() if a), {}))
        server.start()
        break
    except Exception as e:
        log("server failed for", MODEL, repr(e)[-3000:])
        try:                                           # the reason lives in the server's own log, not in the exception
            import glob as _g
            logs = [str(getattr(server, a)) for a in ("log_path", "log_file", "_log_path", "_log_file")
                    if server is not None and getattr(server, a, None)]
            logs += sorted(_g.glob("/tmp/**/*vllm*.log", recursive=True) + _g.glob("/kaggle/working/**/*vllm*.log", recursive=True))
            for lp in dict.fromkeys(logs):
                log("server log", lp, "\n" + open(lp, errors="replace").read()[-6000:])
        except Exception as e2:
            log("server log unreadable", repr(e2))
        try:
            server.stop()
        except Exception:
            pass
        server = None
if server is None:
    raise SystemExit("no model could be served")
log("vLLM up", server.base_url, "model", MODEL, "lora", any_lora)
models = server.create_model_registry(aliases=[declared, "gemma-4-31b-it-qat-w4a16-ct"], model_prefix="openai/", api_key="EMPTY")

tasks = load_tasks(DATA / "tasks.jsonl")
k, m = (int(x) for x in SHARD.split("/"))
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

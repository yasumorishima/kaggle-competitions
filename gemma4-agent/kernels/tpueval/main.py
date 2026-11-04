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
CONFIGS = {'v21': {'eval_config.yaml': 'evaluation:\n  timeout_seconds: 240\n  max_tool_calls: 28\n  max_time_minutes: 8\n  max_turns: 80\n', 'agent.yaml': 'name: swe_agent\nmodel: gemma-4-31b-it-qat-w4a16-ct\ninstruction: !include prompts/system.md\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\n  - get_code_neighbors\n  - search_similar_code\n  - get_code_subgraph\ngenerate_content_config: !include configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n', 'prompts/system.md': 'You are an autonomous software engineer. Your job: make the smallest correct change to the SOURCE code in /workspace so that the hidden tests for the issue below pass. The hidden tests were written from the issue text, so the exact names, messages, types and behaviours in the issue are the specification.\n\n# HARD LIMITS\n- 28 tool calls and 8 minutes in total. Call `get_status` (free) after your edit: with under 120 seconds left, skip verification and go straight to the final check and `submit_patch`. When either runs out, the working tree is graded as it is. An edit that exists counts, even if you never call `submit_patch`. An unedited tree always scores zero.\n- Never create, modify or delete test files (`test_*.py`, `*_test.py`, anything under `tests/`, `pytest.ini`, `conftest.py`). This is an automatic failure.\n- Scratch files go in /tmp only. Any file left under /workspace becomes part of the graded patch.\n- Never run a full test suite (bare `pytest`, `pytest .`, `unittest discover`). Always name one test file, and add `-k` when you can.\n- Never look outside /workspace. All dependencies are already installed.\n- Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. They return nothing useful.\n\n# CALL PLAN\n- Calls 1-8: locate and read. Calls 9-12: edit. You MUST have edited a source file by call 12. If you are unsure, make your best-guess edit in the most likely place now. A wrong edit can still be corrected. Pure reading cannot score.\n- Calls 13-20: verify and correct.\n- Call 20 onward: no new exploration. Check, fix, submit.\n- Always finish with `git status --short && git diff` followed by `submit_patch`.\n\n# PROCEDURE\n\n## 1. Extract the specification (no tool call)\nFrom the issue, list for yourself: the exact file, function, class or CLI name; the exact error message and exception type; the expected return value or output; the inputs that trigger the bug; every edge case mentioned (None, empty, negative, unicode, duplicates, defaults). Treat every concrete string and name in the issue as a literal that must appear exactly in your code.\n\n## 2. Locate\n- Use exact paths from the issue directly. When searches are necessary, combine independent searches for the issue\'s exact symbols, their callers and relevant test paths into one run_command. Use git grep with fixed-string patterns for literal identifiers and git ls-files for tracked file paths; each result should show its file path and line number where applicable.\n- Batch only searches whose inputs are already known. If the next search depends on the first result, inspect that result before searching again. Limit each search to about 20 matching lines and keep total output short; never dump whole files or the repository tree.\n- Once the target is located, read the relevant implementation plus one relevant caller or test case. When both paths and line ranges are known, read these small regions in one run_command using sed with explicit line ranges, or use read_file for a single region. Aim for about 40 lines per region and at most three regions per command. Include enough surrounding code to understand signatures, defaults and error handling.\n- For a bug: find where the wrong behaviour is produced and use the caller or test to understand the contract before editing.\n- For a feature: find the closest existing sibling (a similar function, option, class, or doc example) and copy its pattern, including registration points such as __all__, option lists, dispatch tables and imports.\n- Reuse paths and test names already discovered. Do not spend another tool call repeating a search that already answered the question.\n- If the issue includes a code snippet that reproduces the problem, you may run it once with python -c (one call, never create a file) to see the real error.\n\n## 3. Edit\n- Use edit_file for changes to existing code. Copy old_string exactly from the source, including whitespace, and choose a short unique match. Use write_file only for a genuinely new source file that the issue requires.\n- If edit_file reports no match or multiple matches, read the exact source region and retry with a smaller unique snippet. Never repeat an identical failed edit. After two matching failures, use one inline Python script through run_command with a quoted heredoc; assert that the old text occurs exactly once before replacing it and writing the file. Preserve unrelated content and do not rewrite the file from memory.\n- Fix the cause, not the symptom. Do not special-case the example from the issue.\n- The grader runs the whole test file, including tests that pass today. Keep the current behaviour for every input the issue does not mention.\n- Change only what the issue requires. Keep existing signatures, defaults, and backward compatibility unless the issue says otherwise. Match the style, naming and error-handling idioms of the surrounding code.\n- If the same behaviour has parallel implementations (sync and async variants, Client and AsyncClient, several backends or transports, typing overloads, re-exports in `__init__.py` or `__all__`), apply the change to every one of them. Hidden tests often exercise more than one.\n- Use the exact exception class and exact message text from the issue. If the issue only describes the error, reuse the wording style of neighbouring error messages.\n- Handle every edge case you listed in step 1.\n- For documentation-code tasks (for example FastAPI), edit the executable code under `docs_src/`.\n- Sibling sites (from 129 reference fixes of these repositories; FastAPI fixes change 2.5 files on average). After your main edit, check the listed partners of the file you edited with one `git grep -n` for the name you changed, and apply the same change where it is used:\n  - fastapi: `routing.py` (APIRoute, APIRouter.add_api_route and every method decorator) with `applications.py` (the FastAPI methods that forward to the router) and `openapi/utils.py`; `params.py` with `param_functions.py`, `temp_pydantic_v1_params.py` and `dependencies/utils.py`; `dependencies/models.py` with `dependencies/utils.py`; `_compat/v1.py` with `_compat/v2.py` and `_compat/__init__.py`; a new error type in `exceptions.py` is raised in `routing.py`. A `docs_src/<topic>/` change goes into every variant file of that tutorial (`_an`, `_py39`, `_py310`).\n  - rich: a new style name goes into `default_styles.py`; `cells.py` with `segment.py` and `_wrap.py`; `console.py` with `segment.py`.\n  - requests: `utils.py` with `compat.py`; `models.py` with `_types.py`.\n  - httpx: `src/httpx/` and `src/ahttpx/` are sync and async twins; change both.\n\n## 4. Verify (at most 2 calls)\n- If the issue contains example code or expected output, first rerun that example with python -c and compare the result with what the issue expects: exact text, type, exception and status code. This is the closest check to the hidden test.\n- If calls and time remain, find the test file with `find tests -name "*keyword*.py"` and run `timeout 90 python -m pytest tests/path/test_x.py -x -q -k keyword`. Never run a whole test file. Use `python3 -m unittest tests.test_x` for unittest projects.\n- Failures that come from missing fixtures, missing data, or import errors unrelated to your change are pre-existing. Ignore them. Never repair tests or the environment.\n- If a test related to your change fails, read the traceback, fix the source once, and rerun once. If no test can run, do one `python -c` check or `python -m py_compile` on the edited file.\n\n## 5. Final check and submit\n- Run `git status --short && git diff` in one call. `git diff` alone hides untracked files, but `submit_patch` includes them. If you see an untracked scratch file or a changed test file, remove it or restore it with `git checkout --` before submitting. Confirm that: only source files changed; every name, message and type from step 1 appears exactly; nothing unrelated was touched; there are no leftover debug prints or stray files.\n- Call `submit_patch`. Confirm `patch_size > 0` and `files_changed > 0`. If either is 0, you have not edited anything. Edit now and submit again.\n- End with a single-sentence summary of the fix.\n\n# DECISION RULES\n- Ambiguous issue: choose the most literal reading of the issue text and act on it. Do not ask questions and do not stall.\n- Several possible locations: pick the one where the wrong behaviour is produced, not where it is merely observed.\n- Search returned nothing: try one different specific string, then edit your best candidate anyway.\n- Stuck after a failed fix: make a different, smaller change. Do not repeat the same edit.\n- Never conclude that nothing needs to change. Every task requires a non-empty source patch.\n'}, 'v28': {'eval_config.yaml': 'evaluation:\n  timeout_seconds: 150\n  max_tool_calls: 28\n  max_time_minutes: 7\n  max_turns: 80\n', 'agent.yaml': 'name: swe_agent\nmodel: gemma-4-31b-it-qat-w4a16-ct\ninstruction: !include prompts/system.md\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\ngenerate_content_config: !include configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 1.0\ntop_p: 0.95\ntop_k: 64\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n', 'prompts/system.md': 'You are an expert autonomous software engineer. Resolve the user issue with the smallest source change that satisfies the stated behavior.\n\nThe hidden grader applies your git diff to a fresh repository and runs its own tests. It never sees your reasoning. An empty diff scores zero. A correct source diff scores the task even if your local test was red or you never ran a test.\n\n## Budget\nYou have 28 tool calls and 7 minutes. When either limit hits, the working tree is graded as it is. An edit made in time counts even if you do not reach submit_patch. Finish as soon as the edit is in so later tasks still have time.\nTime guard: call `get_status` (free) right after your first edit. If it shows less than 150 seconds remaining, skip the check and call `submit_patch` now. Any command is killed after 150 seconds, so wrap a test run in `timeout 90`.\n\n## 1. Contract, before any tool call\nRead the issue and keep a short list of the exact symbols, signatures, error strings, HTTP status codes, and return values it requires. Copy them character for character. The grader checks those strings exactly.\n\n## 2. Locate in a few reads\n- Search once with `git grep -n "SPECIFIC_TOKEN" -- \'*.py\' | head -20`. Use the most specific identifier or error string from the issue.\n- Open only the matching region with `read_file`, at most 120 lines (`start_line` and `end_line`).\n- You may do one follow-up grep or read if the first hit only re-exports the symbol.\n- Do not page through unrelated files. Do not read a whole file when a range will do.\n\n## 3. Edit by tool call 12\n- By your 12th tool call a source file must already be edited. If it is not, edit the most likely function immediately. Runs that are still only reading after 12 calls almost never succeed.\n- Prefer `edit_file`. `old_string` must match the file exactly, including indentation. Change the smallest span that implements the contract.\n- FastAPI documentation tasks: edit the executable example under `docs_src/`.\n- Do not reformat or refactor unrelated code.\n- Create a new file only when the issue explicitly requires a new module. Never create or edit `test_*.py`, `*_test.py`, anything under `tests/`, `pytest.ini`, or `conftest.py`.\n- Any scratch script goes in `/tmp` through `run_command`. Never write scratch files into `/workspace`. Files left in the repository become part of the diff and fail grading.\n\n## 4. One check, one repair, then submit\n- After the edit, run at most one focused check: `timeout 90 python -m pytest tests/test_file.py -k test_name -q` or `python -c "..."`.\n- Never run bare `pytest`, `pytest .`, or `python -m unittest discover`.\n- If that check fails with an assertion about the behavior you changed, apply one `edit_file` using the assertion text, then stop testing.\n- If it fails because of missing fixtures, import errors, or pre-existing breakage, ignore it. Do not repair tests and do not install packages.\n- Call `submit_patch` after that, even when the local check is red or you skipped it. A local green test is not the grader.\n- If the diff contains a test path, run `git checkout --` on that path and call `submit_patch` again.\n- If `get_status` shows less than 150 seconds remaining, call `submit_patch` immediately.\n\n## Never\n- Conclude that the code is already correct. Every task needs a source modification and a non-empty patch.\n- Search outside `/workspace` (`/usr`, `/wheels`, `/opt`). Dependencies are already installed.\n- Spend calls on broad exploration after the first edit.\n'}}
LIMIT = int(os.environ.get("G4_LIMIT", "1000"))
WORKERS = int(os.environ.get("G4_WORKERS", "1"))
SHARD = "0/1"


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
    # by package name, not version: the wheelhouse is re-versioned (adk_submission 0.2.12 -> 0.2.13 on 10-09)
    wheels = " ".join(sorted(glob.glob(f"{WH}/{p}-*.whl"))[-1] for p in (
        "adk_eval_core", "adk_submission", "google_adk", "google_genai", "swegemma", "anthropic"))
    log("eval wheels", wheels)
    sh(f"{UV} venv -q -p 3.12 /tmp/ev && {UV} pip install -q -p /tmp/ev/bin/python {wheels} litellm pyyaml")
    # Phase 2 runs pytest in a venv that only sees this process's site-packages (swegemma SubprocessSandbox writes
    # sys.path into _host_env.pth; install_test_dependencies is a no-op for subprocess). The GPU image has pytest in
    # its system python; /tmp/ev does not (tpueval-6: every gold patch failed with "No module named pytest").
    try:
        sh(f"{UV} pip install -q -p /tmp/ev/bin/python pytest==8.3.4 inline-snapshot dirty-equals anyio trio "
           "sqlalchemy sqlmodel python-multipart email-validator orjson ujson pyjwt passlib flask attrs")
    except SystemExit as e:
        log("test deps (full) failed", e)
        sh(f"{UV} pip install -q -p /tmp/ev/bin/python pytest==8.3.4 inline-snapshot dirty-equals")
    env = dict(os.environ, G4_INNER="1", G4_T0=str(T0))
    sys.exit(subprocess.run(["/tmp/ev/bin/python", os.path.abspath(__file__)], env=env).returncode)

os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_NO_USAGE_STATS": "1",
                   "OTEL_SDK_DISABLED": "true", "VLLM_ENGINE_READY_TIMEOUT_S": "3000"})
import litellm  # noqa: E402
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

# diagnostics (ours, no behaviour change): print the argument names that arrived when ADK rejects a tool call for a
# missing mandatory argument (localeval-8: old_string is cut at its first colon by the tool-call parse)
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
def model_dir(tag):
    c = sorted(glob.glob(f"/kaggle/input/**/{tag}/**/config.json", recursive=True), key=len)
    return Path(c[0]).parent if c else None


# bf16 only by default: the w4a16 checkpoint ran out of HBM on v5e-8 (tpueval-5: 16.30G of 15.75G, 18 min lost)
MODELS = [m for m in ((model_dir("gemma-4-31b-it-qat-w4a16-ct"),) if os.environ.get("G4_TRY_W4") == "1" else ())
          + (model_dir("gemma-4-31b-it-qat-q4_0-unquantized"),) if m]
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
ONLY = [x for x in "fastapi_15589,fastapi_14986,fastapi_14873,fastapi_14463,fastapi_14487,fastapi_9555,fastapi_14419,rich_3944,fastapi_14616,rich_3180,rich_3772,rich_3052,rich_2943,rich_3067".split(",") if x and x != "fastapi_15589,fastapi_14986,fastapi_14873,fastapi_14463,fastapi_14487,fastapi_9555,fastapi_14419,rich_3944,fastapi_14616,rich_3180,rich_3772,rich_3052,rich_2943,rich_3067"]   # e.g. the healthy tasks from kernels/control
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


# does Phase 2 work on the TPU machine? the reference patch must pass and the empty patch must fail (tpueval-5: v7 and
# v8 both 0/6 on tasks that are healthy on the GPU/CPU machines, so the grader here is checked first)
try:
    import swegemma.harness.verification as V
    resolve_task_snapshot_paths = _find_attr("resolve_task_snapshot_paths")
    gev = evaluator_for(next(iter(dirs)), first)
    health = []
    n_gold = 0
    for t in tasks:
        row = []
        for tag, patch in (("noop", ""), ("gold", t.patch)):
            try:
                tt = gev._hydrate_task_from_secret(t)
                snap, base, pp = resolve_task_snapshot_paths(gev.config.snapshots_dir, tt.instance_id, tt.repo)
                r = asyncio.run(V.verify_task(gev.docker, gev.config, tt, snap, base_snapshot_path=base, patch_path=pp,
                                              agent_patch=patch, start_time=time.perf_counter()))
                row.append(int(bool(r.resolved)))
                if tag == "gold" and not r.resolved:
                    log("gold fails", t.instance_id, getattr(r, "test_exit_code", None),
                        (getattr(r, "test_output", "") or "")[-800:])
            except Exception as e:
                row.append(-1)
                log("verify error", t.instance_id, tag, repr(e)[:400])
        health.append(f"{t.instance_id}:{row[0]}{row[1]}")
        n_gold += row[1] == 1
    log("HEALTH", " ".join(health), "gold_ok", n_gold)
except Exception as e:
    log("gold check failed", repr(e)[:600])
    n_gold = -1
if n_gold == 0:
    # the grader cannot resolve anything here: stop before spending hours of TPU on agent runs that cannot score
    raise SystemExit("no gold patch resolves on this machine; agent runs skipped")

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

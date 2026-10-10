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
CONFIGS = {'v7': {'eval_config.yaml': 'evaluation:\n  timeout_seconds: 120\n  max_tool_calls: 80\n  max_time_minutes: 5\n  max_turns: 120\n', 'agent.yaml': 'name: fixer\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Fixes a bug or adds a small feature in a Python repository and submits the patch.\ninstruction: !include prompts/system.md\ngenerate_content_config: !include configs/sampling.yaml\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - search_similar_code\n  - get_code_neighbors\n  - get_code_subgraph\n  - get_status\n  - submit_patch\n  - agent_tool:\n      config_path: sub_agents/locator.yaml\n      skip_summarization: false\n  - agent_tool:\n      config_path: sub_agents/checker.yaml\n      skip_summarization: false\n', 'sub_agents/locator.yaml': 'name: locator\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only code locator. Call it once with a short note on what to look for; it reads the repository and returns the file, function, line numbers and the change to make.\ninstruction: !include ../prompts/locator.md\ntools:\n  - run_command\n  - read_file\n  - get_code_neighbors\ngenerate_content_config: !include ../configs/sampling.yaml\n', 'sub_agents/checker.yaml': 'name: checker\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only change checker. Call it once after editing, with a one-line note of what the change should now do; it runs a scratch check and the nearby existing tests in its own context and returns VERDICT, EVIDENCE and FIX.\ninstruction: !include ../prompts/checker.md\ntools:\n  - run_command\n  - read_file\ngenerate_content_config: !include ../configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 0\n  include_thoughts: false\n', 'prompts/system.md': 'You are a careful Python maintainer. The repository is checked out at /workspace. Your job is to make the\nsmallest correct source change that resolves the issue below, so that the project\'s own (hidden) tests for\nthis issue pass. The issue is in the first user message. You have a hard budget of about 80 tool calls and\n5 minutes, so work in a straight line.\n\nWork in this order:\n\n1. Locate (one call). First call the `locator` tool once with a one-line note of what to find (for example\n   "where the `timeout` option of `Client.send` is applied"). It reads the code in a separate context and returns\n   FILES, CAUSE, CHANGE, CODE and CHECK. Then confirm its answer with one read_file\n   of the named lines. Only if its answer is empty or clearly wrong, locate yourself with `git grep -n "name" -- \'*.py\' | head -30`\n   (there is no `rg`) and `read_file` of focused ranges; always cut long output with `| head`.\n2. Understand. Write down in a few sentences (as plain text, not only in your head) the file, function and\n   line numbers to change and what the expected behaviour is. Older tool outputs are dropped from the\n   conversation when it gets long; your own notes are kept.\n   If the issue shows a snippet, it usually describes the intended behaviour exactly; follow it.\n3. Edit. Change library code only (never tests, never /workspace/pytest.ini or /workspace/conftest.py).\n   Use `edit_file` with a short, unique `old_string` copied exactly from `read_file` output; make several small\n   edits rather than one large one. Keep the existing style, names and public signatures; when the issue asks\n   for a new parameter or option, add it with a backward-compatible default.\n4. Check (one call). Call the `checker` tool once with a one-line note of what the code should now do\n   (for example "`Client.send(timeout=None)` no longer raises"). In its own context it runs a scratch\n   reproduction and the nearest existing tests and returns VERDICT, EVIDENCE and FIX. If VERDICT is FAIL\n   because of your change, apply its FIX with `edit_file` and call `checker` once more at most. If the check\n   itself was broken, do not spend more calls on it. Only if `checker` errors, check yourself: write a scratch\n   script with `run_command` and a heredoc outside /workspace (`cat > /tmp/check.py <<\'EOF\' ... EOF`, then\n   `python /tmp/check.py 2>&1 | tail -20`) or run `python -m pytest -x -q tests/test_x.py -k name 2>&1 | tail -25`.\n   Command output is cut after its first 5,000 characters, so always end long commands with `| tail`.\n5. Submit. Run `git status --short` and `git diff` to confirm only intended source files changed, then call\n   `submit_patch` as your final action. Always submit before the budget runs out: a reasonable patch scores,\n   no patch never does. Use `get_status` (free) if you are unsure how much budget is left.\n\nTool-call rules (most failed calls so far broke these):\n- Give every argument in its own field. For read_file: path is only the file path, such as fastapi/routing.py,\n  with no backticks, quotes or line numbers inside it; start_line and end_line are separate numbers, with\n  end_line >= start_line and at most 150 lines apart.\n- If a call returns "Source path ... not found", your path carried extra characters: retry once with the bare path.\n- edit_file needs path, old_string (text that exists in the file now, copied exactly) and new_string. To create a\n  new file, use write_file instead.\n- Never repeat a call that just failed with the same arguments; change the arguments or the approach.\n\nDo not call `search_similar_code`: it returns whole function bodies with no length limit and can overflow\nthe context, which loses the task. If `get_code_neighbors` or `get_code_subgraph` errors, stop using them\n(they cannot see async functions) and use `git grep` and `read_file` instead.\n\nRules: no pip installs (the environment is offline and complete), no network, no edits outside /workspace\nsource files, no rewriting of unrelated code, no new test files in /workspace.\n\nThe issue you are solving (repeated here because the first message may be summarized away):\n\n{problem_description}\n', 'prompts/checker.md': 'You are a read-only checker for the Python repository at /workspace. Another agent has just edited the\nsource to resolve the issue below. You never edit files in /workspace; you only run checks and report.\nYour answer is all the other agent sees of your work, so make it short and exact.\n\nThe issue:\n\n{problem_description}\n\nHow to work (at most about 8 tool calls):\n1. `run_command`: `cd /workspace && git diff | head -80` to see the change.\n2. Write a scratch script that reproduces the issue\'s example (or the behaviour it asks for) with a\n   heredoc outside /workspace, e.g. `cat > /tmp/chk.py <<\'PYEOF\' ... PYEOF`, then\n   `cd /workspace && python /tmp/chk.py 2>&1 | tail -20`.\n3. Find the existing tests closest to the changed function, e.g.\n   `git grep -ln "function_name" -- \'test*\' \'*/test*\' | head -5`, and run only those:\n   `python -m pytest -x -q <file> -k <name> 2>&1 | tail -25`. Output is cut after its first 5,000\n   characters, so always end with `| tail`.\n4. Do not install anything and do not run the whole test suite.\n\nAnswer in exactly this format, and nothing else:\n\nVERDICT: PASS or FAIL\nEVIDENCE: <the 3-10 most telling output lines: the scratch result and the pytest summary line>\nFIX: <if FAIL because of the change: file, line and what to change; if the check itself was broken or no\nrelevant test exists, say so in one line; if PASS: none>\n', 'prompts/locator.md': 'You are a read-only code locator for the Python repository at /workspace. You never edit files. Another\nagent will make the change; your answer is all it sees of your work, so make it precise and short.\n\nThe issue:\n\n{problem_description}\n\nHow to work (at most about 12 tool calls):\n1. Pull exact identifiers out of the issue (function, class, method, option, error text, file name) and find\n   them with `run_command`, e.g. `git grep -n "name" -- \'*.py\' | grep -v test | head -30` (there is no `rg`).\n   Always cut long output with `| head` or `| tail`; output is cut after its first 5,000 characters.\n2. Read only the relevant regions with read_file (at most 150 lines per call).\n   Follow the call path from the public entry point named in the issue to the code that behaves wrongly.\n   Use `get_code_neighbors` for callers or callees if `git grep` is not enough.\n3. Do not call `search_similar_code` (it can overflow the context).\n\nTool-call rules (most failed calls so far broke these):\n- Give every argument in its own field. For read_file: path is only the file path, such as fastapi/routing.py,\n  with no backticks, quotes or line numbers inside it; start_line and end_line are separate numbers, with\n  end_line >= start_line and at most 150 lines apart.\n- If a call returns "Source path ... not found", your path carried extra characters: retry once with the bare path.\n- edit_file needs path, old_string (text that exists in the file now, copied exactly) and new_string. To create a\n  new file, use write_file instead.\n- Never repeat a call that just failed with the same arguments; change the arguments or the approach.\n\n\nAnswer in exactly this format, and nothing else:\n\nFILES: <path>:<start>-<end> (<function or class>) [one line per place to change, most important first]\nCAUSE: <one or two sentences: what the code does now and why that is wrong for the issue>\nCHANGE: <concrete edit plan: which lines, what to add or replace, new parameter names and defaults>\nCODE: <the 5-15 current lines that must change, copied exactly from read_file without line numbers>\nCHECK: <one short python snippet or pytest -k command that shows the fix works>\n'}, 'v8': {'eval_config.yaml': 'evaluation:\n  timeout_seconds: 120\n  max_tool_calls: 80\n  max_time_minutes: 5\n  max_turns: 120\n', 'agent.yaml': 'name: fixer\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Fixes a bug or adds a small feature in a Python repository and submits the patch.\ninstruction: !include prompts/system.md\ngenerate_content_config: !include configs/sampling.yaml\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - search_similar_code\n  - get_code_neighbors\n  - get_code_subgraph\n  - get_status\n  - submit_patch\n  - agent_tool:\n      config_path: sub_agents/checker.yaml\n      skip_summarization: false\n', 'sub_agents/checker.yaml': 'name: checker\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only change checker. Call it once after editing, with a one-line note of what the change should now do; it runs a scratch check and the nearby existing tests in its own context and returns VERDICT, EVIDENCE and FIX.\ninstruction: !include ../prompts/checker.md\ntools:\n  - run_command\n  - read_file\ngenerate_content_config: !include ../configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 0\n  include_thoughts: false\n', 'prompts/system.md': 'You are a careful Python maintainer. The repository is checked out at /workspace. Your job is to make the\nsmallest correct source change that resolves the issue below, so that the project\'s own (hidden) tests for\nthis issue pass. The issue is in the first user message. You have a hard budget of about 80 tool calls and\n5 minutes, so work in a straight line.\n\nWork in this order. The clock matters more than anything else: a task where you never edit scores nothing,\nwhile an edited workspace is scored even if time runs out (the harness takes `git diff` at the end).\n\n1. Locate yourself, in at most 4 calls. Pick the most specific name in the issue (a function, option,\n   class, error message or parameter) and run `git grep -n "name" -- \'*.py\' | grep -v test | head -20`\n   (there is no `rg`). Then `read_file` the 40-100 lines around the best hit. A second grep or read only if\n   the first was clearly the wrong place.\n2. Write your plan as plain text in 2-3 sentences: file, function, line numbers, and the behaviour the issue\n   asks for. Older tool outputs are dropped from the conversation when it gets long; your own notes are kept.\n   If the issue shows a snippet, it usually describes the intended behaviour exactly; follow it.\n3. Edit now (by call 7 at the latest). Change library code only (never tests, never /workspace/pytest.ini or\n   /workspace/conftest.py). Use `edit_file` with a short, unique `old_string` copied exactly from `read_file`\n   output; make several small edits rather than one large one. Keep the existing style, names and public\n   signatures; when the issue asks for a new parameter or option, add it with a backward-compatible default.\n   Make your best edit even when unsure: an imperfect edit can still pass, no edit never does.\n4. Call `get_status` (free). If `time_seconds_remaining` is above 120, call the `checker` tool once with a\n   one-line note of what the code should now do (for example "`Client.send(timeout=None)` no longer raises").\n   It runs a scratch reproduction and the nearest existing tests in its own context and returns VERDICT,\n   EVIDENCE and FIX. If VERDICT is FAIL because of your change, apply its FIX with `edit_file`. Do not call\n   `checker` a second time. If 120 seconds or less remain, skip the check.\n5. Submit. Run `git status --short` to confirm only intended source files changed (remove any scratch file\n   you created inside /workspace), then call `submit_patch` as your final action.\n\nIf you only have the edit half done when `get_status` shows under 60 seconds left, submit what you have.\n\nTool-call rules (most failed calls so far broke these):\n- Give every argument in its own field. For read_file: path is only the file path, such as fastapi/routing.py,\n  with no backticks, quotes or line numbers inside it; start_line and end_line are separate numbers, with\n  end_line >= start_line and at most 150 lines apart.\n- If a call returns "Source path ... not found", your path carried extra characters: retry once with the bare path.\n- edit_file needs path, old_string (text that exists in the file now, copied exactly) and new_string. To create a\n  new file, use write_file instead.\n- Never repeat a call that just failed with the same arguments; change the arguments or the approach.\n\nDo not call `search_similar_code`: it returns whole function bodies with no length limit and can overflow\nthe context, which loses the task. If `get_code_neighbors` or `get_code_subgraph` errors, stop using them\n(they cannot see async functions) and use `git grep` and `read_file` instead.\n\nRules: no pip installs (the environment is offline and complete), no network, no edits outside /workspace\nsource files, no rewriting of unrelated code, no new test files in /workspace.\n\nThe issue you are solving (repeated here because the first message may be summarized away):\n\n{problem_description}\n', 'prompts/checker.md': 'You are a read-only checker for the Python repository at /workspace. Another agent has just edited the\nsource to resolve the issue below. You never edit files in /workspace; you only run checks and report.\nYour answer is all the other agent sees of your work, so make it short and exact.\n\nThe issue:\n\n{problem_description}\n\nHow to work (at most about 8 tool calls):\n1. `run_command`: `cd /workspace && git diff | head -80` to see the change.\n2. Write a scratch script that reproduces the issue\'s example (or the behaviour it asks for) with a\n   heredoc outside /workspace, e.g. `cat > /tmp/chk.py <<\'PYEOF\' ... PYEOF`, then\n   `cd /workspace && python /tmp/chk.py 2>&1 | tail -20`.\n3. Find the existing tests closest to the changed function, e.g.\n   `git grep -ln "function_name" -- \'test*\' \'*/test*\' | head -5`, and run only those:\n   `python -m pytest -x -q <file> -k <name> 2>&1 | tail -25`. Output is cut after its first 5,000\n   characters, so always end with `| tail`.\n4. Do not install anything and do not run the whole test suite.\n\nAnswer in exactly this format, and nothing else:\n\nVERDICT: PASS or FAIL\nEVIDENCE: <the 3-10 most telling output lines: the scratch result and the pytest summary line>\nFIX: <if FAIL because of the change: file, line and what to change; if the check itself was broken or no\nrelevant test exists, say so in one line; if PASS: none>\n'}}          # name -> {relative path: text}; "@sample" = the official sample_submission
LIMIT = int(os.environ.get("G4_LIMIT", "1000"))
WORKERS = int(os.environ.get("G4_WORKERS", "1"))
SHARD = "0/2"            # "k/M": tasks with index % M == k


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
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.models.discovery import validate_single_declared_model  # noqa: E402

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
ONLY = [x for x in "fastapi_15661,fastapi_15588,fastapi_15589,fastapi_15280,fastapi_14986,rich_4070,rich_4077,rich_4079,rich_4076,rich_4075,rich_4006,fastapi_14258,fastapi_14873,fastapi_13920,fastapi_14794,fastapi_14786,fastapi_14492,fastapi_14479,fastapi_14463,fastapi_14372,fastapi_14487,fastapi_14306,fastapi_14458,fastapi_14448,fastapi_9753,fastapi_5077,fastapi_9555,fastapi_5624,fastapi_14419,fastapi_14301,fastapi_9425,rich_3718,rich_3934,rich_3953,rich_3944,rich_3942,rich_3938,rich_3882,rich_3894,rich_3935,rich_3905,rich_3930,fastapi_15800,fastapi_13537,fastapi_14616,fastapi_11194,rich_3180,rich_3777,rich_3468,rich_3782,rich_3772,rich_3675,rich_3676,rich_3518,rich_3535,rich_3521,rich_3506,rich_3480,rich_3472,rich_3454,rich_3471,rich_3470,rich_3052,rich_3278,rich_2943,rich_2725,rich_3105,rich_3043,rich_3067,rich_3006,rich_3063,rich_3061".split(",") if x and x != "fastapi_15661,fastapi_15588,fastapi_15589,fastapi_15280,fastapi_14986,rich_4070,rich_4077,rich_4079,rich_4076,rich_4075,rich_4006,fastapi_14258,fastapi_14873,fastapi_13920,fastapi_14794,fastapi_14786,fastapi_14492,fastapi_14479,fastapi_14463,fastapi_14372,fastapi_14487,fastapi_14306,fastapi_14458,fastapi_14448,fastapi_9753,fastapi_5077,fastapi_9555,fastapi_5624,fastapi_14419,fastapi_14301,fastapi_9425,rich_3718,rich_3934,rich_3953,rich_3944,rich_3942,rich_3938,rich_3882,rich_3894,rich_3935,rich_3905,rich_3930,fastapi_15800,fastapi_13537,fastapi_14616,fastapi_11194,rich_3180,rich_3777,rich_3468,rich_3782,rich_3772,rich_3675,rich_3676,rich_3518,rich_3535,rich_3521,rich_3506,rich_3480,rich_3472,rich_3454,rich_3471,rich_3470,rich_3052,rich_3278,rich_2943,rich_2725,rich_3105,rich_3043,rich_3067,rich_3006,rich_3063,rich_3061"]   # e.g. the healthy tasks from kernels/control
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

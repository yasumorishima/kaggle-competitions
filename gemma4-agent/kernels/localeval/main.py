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
CONFIGS = {'v6': {'eval_config.yaml': 'evaluation:\n  timeout_seconds: 120\n  max_tool_calls: 80\n  max_time_minutes: 5\n  max_turns: 120\n', 'agent.yaml': 'name: fixer\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Fixes a bug or adds a small feature in a Python repository and submits the patch.\ninstruction: !include prompts/system.md\ngenerate_content_config: !include configs/sampling.yaml\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - search_similar_code\n  - get_code_neighbors\n  - get_code_subgraph\n  - get_status\n  - submit_patch\n  - agent_tool:\n      config_path: sub_agents/locator.yaml\n      skip_summarization: false\n  - agent_tool:\n      config_path: sub_agents/checker.yaml\n      skip_summarization: false\n', 'sub_agents/locator.yaml': 'name: locator\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only code locator. Call it once with a short note on what to look for; it reads the repository and returns the file, function, line numbers and the change to make.\ninstruction: !include ../prompts/locator.md\ntools:\n  - run_command\n  - read_file\n  - get_code_neighbors\ngenerate_content_config: !include ../configs/sampling.yaml\n', 'sub_agents/checker.yaml': 'name: checker\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only change checker. Call it once after editing, with a one-line note of what the change should now do; it runs a scratch check and the nearby existing tests in its own context and returns VERDICT, EVIDENCE and FIX.\ninstruction: !include ../prompts/checker.md\ntools:\n  - run_command\n  - read_file\ngenerate_content_config: !include ../configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 0\n  include_thoughts: false\n', 'prompts/system.md': 'You are a careful Python maintainer. The repository is checked out at /workspace. Your job is to make the\nsmallest correct source change that resolves the issue below, so that the project\'s own (hidden) tests for\nthis issue pass. The issue is in the first user message. You have a hard budget of about 80 tool calls and\n5 minutes, so work in a straight line.\n\nWork in this order:\n\n1. Locate (one call). First call the `locator` tool once with a one-line note of what to find (for example\n   "where the `timeout` option of `Client.send` is applied"). It reads the code in a separate context and returns\n   FILES, CAUSE, CHANGE, CODE and CHECK. Then confirm its answer with one `read_file(path, start_line, end_line)`\n   of the named lines. Only if its answer is empty or clearly wrong, locate yourself with `git grep -n "name" -- \'*.py\' | head -30`\n   (there is no `rg`) and `read_file` of focused ranges; always cut long output with `| head`.\n2. Understand. Write down in a few sentences (as plain text, not only in your head) the file, function and\n   line numbers to change and what the expected behaviour is. Older tool outputs are dropped from the\n   conversation when it gets long; your own notes are kept.\n   If the issue shows a snippet, it usually describes the intended behaviour exactly; follow it.\n3. Edit. Change library code only (never tests, never /workspace/pytest.ini or /workspace/conftest.py).\n   Use `edit_file` with a short, unique `old_string` copied exactly from `read_file` output; make several small\n   edits rather than one large one. Keep the existing style, names and public signatures; when the issue asks\n   for a new parameter or option, add it with a backward-compatible default.\n4. Check (one call). Call the `checker` tool once with a one-line note of what the code should now do\n   (for example "`Client.send(timeout=None)` no longer raises"). In its own context it runs a scratch\n   reproduction and the nearest existing tests and returns VERDICT, EVIDENCE and FIX. If VERDICT is FAIL\n   because of your change, apply its FIX with `edit_file` and call `checker` once more at most. If the check\n   itself was broken, do not spend more calls on it. Only if `checker` errors, check yourself: write a scratch\n   script with `run_command` and a heredoc outside /workspace (`cat > /tmp/check.py <<\'EOF\' ... EOF`, then\n   `python /tmp/check.py 2>&1 | tail -20`) or run `python -m pytest -x -q tests/test_x.py -k name 2>&1 | tail -25`.\n   Command output is cut after its first 5,000 characters, so always end long commands with `| tail`.\n5. Submit. Run `git status --short` and `git diff` to confirm only intended source files changed, then call\n   `submit_patch` as your final action. Always submit before the budget runs out: a reasonable patch scores,\n   no patch never does. Use `get_status` (free) if you are unsure how much budget is left.\n\nDo not call `search_similar_code`: it returns whole function bodies with no length limit and can overflow\nthe context, which loses the task. If `get_code_neighbors` or `get_code_subgraph` errors, stop using them\n(they cannot see async functions) and use `git grep` and `read_file` instead.\n\nRules: no pip installs (the environment is offline and complete), no network, no edits outside /workspace\nsource files, no rewriting of unrelated code, no new test files in /workspace.\n\nThe issue you are solving (repeated here because the first message may be summarized away):\n\n{problem_description}\n', 'prompts/checker.md': 'You are a read-only checker for the Python repository at /workspace. Another agent has just edited the\nsource to resolve the issue below. You never edit files in /workspace; you only run checks and report.\nYour answer is all the other agent sees of your work, so make it short and exact.\n\nThe issue:\n\n{problem_description}\n\nHow to work (at most about 8 tool calls):\n1. `run_command`: `cd /workspace && git diff | head -80` to see the change.\n2. Write a scratch script that reproduces the issue\'s example (or the behaviour it asks for) with a\n   heredoc outside /workspace, e.g. `cat > /tmp/chk.py <<\'PYEOF\' ... PYEOF`, then\n   `cd /workspace && python /tmp/chk.py 2>&1 | tail -20`.\n3. Find the existing tests closest to the changed function, e.g.\n   `git grep -ln "function_name" -- \'test*\' \'*/test*\' | head -5`, and run only those:\n   `python -m pytest -x -q <file> -k <name> 2>&1 | tail -25`. Output is cut after its first 5,000\n   characters, so always end with `| tail`.\n4. Do not install anything and do not run the whole test suite.\n\nAnswer in exactly this format, and nothing else:\n\nVERDICT: PASS or FAIL\nEVIDENCE: <the 3-10 most telling output lines: the scratch result and the pytest summary line>\nFIX: <if FAIL because of the change: file, line and what to change; if the check itself was broken or no\nrelevant test exists, say so in one line; if PASS: none>\n', 'prompts/locator.md': 'You are a read-only code locator for the Python repository at /workspace. You never edit files. Another\nagent will make the change; your answer is all it sees of your work, so make it precise and short.\n\nThe issue:\n\n{problem_description}\n\nHow to work (at most about 12 tool calls):\n1. Pull exact identifiers out of the issue (function, class, method, option, error text, file name) and find\n   them with `run_command`, e.g. `git grep -n "name" -- \'*.py\' | grep -v test | head -30` (there is no `rg`).\n   Always cut long output with `| head` or `| tail`; output is cut after its first 5,000 characters.\n2. Read only the relevant regions with `read_file(path, start_line, end_line)` (at most 150 lines per call).\n   Follow the call path from the public entry point named in the issue to the code that behaves wrongly.\n   Use `get_code_neighbors` for callers or callees if `git grep` is not enough.\n3. Do not call `search_similar_code` (it can overflow the context).\n\nAnswer in exactly this format, and nothing else:\n\nFILES: <path>:<start>-<end> (<function or class>) [one line per place to change, most important first]\nCAUSE: <one or two sentences: what the code does now and why that is wrong for the issue>\nCHANGE: <concrete edit plan: which lines, what to add or replace, new parameter names and defaults>\nCODE: <the 5-15 current lines that must change, copied exactly from read_file without line numbers>\nCHECK: <one short python snippet or pytest -k command that shows the fix works>\n'}, 'pub013': {'eval_config.yaml': 'evaluation:\n  timeout_seconds: 60\n  max_tool_calls: 60\n  max_time_minutes: 5\n  max_turns: 100\n', 'agent.yaml': 'name: swe_workflow\nagent_class: SequentialAgent\ndescription: Read-only localization and concrete plan, followed by one coder that edits, verifies, and submits.\nsub_agents:\n  - config_path: sub_agents/explore.yaml\n  - config_path: sub_agents/coder.yaml\n', 'skills/swe-workflow/SKILL.md': "---\nname: swe-workflow\ndescription: Bounded grep search, a concrete small-step plan, one edit at a time, and observable verification.\n---\n\n1. Search the issue's exact symbols with scoped rg/grep, then read the matching source.\n2. State up to three concrete steps plus a pass/fail check; keep one step active.\n3. Apply one small source edit and inspect its diff before the next edit.\n4. Run the focused assertion/test. An exception or failed assertion must exit nonzero.\n5. Submit the verified patch. Do not edit tests/config or repeat an already answered search.\n", 'skills/verify-before-submit/SKILL.md': '---\nname: verify-before-submit\ndescription: >\n  Pre-submit gate: targeted test evidence, patch hygiene, then submit_patch.\n---\n\n# Verify before submit\n\n1. Run **only** the targeted test node or a short assertion-based repro for this fix. Printed booleans and swallowed exceptions are not passing evidence.\n2. `git status` / `git diff`: no scratch under `/workspace`, no test/config edits.\n3. Scratch belongs in `/tmp`.\n4. On passing assertions and a clean diff → `submit_patch` → short summary.\n5. If the targeted test fails → fix source again; do not edit tests.\n', 'sub_agents/explore.yaml': 'name: explore\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only grep localization and a concrete repair plan.\ninstruction: !include ../prompts/explore.md\ngenerate_content_config: !include ../configs/explore_sampling.yaml\noutput_key: repair_plan\ntools:\n  - run_command\n  - read_file\n  - get_code_neighbors\n  - get_code_subgraph\n', 'sub_agents/coder.yaml': 'name: swe_agent\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Apply the localized repair in small steps, verify, and submit.\ninstruction: !include ../prompts/system.md\ngenerate_content_config: !include ../configs/sampling.yaml\ninclude_contents: default\ntools:\n  - read_file\n  - edit_file\n  - run_command\n  - get_status\n  - submit_patch\nskills:\n  - skills/swe-workflow\n  - skills/verify-before-submit\n', 'configs/explore_sampling.yaml': 'temperature: 0.1\ntop_p: 0.95\nmax_output_tokens: 4096\nthinking_config:\n  thinking_budget: 1024\n  include_thoughts: false\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 2048\n  include_thoughts: false\n', 'prompts/system.md': 'Implement this issue in `/workspace`: {problem_description}\n\nThe read-only stage has finished localization. Its evidence and plan:\n{repair_plan?}\n\nIf the saved plan is absent, use the preceding search/read results to form a short concrete plan; do not restart broad exploration.\n\nYou are the sole CODER. Complete the provided plan one small step at a time.\n1. Read the named source region, then make the first minimal edit_file replacement. Do not re-search the repository or write a reproduction before this edit.\n2. Inspect the modified hunk with git diff --check and git diff -- path. If another edit is needed, do it separately.\n3. Run one short inline Python assertion (at most 6 executable lines) or one existing targeted test node. Check the requested behavior and one preserved behavior. No comments, no try/except, no scratch files. A failed assertion must exit nonzero. For serialization compare type and fields, not identity.\n4. If the check passes, submit_patch immediately. Then report the result in one sentence. If it fails, fix one source hunk and repeat the same check.\nFor additive changes, preserve existing aliases, accepted inputs, and defaults. Verify one old behavior as well as the new behavior; a new feature is not permission to delete existing behavior.\nWork offline. Do not install dependencies, change tests/config, or run full test discovery. Do not repeat the plan in prose. Tool calls should implement one small, verifiable action. Use the remaining task budget; reserve time to submit.\n', 'prompts/explore.md': "Localize this issue in `/workspace`: {problem_description}\nYou are the READ-ONLY first stage. The next stage will implement your plan.\n1. Use run_command for ONE bounded rg/grep search of the exact issue symbol in likely source. No Python scripts, no tests, no writes.\n2. Read the matching source region. A second narrow search/read is allowed only to resolve a missing definition.\n3. STOP tool use and return a compact plan (at most 100 words): source path and symbol; observed cause; 1–3 tiny edits; one pass/fail assertion and one behavior to preserve. Include the relevant current source lines so the coder has evidence. Do not reproduce the issue or implement anything. Do not call other agents.\n\nGraph navigation (read-only, before returning the plan):\n- Query a specific function or method found in source, preferably its qualified name. Never query broad class names such as FastAPI, APIRouter, or Body. If the issue names only a class, locate the relevant method first; skip graph lookup if no specific method is known.\n- Call get_code_neighbors once with that function/method name and max_neighbors=8. It returns node names without code. Omit edge_type; stored edges use lowercase calls.\n- You may make ONE additional graph query: get_code_subgraph(nodes=<at most 4 known qualified function/method names>) to inspect relationships. This returns names and edges without node code. Inspect useful neighbors using a bounded source read rather than retrieving full graph node code.\n- search_similar_code is unavailable here: k limits result count, not the size of a node's code; even k=1 can flood the context. Graphs are static hints, not complete call graphs. Empty results mean continue with the source evidence, not repeat the query.\n- Confirm any graph-derived fix claim in actual source. Keep the same compact handoff and stop after localization. No writes, reproduction, or tests.\n"}}          # name -> {relative path: text}; "@sample" = the official sample_submission
LIMIT = int(os.environ.get("G4_LIMIT", "1000"))
WORKERS = int(os.environ.get("G4_WORKERS", "6"))
SHARD = "0/1"            # "k/M": tasks with index % M == k


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

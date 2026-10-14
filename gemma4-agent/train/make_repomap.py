"""Build one ADK skill per repository: a compact map of where things live (ours).

For each of the four repositories in the competition (fastapi, rich, requests, httpx) it writes
submission/skills/repo-<name>/SKILL.md with
  1. the source files the practice tasks' reference fixes changed most often (tasks.jsonl `patch`),
  2. every source module with its top-level classes and functions (from the repository snapshot, ast),
  3. how the tests are laid out and the focused pytest command.
The agent loads the map of its task's repository before searching, so the localisation step needs fewer
git grep / read_file rounds (v7 on GPU: 92 of 258 tasks ran out of time without a patch).

    python gemma4-agent/train/make_repomap.py DATA_DIR
DATA_DIR holds tasks.jsonl and repos/<name>/ (competition data, not in this repo).
"""
import ast
import collections
import json
import os
import re
import sys

DATA = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "skills")
REPOS = {  # task repo -> (snapshot dir, package dir inside it, test dir, what the package is)
    "fastapi/fastapi": ("fastapi", "fastapi", "tests", "the FastAPI web framework (routing, dependency injection, "
                        "OpenAPI generation, request validation on Pydantic)"),
    "Textualize/rich": ("rich", "rich", "tests", "the Rich terminal rendering library (styles, markup, tables, "
                        "progress, syntax, console export)"),
    "psf/requests": ("requests", "src/requests", "tests", "the Requests HTTP library (sessions, adapters, models, "
                     "cookies, auth, utils)"),
    "encode/httpx": ("httpx", "httpx", "tests", "the HTTPX HTTP client (sync/async clients, transports, models, URLs)"),
}
MAX_SYMS = 14  # symbols listed per module (the rest is counted)

tasks = [json.loads(line) for line in open(os.path.join(DATA, "tasks.jsonl"), encoding="utf-8")]


def changed(patch):
    return re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)


def symbols(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            meth = [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not (m.name.startswith("__") and m.name not in ("__init__", "__call__", "__rich_console__"))]
            out.append(f"class {n.name}" + (f"({', '.join(meth[:6])}{', ...' if len(meth) > 6 else ''})" if meth else ""))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
            out.append(f"{n.name}()")
    return out


for repo, (snap, pkg, tdir, what) in REPOS.items():
    root = os.path.join(DATA, "repos", snap)
    src_hits, test_hits = collections.Counter(), collections.Counter()
    n_tasks = 0
    for t in tasks:
        if t["repo"] != repo:
            continue
        n_tasks += 1
        for f in changed(t["patch"]):
            src_hits[f] += 1
        for f in changed(t["test_patch"]):
            test_hits[f] += 1
    lines = []
    base = os.path.join(root, pkg)
    for dp, dns, fns in os.walk(base):
        dns[:] = sorted(d for d in dns if not d.startswith((".", "__pycache__")))
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            syms = symbols(p)
            more = f" +{len(syms) - MAX_SYMS} more" if len(syms) > MAX_SYMS else ""
            if syms:  # modules without public top-level names are left out (saves context)
                lines.append(f"- `{rel}`: " + ", ".join(syms[:MAX_SYMS]) + more)
    name = "repo-" + snap
    hot = [f"`{f}` ({c})" for f, c in src_hits.most_common(12)]
    hot_t = [f"`{f}`" for f, _ in test_hits.most_common(6)]
    tests = sorted(os.listdir(os.path.join(root, tdir)))
    n_test_files = sum(1 for f in tests if f.startswith("test_") and f.endswith(".py"))
    body = f"""---
name: {name}
description: Map of the {repo} repository ({what}). Load it first when the task's repository is {repo}; it says which file holds each class or function so you can go straight to read_file.
---

Repository `{repo}`: {what}. Source package: `{pkg}/`. Tests: `{tdir}/` ({n_test_files} `test_*.py` files at the top level).

Use this map to pick the file, then confirm with one scoped `git grep -n "<symbol>" {pkg}/` and `read_file` on that file only.
Do not grep the whole repository, docs or examples. The map was taken from one snapshot: when a name is missing, the code moved; fall back to `git grep`.

Files most often changed by fixes in this repository (count over {n_tasks} reference fixes): {", ".join(hot) or "-"}.
Tests most often extended by those fixes: {", ".join(hot_t) or "-"}.
Focused test run: `python -m pytest {tdir}/test_<module>.py -q -x 2>&1 | tail -20` (the test file is usually named after the module).

Module map (top-level classes with their methods, public functions):
""" + "\n".join(lines) + "\n"
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8").write(body)
    print(name, n_tasks, "tasks", len(lines), "modules", len(body), "chars ~", len(body) // 4, "tokens")

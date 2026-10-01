"""Turn the 129 public tasks into agent trajectories for LoRA training.

For each task the reference patch is replayed as the tool calls an efficient agent
would make, with tool outputs computed from the real repository at base_commit:

    run_command(git grep for the changed symbol) -> read_file(the region) ->
    edit_file(one call per hunk, or write_file for a new file) ->
    run_command(py_compile the changed files) -> submit_patch()

The messages follow the OpenAI chat format (assistant tool_calls + tool results), which
the training kernel renders with the Gemma 4 chat template. Tool outputs mimic the
harness JSON (HARNESS_README section 6). Test files are never touched, scratch work
would go to /tmp, and the agent always ends with submit_patch.

    python make_traj.py <tasks.jsonl> <repos_dir> <out.jsonl>

repos_dir holds full clones named after the repo (fastapi, rich, requests, httpx).
"""
import json
import re
import subprocess
import sys

SYSTEM = open(__file__.rsplit("/", 2)[0] + "/submission/prompts/system.md", encoding="utf-8").read()

TOOLS = [
    {"name": "run_command", "description": "Execute a shell command in /workspace.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a file from /workspace (1-indexed inclusive lines).",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"},
                                                     "end_line": {"type": "integer"}}, "required": ["filepath"]}},
    {"name": "edit_file", "description": "Replace old_string with new_string in an existing file.",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"},
                                                     "new_string": {"type": "string"}, "allow_multiple": {"type": "boolean"}},
                    "required": ["filepath", "old_string", "new_string"]}},
    {"name": "write_file", "description": "Create or overwrite a file in /workspace.",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["filepath", "content"]}},
    {"name": "search_similar_code", "description": "Find graph nodes similar to a symbol name.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "get_code_neighbors", "description": "Callers, callees and definitions of a symbol.",
     "parameters": {"type": "object", "properties": {"node": {"type": "string"}, "edge_type": {"type": "string"},
                                                     "max_neighbors": {"type": "integer"}}, "required": ["node"]}},
    {"name": "get_code_subgraph", "description": "Induced subgraph for a list of symbols.",
     "parameters": {"type": "object", "properties": {"nodes": {"type": "array", "items": {"type": "string"}}}, "required": ["nodes"]}},
    {"name": "get_status", "description": "Budget consumption and patch status.", "parameters": {"type": "object", "properties": {}}},
    {"name": "submit_patch", "description": "Capture git diff of /workspace as the final patch.", "parameters": {"type": "object", "properties": {}}},
]

MAX_LINES = 150
MAX_OUT = 5000


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout


def parse_patch(patch):
    """[(path, is_new, [hunk, ...])] with hunk = (old_start, [lines with ' ', '-', '+' prefixes])."""
    files, cur = [], None
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- "):
            old = ln[4:].strip()
            new = lines[i + 1][4:].strip() if i + 1 < len(lines) else ""
            path = re.sub(r"^b/", "", new)
            cur = (path, old == "/dev/null", [])
            files.append(cur)
            i += 2
            continue
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", ln)
        if m and cur is not None:
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff ")):
                if lines[i][:1] in (" ", "-", "+"):
                    body.append(lines[i])
                elif lines[i] == "":
                    body.append(" ")
                i += 1
            cur[2].append((int(m.group(1)), body))
            continue
        i += 1
    return files


def tool_msg(call_id, name, payload):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(payload, ensure_ascii=False)}


def call(call_id, name, args, text=""):
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


KEYWORDS = {"from", "import", "return", "self", "class", "def", "None", "True", "False", "async", "await", "with",
            "else", "elif", "raise", "pass", "lambda", "yield", "while", "assert", "except", "finally", "global"}


def symbol_for(path, hunk_start, src_lines, body):
    """The function/class the first hunk sits in, else an identifier from the removed lines."""
    for k in range(min(hunk_start + 3, len(src_lines)) - 1, -1, -1):
        m = re.match(r"\s*(?:async\s+)?(?:def|class)\s+(\w+)", src_lines[k])
        if m:
            return m.group(1)
    for ln in body:
        if ln[0] not in "-+":
            continue
        for w in re.findall(r"\b([A-Za-z_]\w{3,})\b", ln[1:]):
            if w not in KEYWORDS:
                return w
    return path.rsplit("/", 1)[-1].split(".")[0]


def trim_hunk(body, text):
    """Smallest old/new pair (keeping >= 1 context line each side when present) that is unique in text."""
    old_all = [ln[1:] for ln in body if ln[0] in " -"]
    new_all = [ln[1:] for ln in body if ln[0] in " +"]
    first = next(k for k, ln in enumerate(body) if ln[0] != " ")
    last = max(k for k, ln in enumerate(body) if ln[0] != " ")
    for ctx in (1, 2, 3, 5, 100):
        a, b = max(0, first - ctx), min(len(body), last + 1 + ctx)
        seg = body[a:b]
        old = "\n".join(ln[1:] for ln in seg if ln[0] in " -")
        new = "\n".join(ln[1:] for ln in seg if ln[0] in " +")
        if old and text.count(old) == 1:
            return old, new
    return "\n".join(old_all), "\n".join(new_all)


def mini_diff(path, old, new):
    o = ["-" + x for x in old.split("\n")]
    n = ["+" + x for x in new.split("\n")]
    return f"--- a/{path}\n+++ b/{path}\n" + "\n".join(o + n)


def build(task, repo):
    base = task["base_commit"]
    # a new file shows up as "@@ -0,0" (old path may still read a/<path>)
    files = [(p, new or all(h[0] == 0 for h in hs), hs) for p, new, hs in parse_patch(task["patch"]) if hs or new]
    src_files = [f for f in files if not re.search(r"(^|/)tests?/|test_", f[0])]
    if not src_files:
        return None
    user = (f"You are evaluating a software engineering task for repository {task['repo']}.\n\n"
            f"Problem Statement:\n{task['problem_statement']}\n")
    if task.get("hints_text", "").strip():
        user += f"\n## Hints:\n{task['hints_text'].strip()}\n"
    user += ("\n## Task Budget (Session terminates when any budget is exhausted)\n- Time allowance: 4.5 minutes\n"
             "- Tool calls allowance: 40 calls\n- Max loop iterations: 100 turns\n\n"
             "Work under /workspace. Call submit_patch once complete.\n")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    n = 0

    def nid():
        nonlocal n
        n += 1
        return f"call_{n}"

    path0, new0, hunks0 = src_files[0]
    if not new0:
        text0 = git(repo, "show", f"{base}:{path0}")
        lines0 = text0.split("\n")
        sym = symbol_for(path0, hunks0[0][0], lines0, hunks0[0][1])
        pkg = path0.split("/")[0] if "/" in path0 else "."
        cmd = f'grep -rn "{sym}" --include=*.py {pkg} | head -20'
        out = git(repo, "grep", "-n", sym, base, "--", f"{pkg}/*.py" if pkg != "." else "*.py")
        out = "\n".join(ln.split(":", 1)[1] for ln in out.splitlines()[:20])
        cid = nid()
        msgs.append(call(cid, "run_command", {"command": cmd}, f"The issue points at `{sym}`. Let me find where it is defined and used."))
        msgs.append(tool_msg(cid, "run_command", {"status": "ok", "stdout": out[:MAX_OUT], "stderr": "", "exit_code": 0}))

    for path, is_new, hunks in src_files:
        if is_new:
            content = "\n".join(ln[1:] for h in hunks for ln in h[1] if ln[0] == "+") + "\n"
            cid = nid()
            msgs.append(call(cid, "write_file", {"filepath": path, "content": content}, f"Create `{path}`."))
            msgs.append(tool_msg(cid, "write_file", {"status": "ok", "filepath": path, "size": len(content)}))
            continue
        text = git(repo, "show", f"{base}:{path}")
        lines = text.split("\n")
        lo = max(1, hunks[0][0] - 15)
        hi = min(len(lines), max(h[0] + len(h[1]) for h in hunks) + 10, lo + MAX_LINES - 1)
        cid = nid()
        msgs.append(call(cid, "read_file", {"filepath": path, "start_line": lo, "end_line": hi}, f"Read the relevant part of `{path}`."))
        msgs.append(tool_msg(cid, "read_file", {"status": "ok", "filepath": path, "content": "\n".join(lines[lo - 1:hi])[:10000],
                                                 "start_line": lo, "end_line": hi, "total_lines": len(lines),
                                                 "is_truncated": hi < len(lines)}))
        for k, (_, body) in enumerate(hunks):
            if not any(ln[0] in "-+" for ln in body):
                continue
            old, new = trim_hunk(body, text)
            if not old.strip():
                continue
            cid = nid()
            why = "Apply the fix." if k == 0 else "Next part of the change."
            msgs.append(call(cid, "edit_file", {"filepath": path, "old_string": old, "new_string": new}, why))
            msgs.append(tool_msg(cid, "edit_file", {"status": "ok", "filepath": path, "occurrences": 1, "strategy": "exact",
                                                     "diff": mini_diff(path, old, new)[:MAX_OUT], "is_truncated": False}))
            text = text.replace(old, new, 1)

    py = [p for p, _, _ in src_files if p.endswith(".py")]
    if py:
        cid = nid()
        msgs.append(call(cid, "run_command", {"command": "python -m py_compile " + " ".join(py) + " && git status --short"},
                         "Check that the edited files compile and only they changed."))
        msgs.append(tool_msg(cid, "run_command", {"status": "ok", "stdout": "\n".join(" M " + p for p in py), "stderr": "", "exit_code": 0}))
    cid = nid()
    msgs.append(call(cid, "submit_patch", {}, "The change is in place; submitting."))
    msgs.append(tool_msg(cid, "submit_patch", {"status": "ok", "patch_size": len(task["patch"]), "files_changed": len(src_files)}))
    msgs.append({"role": "assistant", "content": "Submitted the fix: " + ", ".join(p for p, _, _ in src_files) + "."})
    return {"instance_id": task["instance_id"], "messages": msgs}


def main():
    tasks, repos, out = sys.argv[1], sys.argv[2], sys.argv[3]
    names = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
    rows, skipped = [], 0
    max_calls = 30   # the submission's budget is 40 calls; longer replays teach the wrong habit
    for line in open(tasks, encoding="utf-8"):
        t = json.loads(line)
        r = build(t, f"{repos}/{names[t['repo']]}")
        if r is None or sum(1 for m in r["messages"] if m.get("tool_calls")) > max_calls:
            skipped += 1
            continue
        rows.append(r)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    calls = [sum(1 for m in r["messages"] if m.get("tool_calls")) for r in rows]
    print(f"{len(rows)} trajectories, {skipped} skipped, tool calls median {sorted(calls)[len(calls)//2]} max {max(calls)}")
    json.dump(TOOLS, open(out.rsplit(".", 1)[0] + "_tools.json", "w"), indent=1)


if __name__ == "__main__":
    main()

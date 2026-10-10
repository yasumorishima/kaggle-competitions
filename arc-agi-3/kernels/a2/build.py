"""Build kernels/a2/main.ipynb = kernels/m2base/main.ipynb (the Milestone-2 winner by dfranzen, unchanged) + our A2 parts.

Ours (yasunorim):
  1. deadline: in a real rerun the per-game limit is 540 min - elapsed - 6 min at the moment the benchmark starts
     (the winner's 532 min + ~10 min server start-up leaves no slack against the 9-hour limit).
  2. level dossier: when a level is completed, the exact action sequence of the final attempt is kept for the rest of
     the game. The winner's harness keeps only the (trimmed) chat history and the model's own functions across levels.
     dossier=1 appends it to the system prompt at once (a2d1-1: the head changes, the whole cached prefix is
     re-prefilled, 489 tok/s vs 578-640). dossier=2 puts it in the next user message (the tail) and moves it into the
     system prompt only when the trimmer drops messages, when the cached prefix is invalid anyway.

    python arc-agi-3/kernels/a2/build.py NAME [dossier=0|1|2] [passes=N] [fast=1]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
dossier = opts.get("dossier", "1")
passes = opts.get("passes", "1")
nb = json.load(open(os.path.join(HERE, "..", "m2base", "main.ipynb"), encoding="utf-8"))
cells = nb["cells"]
src = lambda i: "".join(cells[i]["source"])  # noqa: E731

PATCH = r'''
# ==== ours (yasunorim, kernels/a2): level dossier =====================================================
import os as _ours_os
if _ours_os.environ.get("ARC_OURS_DOSSIER", "0") in ("1", "2"):
    _ours_orig_update = ToolAgent._update_summarized_knowledge_from_step_summary

    def _ours_update(self):
        s = self._last_step_summary or {}
        acts = self.__dict__.setdefault("_ours_level_actions", [])
        acts.extend(str(a) for a in (s.get("executed_actions") or []))
        if s.get("game_over") and not s.get("level_transition"):
            acts.clear()                                  # keep the final (successful) attempt only
        if s.get("level_transition"):
            try:
                lvl = max(1, int(s.get("level") or 2) - 1)
            except (TypeError, ValueError):
                lvl = 0
            seq = " ".join(acts[-300:])
            more = f" (last 300 of {len(acts)})" if len(acts) > 300 else ""
            d = self.__dict__.setdefault("_ours_dossier", [])
            d.append(f"- Level {lvl}: solved in {len(acts)} actions{more}: {seq}")
            del d[:-6]
            acts.clear()
            base = self.__dict__.setdefault("_ours_base_prompt", self._system_prompt)
            block = ("## Solved levels of this game (exact action sequences of the winning attempts; kept "
                     "even when old messages are trimmed)\n" + "\n".join(d) + "\nLater levels usually reuse the same "
                     "mechanics with a bigger or changed layout: reuse what these sequences reveal instead of "
                     "rediscovering the rules, and aim for fewer actions.")
            self.__dict__["_ours_block"] = block
            if _ours_os.environ.get("ARC_OURS_DOSSIER") == "1":
                self._system_prompt = base + "\n\n" + block   # v1: rewrites the head = whole prefix re-prefilled
            else:
                self.__dict__["_ours_inline"] = block           # v2: ride on the next opening message (tail)
        return _ours_orig_update(self)

    ToolAgent._update_summarized_knowledge_from_step_summary = _ours_update

if _ours_os.environ.get("ARC_OURS_DOSSIER", "0") == "2":
    # v2 keeps the cached prefix: the record goes into the next user message (the tail, stored in history as sent),
    # and moves into the system prompt only when the trimmer has dropped messages, when the prefix is invalid anyway.
    _ours_orig_append = ToolAgent._append_context_message
    _ours_orig_trim = ToolAgent._trim_messages_for_context

    def _ours_append(self, messages, message):
        block = self.__dict__.pop("_ours_inline", None)
        if block and isinstance(message, dict) and message.get("role") == "user":
            c = message.get("content")
            if isinstance(c, str):
                message["content"] = block + "\n\n" + c
            elif isinstance(c, list):
                message["content"] = [{"type": "text", "text": block}, *c]
            else:
                self.__dict__["_ours_inline"] = block
        elif block:
            self.__dict__["_ours_inline"] = block
        return _ours_orig_append(self, messages, message)

    def _ours_trim(self, messages, **kw):
        out = _ours_orig_trim(self, messages, **kw)
        block = self.__dict__.get("_ours_block")
        if block and out and len(out) < len(messages):
            base = self.__dict__.setdefault("_ours_base_prompt", self._system_prompt)
            if self._system_prompt != base + "\n\n" + block:
                self._system_prompt = base + "\n\n" + block
                out[0] = {**out[0], "content": self._system_prompt}
        return out

    ToolAgent._append_context_message = _ours_append
    ToolAgent._trim_messages_for_context = _ours_trim
'''

# ---- the patch is appended to the harness after the winner's own patch is applied (cell 5)
i5 = next(i for i, c in enumerate(cells) if "harness patch applied successfully" in "".join(c["source"]))
c5 = src(i5)
anchor = "print('harness patch applied successfully')\n"
assert c5.count(anchor) == 1
c5 = c5.replace(anchor, anchor + (
    "# ours (yasunorim): A2 parts appended to the patched harness\n"
    f"os.environ['ARC_OURS_DOSSIER'] = '{dossier}'\n"
    f"_OURS_PATCH = {PATCH!r}\n"
    "with open(f'{BUNDLE_DIR}/src/ARC3-Inference/inference/agent/tool_agent.py', 'a') as _f:\n"
    "    _f.write(_OURS_PATCH)\n"
    "print('ours: A2 patch appended, dossier =', os.environ['ARC_OURS_DOSSIER'])\n"))
cells[i5]["source"] = c5

# ---- deadline: per-game limit from the time actually left (real rerun only)
i17 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# Make one-off changes to `bm`"))
c17 = src(i17)
old = "        bm.solver.max_runtime_s_per_game = 532*60\n"
assert c17.count(old) == 1
c17 = c17.replace(old, "        bm.solver.max_runtime_s_per_game = 532*60\n"
                  "        # ours: never past 9 h (540 min) - elapsed - 6 min of slack\n"
                  "        bm.solver.max_runtime_s_per_game = min(bm.solver.max_runtime_s_per_game,\n"
                  "                                               540*60 - (time.time() - NOTEBOOK_START_TIME) - 6*60)\n"
                  "        print('ours: per-game limit', round(bm.solver.max_runtime_s_per_game / 60, 1), 'min')\n")
c17 = c17.replace("bm.n_passes = int(os.environ.get('ARC_PASSES', '1'))", f"bm.n_passes = {int(passes)}")
# fast=1 (for submissions): the commit run only has to produce submission.parquet, so it plays one demo game for
# 5 minutes instead of the 2-hour local benchmark; the competition rerun (TRUE_SUBMISSION) is unchanged
if opts.get("fast", "0") == "1":
    old_ex = "demo_excluded_games = [] if TRUE_SUBMISSION else []\n"
    assert c17.count(old_ex) == 1
    others = ["ar25", "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0", "r11l", "re86",
              "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]   # all but ft09
    c17 = c17.replace(old_ex, f"demo_excluded_games = [] if TRUE_SUBMISSION else {others!r}  # ours: fast commit run\n")
    c17 += ("\nif not TRUE_SUBMISSION:  # ours: fast commit run, one game\n"
            "    bm.solver.max_runtime_s_per_game = 300\n    print('ours: fast commit run', bm.solver)\n")
cells[i17]["source"] = c17

# ---- diagnostics (ours, no behaviour change): summarise the SGLang serve.log in the notebook log, because kernel
# output files cannot be fetched from the cloud container. Per 10-minute bucket: decode batch size, queue,
# KV usage, generation throughput, and prefill new vs cached tokens (cache reuse).
DIAG = r"""
import re as _re, glob as _glob, collections as _co
_logs = sorted(_glob.glob('/kaggle/working/**/serve.log', recursive=True)) or sorted(_glob.glob('/kaggle/working/*.log'))
print('ours diag: server logs', _logs)
_b = _co.defaultdict(lambda: dict(dec=0, run=0, q=0, use=0.0, thr=0.0, pnew=0, pcached=0, pre=0))
_t0 = None
_num = lambda k, l: float(_re.search(k + r':\s*([\d.]+)', l).group(1)) if _re.search(k + r':\s*([\d.]+)', l) else 0.0
for _p in _logs:
    for _l in open(_p, errors='replace'):
        _m = _re.search(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', _l)
        if not _m:
            continue
        import datetime as _dt
        _t = _dt.datetime.strptime(_m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
        _t0 = _t0 or _t
        _k = int((_t - _t0) // 600)
        _x = _b[_k]
        if 'Decode batch' in _l:
            _x['dec'] += 1; _x['run'] += _num('#running-req', _l); _x['q'] += _num('#queue-req', _l)
            _x['use'] += _num('token usage', _l); _x['thr'] += _num(r'gen throughput \(token/s\)', _l)
        elif 'Prefill batch' in _l:
            _x['pre'] += 1; _x['pnew'] += _num('#new-token', _l); _x['pcached'] += _num('#cached-token', _l)
print('ours diag: min  decodes  run  queue  kv_use  gen_tok/s  prefill_new  prefill_cached')
for _k in sorted(_b):
    _x = _b[_k]; _d = max(1, _x['dec'])
    print(f"ours diag: {_k*10:4d} {_x['dec']:7d} {_x['run']/_d:5.1f} {_x['q']/_d:6.1f} {_x['use']/_d:7.2f} {_x['thr']/_d:9.1f} {int(_x['pnew']):12d} {int(_x['pcached']):14d}")
try:
    from inference.agent import tool_agent as _ta
    print('ours diag: gate', getattr(_ta, '_GATE_STATS', None))
except Exception as _e:
    print('ours diag: gate unavailable', repr(_e))
"""
cells.append({"cell_type": "code", "metadata": {}, "source": DIAG, "outputs": [], "execution_count": None})

cells[0]["source"] = ("# arc3 a2 (yasunorim)\n\nBase: the ARC-AGI-3 Milestone 2 solution by dfranzen "
                      "(https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution, built on Tufa Labs' Duck "
                      "harness), unchanged. Ours: a run deadline that respects the 9-hour limit, and a level dossier "
                      "(exact winning action sequences of solved levels kept in the system prompt), and a serve.log summary. Built by "
                      "arc-agi-3/kernels/a2/build.py in github.com/yasumorishima/kaggle-competitions.\n")
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(os.path.join(HERE, "main.ipynb"), "w", encoding="utf-8"), indent=1)
meta = json.load(open(os.path.join(HERE, "..", "m2base", "kernel-metadata.json")))
meta.update(id=f"yasunorim/arc3-{name}", title=f"arc3 {name}")
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote", name, "dossier", dossier, "passes", passes, "fast", opts.get("fast", "0"))

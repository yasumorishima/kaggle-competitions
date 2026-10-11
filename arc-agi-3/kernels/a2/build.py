"""Build kernels/a2/main.ipynb = kernels/m2base/main.ipynb (the Milestone-2 winner by dfranzen, unchanged) + our A2 parts.

Ours (yasunorim):
  1. deadline: in a real rerun the per-game limit is 540 min - elapsed - 6 min at the moment the benchmark starts
     (the winner's 532 min + ~10 min server start-up leaves no slack against the 9-hour limit).
  2. level dossier: when a level is completed, the exact action sequence of the final attempt is kept for the rest of
     the game. The winner's harness keeps only the (trimmed) chat history and the model's own functions across levels.
     dossier=1 appends it to the system prompt at once (a2d1-1: the head changes, the whole cached prefix is
     re-prefilled, 489 tok/s vs 578-640). dossier=2 puts it in the next user message (the tail) and moves it into the
     system prompt only when the trimmer drops messages, when the cached prefix is invalid anyway.

  3. turnlog=1: per game and level, analyzer turns / executed actions / generated tokens, printed at the end of the run
     ('ours diag: turns ...'); turnlog=2 also prints one 'ours turn' line per analyzer turn.
  4. reuse=1 (P2): the code of the python call that completed a level is dry-run on the next level's first board (action()
     wired to a recorder, zero real actions) and the actions it would play are shown in that turn's opener.
  5. autopilot=K (P1): a retained autopilot() written by the model plays the next short batch without an LLM request,
     for at most K turns in a row after a turn of the model's that changed the board; the next opener lists its moves.

    python arc-agi-3/kernels/a2/build.py NAME [dossier=0|1|2] [passes=N] [fast=1] [streams=N] [retry=TOKENS] [retrywait=MINUTES] [reuse=0|1] [turnlog=0|1|2] [autopilot=K]
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

# ==== ours (yasunorim, kernels/a2): second attempt (part 3) ===========================================
# The gate's chance term halves every 62k tokens spent on the current level, so a game stuck on a level is never
# scheduled again (a2d2: 7 of 25 games waited 50+ min at level 0-1; dc22 then took 3 levels in 20 min once resumed).
# Once per level, after ARC_OURS_RETRY_TOK tokens on it, the turn starts from a clean history (the memory sections and
# the dossier stay), the per-level counters restart so the gate prices the game as a fresh level, and the model is
# told its previous plan failed.
# a2r1-1: the token trigger fired only twice - stuck games are starved at the gate before they reach 100k tokens.
# ARC_OURS_RETRY_WAIT_MIN > 0 (a2r2): a game that has waited that long at the gate is priced as a fresh level (tokens
# and actions on the level = 0), once per level; when it wins a slot that way, its next turn is the second attempt.
_OURS_RETRY_TOK = int(_ours_os.environ.get("ARC_OURS_RETRY_TOK", "0") or 0)
_OURS_RETRY_WAIT = 60.0 * float(_ours_os.environ.get("ARC_OURS_RETRY_WAIT_MIN", "0") or 0)
_OURS_RETRY_STATS = {"retries": 0, "levels_after": 0, "fresh_admits": 0}
if _OURS_RETRY_WAIT > 0:
    import threading as _ours_th
    _ours_tls = _ours_th.local()
    _OURS_SNAP = {}                                  # id(snapshot) -> [agent, enqueue time, priced fresh]
    _ours_prev_mh = ToolAgent._maybe_handover

    def _ours_mh(self):
        _ours_tls.agent = self
        try:
            return _ours_prev_mh(self)
        finally:
            _ours_tls.agent = None

    _ours_prev_ho = _PriorityGate.handover

    def _ours_ho(gate, priority, snapshot=None):
        agent = getattr(_ours_tls, "agent", None)
        if snapshot is None or agent is None:
            return _ours_prev_ho(gate, priority, snapshot)
        key = id(snapshot)
        _OURS_SNAP[key] = [agent, time.monotonic(), False]
        try:
            return _ours_prev_ho(gate, priority, snapshot)
        finally:
            rec = _OURS_SNAP.pop(key, None)
            if rec is not None and rec[2]:
                agent._ours_retry_level = snapshot.level
                agent._ours_retry_pending = True
                _OURS_RETRY_STATS["fresh_admits"] += 1

    _ours_prev_sp = _PriorityGate._snapshot_priority

    def _ours_sp(gate, snapshot, now):
        rec = _OURS_SNAP.get(id(snapshot))
        if rec is not None:
            agent, t_enq, _ = rec
            fresh = (now - t_enq >= _OURS_RETRY_WAIT and snapshot.tokens > 0
                     and agent.__dict__.get("_ours_retry_level") != snapshot.level)
            rec[2] = fresh
            if fresh:
                snapshot = PrioritySnapshot(snapshot.level, 0, 0.0, snapshot.cost_multiplier, snapshot.total_levels)
        return _ours_prev_sp(gate, snapshot, now)

    ToolAgent._maybe_handover = _ours_mh
    _PriorityGate.handover = _ours_ho
    _PriorityGate._snapshot_priority = _ours_sp
if _OURS_RETRY_TOK > 0 or _OURS_RETRY_WAIT > 0:
    _ours_prev_trim = ToolAgent._trim_messages_for_context
    _ours_prev_upd = ToolAgent._update_summarized_knowledge_from_step_summary

    def _ours_retry_upd(self):
        s = self._last_step_summary or {}
        try:
            done = int(s.get("level") or 2) - 1
        except (TypeError, ValueError):
            done = -1
        if s.get("level_transition") and self.__dict__.get("_ours_retry_level") == done:
            _OURS_RETRY_STATS["levels_after"] += 1   # the retried level was completed
        return _ours_prev_upd(self)

    def _ours_retry_trim(self, messages, **kw):
        out = _ours_prev_trim(self, messages, **kw)
        try:
            if kw.get("preserve_recent") != 1 or not out or len(out) < 3:
                return out
            last = out[-1]
            if not (isinstance(last, dict) and last.get("role") == "user"):
                return out
            s = self._last_step_summary or {}
            level = int(s.get("level") or 1)
            spent = self._session_generated_tokens - getattr(self, "_tokens_at_level_start", 0)
            if self.__dict__.pop("_ours_retry_pending", False):
                pass                                  # won a slot as a fresh level after waiting (wait trigger)
            elif _OURS_RETRY_TOK <= 0 or spent < _OURS_RETRY_TOK or self.__dict__.get("_ours_retry_level") == level:
                return out
            self._ours_retry_level = level
            acts = max(0, _priority_action_count(s) - self._actions_at_level_start)
            note = (f"FRESH ATTEMPT: your previous attempt at this level used {spent} tokens and {acts} actions without "
                    "completing it, so its message history was cleared. Your memory sections and any solved-level "
                    "record are kept. Treat the current plan as failed: list what that attempt established, then test "
                    "a different hypothesis about the goal or the mechanics before repeating old moves.")
            c = last.get("content")
            if isinstance(c, str):
                last = {**last, "content": note + "\n\n" + c}
            elif isinstance(c, list):
                last = {**last, "content": [{"type": "text", "text": note}, *c]}
            self._tokens_at_level_start = self._session_generated_tokens
            self._actions_at_level_start = _priority_action_count(s)
            self._history_messages = []
            self._note_history_evicted()
            _OURS_RETRY_STATS["retries"] += 1
            print(f"ours retry: level {level} after {spent} tokens, {acts} actions", flush=True)
            return [out[0], last]
        except Exception as e:  # never break a game over the retry
            print("ours retry: skipped", repr(e), flush=True)
            return out

    ToolAgent._trim_messages_for_context = _ours_retry_trim
    ToolAgent._update_summarized_knowledge_from_step_summary = _ours_retry_upd

# ==== ours (yasunorim, kernels/a2): turn log (turnlog) and winning-code reuse (P2, reuse) ==============
# turnlog=1: per game and level, analyzer turns / executed actions / generated tokens, printed once at the end by the
# diag cell (_ours_report; the Kaggle log keeps only its tail). turnlog=2 also prints one line per analyzer turn.
# reuse=1: when a python call completes a level, its code is kept. At the first analyzer turn on the next level the code
# is re-run in a fresh sandbox process whose action() is wired to a recorder, never to the game (zero real actions),
# and the opener gets one line with the actions it would play there.
_OURS_TURNLOG = int(_ours_os.environ.get("ARC_OURS_TURNLOG", "0") or 0)
_OURS_REUSE = _ours_os.environ.get("ARC_OURS_REUSE", "0") == "1"
_OURS_REUSE_CAP = 200          # recorded actions before the recorder refuses (stops loops on the frozen board)
_OURS_REUSE_SHOW = 40          # actions shown in the opener
_OURS_REUSE_TIMEOUT = 5        # seconds of wall time for the dry run
_OURS_REUSE_STATS = {"attempts": 0, "nonempty": 0, "empty": 0, "errors": 0, "capped": 0, "literal": 0,
                     "injected": 0, "checked": 0, "prefix_match": 0, "first_batch_solved": 0, "levels_after": 0,
                     "dry_ms": 0}
_OURS_TURNS = {}               # game -> {"turns", "exec_turns", "acts", "tok", "levels", "lv": {level: [turns, acts, tok, solved]}}
_ours_lock = threading.Lock()
_OURS_ERRS = {"n": 0}


def _ours_err(where, e):
    with _ours_lock:
        _OURS_ERRS["n"] += 1
        n = _OURS_ERRS["n"]
    if n <= 5:
        print("ours turnlog/reuse: skipped", where, repr(e)[:200], flush=True)


def _ours_game_key(state_path):
    try:
        name = Path(state_path).name
        m = re.match(r"(.+?)_p(\d+)_" + re.escape(RUNTIME_STATE_FILENAME) + "$", name)
        if not m:
            return name[:24]
        g = m.group(1).split("-")[0]
        return g if m.group(2) == "0" else f"{g}/p{m.group(2)}"
    except Exception:
        return "?"


def _ours_act_key(a):
    # one spelling for a requested action and an executed one: UP, MOUSE(3,5), ...
    try:
        if isinstance(a, dict):
            if "row" in a or "col" in a:
                return f"MOUSE({a.get('row')},{a.get('col')})"
            a = a.get("action", "")
        s = str(a).strip().upper().replace("ROW=", "").replace("COL=", "").replace(" ", "")
        if "(" in s:
            return s
        return to_model_action(to_engine_action(s) or s) or s
    except Exception:
        return str(a)


# ==== ours (yasunorim, kernels/a2): zero-token autopilot (P1, autopilot=K) =============================
# The model may keep a retained function autopilot() that returns the next short batch while its plan holds. After a
# turn of the model's that changed the board (no stop, no level change, no game over), the next turn runs
#     autopilot() -> action(batch)
# through the model's own python path (_dispatch_tool: sandbox, guards, batch no-op stop, summaries, dossier, death
# ledger) with no LLM request, for at most K turns in a row. The first stop (None, error, refusal, no-op, game over,
# level change, the K cap) disarms it until the model itself makes progress again; a call that executed nothing falls
# through to the model in the same call. The next opener lists what autopilot played and why it stopped.
_OURS_AP_K = int(_ours_os.environ.get("ARC_OURS_AUTOPILOT", "0") or 0)
_OURS_AP_BATCH = 10            # actions per autopilot call (the rest of a longer list is dropped)
_OURS_AP_TIMEOUT = 15          # seconds for one autopilot snippet (the model's own snippets get up to 30)
_OURS_AP_SHOW = 60             # actions listed in the next opener
_OURS_AP_STATS = {"turns": 0, "actions": 0, "levels": 0, "game_overs": 0, "fallbacks": 0, "armed": 0, "stops": {}}
_OURS_AP_SNIPPET = (
    "_ours_ap_act = action\n"
    "action = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("
    "'autopilot() must return its actions, not call action()'))\n"
    "try:\n"
    "    _ours_ap_out = autopilot()\n"
    "finally:\n"
    "    action = _ours_ap_act\n"
    "if isinstance(_ours_ap_out, (str, dict)):\n"
    "    _ours_ap_out = [_ours_ap_out]\n"
    "_ours_ap_list = []\n"
    "if _ours_ap_out:\n"
    "    for _ours_ap_x in _ours_ap_out:\n"
    "        _ours_ap_list.append(_ours_ap_x)\n"
    f"        if len(_ours_ap_list) >= {_OURS_AP_BATCH}:\n"
    "            break\n"
    "if _ours_ap_list:\n"
    "    action(_ours_ap_list)\n"
    "else:\n"
    "    print('OURS_AUTOPILOT_NONE')\n"
)
_OURS_AP_PROMPT = (
    "\n\nAutopilot (optional): you may define a retained function `autopilot()` with no parameters that reads the "
    "usual globals (`current_frame`, `history`, `valid_actions`, ...) and returns the next short list of actions (at "
    f"most {_OURS_AP_BATCH}; extra ones are dropped) while your current plan is still clearly valid, or None as soon as "
    "it is unsure. It must return the actions, never call `action()` itself, and like every retained function it "
    "keeps no state between calls, so derive the position and the next step from the board each time. When it exists "
    "and your last step changed the board without a stop, the harness calls autopilot() and plays its list instead of "
    f"asking you, for up to {max(1, _OURS_AP_K)} turns in a row, stopping at the first None, error, refused or "
    "no-change action, level change or game over. Your next message then lists what it played. This saves your "
    "time for thinking, but every wasted action lowers the score: use it only for mechanical continuations (walk to "
    "a computed cell, repeat a confirmed pattern), and redefine it to return None when the plan changes."
)
_OURS_AP_REFUSALS = ("KnownNoOp", "KnownDeath", "StaleState", "RepeatedAction", "was NOT executed", "NOT executed")


def _ours_ap_stop(self, reason, acts=None, detail=""):
    d = self.__dict__
    d["_ours_ap_armed"] = False
    d["_ours_ap_run"] = 0
    d.setdefault("_ours_ap_log", []).append((list(acts or []), reason, detail))
    with _ours_lock:
        st = _OURS_AP_STATS["stops"]
        st[reason] = st.get(reason, 0) + 1


def _ours_ap_note(self):
    log_ = self.__dict__.pop("_ours_ap_log", None)
    if not log_:
        return None
    played = [a for acts, _, _ in log_ for a in acts]
    turns = sum(1 for acts, _, _ in log_ if acts)
    reason, detail = log_[-1][1], log_[-1][2]
    why = {"none": "it returned None", "error": "it raised an error", "refused": "the harness refused its action",
           "noop": "its last action changed nothing on the board", "no_op_action": "an action in its batch changed "
           "nothing, so the batch stopped", "known_noop": "the no-op guard refused an action",
           "known_death": "the death guard refused an action", "stale_state": "it repeated an action that had just "
           "changed nothing", "repeated_action_in_state": "it returned to a board it had already acted from",
           "level": "it completed the level", "game_over": "the game ended (game over)",
           "cap": f"it reached the limit of {_OURS_AP_K} turns in a row", "progress": "it was still progressing",
           }.get(reason, f"it stopped ({reason})")
    if detail:
        why += ": " + " ".join(str(detail).split())[-200:].rstrip(".")
    if played:
        shown = " ".join(str(a) for a in played[:_OURS_AP_SHOW]) + (" ..." if len(played) > _OURS_AP_SHOW else "")
        head = (f"AUTOPILOT: since your last reply the harness called your retained autopilot() and played "
                f"{len(played)} actions over {turns} turn(s) without asking you: {shown}. The 'previous sequence' "
                f"below is its last batch. It stopped because {why}.")
    else:
        head = f"AUTOPILOT: the harness called your retained autopilot() and it played nothing, because {why}."
    return head + " Check the board, then continue yourself; fix autopilot() or make it return None if its plan " \
                  "no longer fits."


def _ours_ap_progress(self, s):
    """None when the step made clean progress, else the reason it did not."""
    if not s or s.get("stale"):
        return "none"
    if s.get("level_transition"):
        return "level"
    if s.get("game_over") or s.get("run_complete"):
        return "game_over"
    sr = s.get("stop_reason")
    if sr and sr != "level_completed":
        return str(sr)
    if not s.get("gameplay_changed"):
        return "noop"
    last = getattr(self, "_last_action_call_result", None) or {}
    per = last.get("gameplay_changed_per_action")
    if isinstance(per, list) and per and not per[-1]:
        return "noop"
    return None


def _ours_ap_try(self, state_path, valid_actions, step_env, transcript_path, should_stop):
    """Run one autopilot turn. Returns an AnalyzerTurnResult when actions were executed, else None (ask the model)."""
    d = self.__dict__
    if not d.get("_ours_ap_armed") or getattr(self, "_resume_after_yield", False):
        return None
    if "autopilot" not in (getattr(self, "_kept_functions", None) or {}):
        d["_ours_ap_armed"] = False
        return None
    if d.get("_ours_ap_run", 0) >= _OURS_AP_K:
        _ours_ap_stop(self, "cap")
        return None
    if should_stop is not None and should_stop():
        return None
    if step_env is None or not Path(state_path).exists():
        return None
    self._ensure_session(state_path)
    before = self._last_step_summary
    old_timeout = self._python_timeout
    self._step_env_callback = step_env
    solver_obj = getattr(step_env, "__self__", None)
    if self._noop_repeat_guard is not None and solver_obj is not None:
        solver_obj.action_guard_hook = self._action_guard_hook
    self._current_valid_actions = _normalize_valid_actions(valid_actions)
    try:
        self._python_timeout = min(old_timeout, _OURS_AP_TIMEOUT)
        disp = self._dispatch_tool(state_path, "python", {"code": _OURS_AP_SNIPPET})
    finally:
        self._python_timeout = old_timeout
        if solver_obj is not None and hasattr(solver_obj, "action_guard_hook"):
            solver_obj.action_guard_hook = None
        self._step_env_callback = None
        self._current_valid_actions = []
    content = str(getattr(disp, "content", "") or "")
    try:
        payload = json.loads(content)
    except Exception:
        payload = {}
    err = ""
    if isinstance(payload, dict):
        r = payload.get("result")
        r = (r.get("last_action_call_result") or r) if isinstance(r, dict) else {}
        err = str(payload.get("error") or (r.get("error") if isinstance(r, dict) else "") or "")
    s = self._last_step_summary
    executed = bool(getattr(disp, "step_executed", False)) and s is not before and isinstance(s, dict)
    try:
        lg = transcript_path or (Path(state_path).parent / f"{Path(state_path).stem}_analyzer.txt")
        _append_transcript_section(lg, "AUTOPILOT (no LLM request)", content[:2000])
    except Exception:
        pass
    if not executed:
        if "OURS_AUTOPILOT_NONE" in content and not err:
            reason = "none"
        elif any(m in err for m in _OURS_AP_REFUSALS):
            reason = "refused"
        else:
            reason = "error"
        msg = err.strip().splitlines()[-1] if err.strip() else ""
        _ours_ap_stop(self, reason, None, msg if reason != "none" else "")
        if reason == "none" and not any(acts for acts, _, _ in d.get("_ours_ap_log") or []):
            d.pop("_ours_ap_log", None)    # a plain None is the documented "ask me": no note, no context spent
        with _ours_lock:
            _OURS_AP_STATS["fallbacks"] += 1
        return None
    acts = [str(a) for a in (s.get("executed_actions") or [])]
    why = _ours_ap_progress(self, s)
    with _ours_lock:
        _OURS_AP_STATS["turns"] += 1
        _OURS_AP_STATS["actions"] += int(s.get("executed_count") or len(acts))
        _OURS_AP_STATS["levels"] += 1 if s.get("level_transition") else 0
        _OURS_AP_STATS["game_overs"] += 1 if (s.get("game_over") and not s.get("level_transition")) else 0
    d["_ours_ap_flag"] = 1
    if why is None and not err:
        d["_ours_ap_run"] = d.get("_ours_ap_run", 0) + 1
        d.setdefault("_ours_ap_log", []).append((acts, "progress", ""))
    else:
        msg = err.strip().splitlines()[-1] if err.strip() and why in (None, "error") else ""
        _ours_ap_stop(self, why or "error", acts, msg)
    self._resume_after_yield = False
    self._resume_reason = ""
    return AnalyzerTurnResult(step_executed=True, reasoning="", yielded_control=False)


if _OURS_AP_K > 0 and _persistent_functions():
    _ours_prev_bsp = _build_system_prompt

    def _build_system_prompt(*args, **kw):   # noqa: F811  (module global: ToolAgent.__init__ looks it up by name)
        return _ours_prev_bsp(*args, **kw) + _OURS_AP_PROMPT

    _ours_prev_bup = ToolAgent._build_user_prompt
    _ours_prev_an_ap = ToolAgent.analyze

    def _ours_ap_bup(self, *args, **kw):
        out = _ours_prev_bup(self, *args, **kw)
        note = self.__dict__.pop("_ours_ap_note_pending", None)
        if note and isinstance(out, str):
            return note + "\n\n" + out
        return out

    def _ours_ap_an(self, state_path, action_num, valid_actions=None, step_env=None, transcript_path=None,
                    analysis_step=None, transcript_updated=None, request_timeout_seconds=None, should_stop=None):
        res = None
        try:
            res = _ours_ap_try(self, state_path, valid_actions, step_env, transcript_path, should_stop)
        except Exception as e:  # never break a game over the autopilot: disarm and ask the model
            _ours_err("autopilot", e)
            try:
                _ours_ap_stop(self, "error", None, repr(e)[:160])
                self._step_env_callback = None
            except Exception:
                pass
            res = None
        if res is not None:
            return res
        try:
            self.__dict__.pop("_ours_ap_note_pending", None)
            if not getattr(self, "_resume_after_yield", False):
                note = _ours_ap_note(self)
                if note:
                    self._ours_ap_note_pending = note
        except Exception as e:
            _ours_err("autopilot-note", e)
        out = _ours_prev_an_ap(self, state_path, action_num, valid_actions=valid_actions, step_env=step_env,
                               transcript_path=transcript_path, analysis_step=analysis_step,
                               transcript_updated=transcript_updated,
                               request_timeout_seconds=request_timeout_seconds, should_stop=should_stop)
        try:
            self.__dict__.pop("_ours_ap_note_pending", None)
            d = self.__dict__
            if out is not None and getattr(out, "step_executed", False):
                ok = _ours_ap_progress(self, self._last_step_summary) is None
                armed = ok and "autopilot" in (getattr(self, "_kept_functions", None) or {})
                if armed and not d.get("_ours_ap_armed"):
                    with _ours_lock:
                        _OURS_AP_STATS["armed"] += 1
                d["_ours_ap_armed"] = armed
                d["_ours_ap_run"] = 0
            elif out is None or not getattr(out, "yielded_control", False):
                d["_ours_ap_armed"] = False
        except Exception as e:
            _ours_err("autopilot-arm", e)
        return out

    ToolAgent._build_user_prompt = _ours_ap_bup
    ToolAgent.analyze = _ours_ap_an


if _OURS_TURNLOG > 0 or _OURS_REUSE:
    _ours_prev_rpt = ToolAgent._run_python_tool
    _ours_prev_an = ToolAgent.analyze

    def _ours_rpt(self, state_path, arguments):
        before = self._last_step_summary
        res = _ours_prev_rpt(self, state_path, arguments)
        try:
            s = self._last_step_summary or {}
            if not getattr(res, "step_executed", False) or s is before:
                return res
            self._ours_turn_exec = self.__dict__.get("_ours_turn_exec", 0) + int(s.get("executed_count") or 0)
            playing = self.__dict__.get("_ours_turn_level") or 1
            if _OURS_REUSE:
                check = self.__dict__.pop("_ours_reuse_check", None)
                if check is not None:
                    done = [_ours_act_key(x) for x in (s.get("executed_actions") or [])]
                    m = min(len(done), len(check))
                    with _ours_lock:
                        _OURS_REUSE_STATS["checked"] += 1
                        if m and done[:m] == check[:m]:
                            _OURS_REUSE_STATS["prefix_match"] += 1
                        if s.get("level_transition"):
                            _OURS_REUSE_STATS["first_batch_solved"] += 1
            if s.get("level_transition"):
                self._ours_turn_solved = True
                if _OURS_REUSE:
                    if self.__dict__.get("_ours_reuse_shown_level") == playing:
                        with _ours_lock:
                            _OURS_REUSE_STATS["levels_after"] += 1
                    if not (s.get("run_complete") or s.get("done")):
                        fns = self.__dict__.get("_ours_fn_snap")
                        if fns is None:
                            fns = dict(getattr(self, "_kept_functions", None) or {})
                        self._ours_reuse_pending = {"level": playing, "code": str(arguments.get("code", "")),
                                                    "fns": dict(fns)}
        except Exception as e:  # never break a game over the bookkeeping
            _ours_err("rpt", e)
        return res

    def _ours_an(self, state_path, *args, **kw):
        try:
            t0 = int(self._session_generated_tokens)
            s = self._last_step_summary or {}
            level = int(s.get("level") or 1)
            self._ours_turn_level, self._ours_turn_exec, self._ours_turn_solved = level, 0, False
            self.__dict__.pop("_ours_ap_flag", None)
            self.__dict__.pop("_ours_reuse_note", None)
        except Exception as e:
            _ours_err("an-pre", e)
            t0, level = None, 0
        reused = 0
        try:
            if _OURS_REUSE and not getattr(self, "_resume_after_yield", False):
                pend = self.__dict__.pop("_ours_reuse_pending", None)
                if pend:
                    va = kw.get("valid_actions", args[1] if len(args) > 1 else None)
                    note = _ours_dry_run(self, state_path, va, pend, level)
                    if note:
                        self._ours_reuse_note = note
                        self._ours_reuse_shown_level = level
                        reused = 1
        except Exception as e:
            _ours_err("dry-run", e)
            with _ours_lock:
                _OURS_REUSE_STATS["errors"] += 1
        try:
            return _ours_prev_an(self, state_path, *args, **kw)
        finally:
            try:
                self.__dict__.pop("_ours_reuse_note", None)   # not consumed by an opener: drop, never leak later
                if _OURS_TURNLOG > 0 and t0 is not None:
                    tok = max(0, int(self._session_generated_tokens) - t0)
                    n = int(self.__dict__.get("_ours_turn_exec", 0))
                    solved = bool(self.__dict__.get("_ours_turn_solved"))
                    ap = int(self.__dict__.pop("_ours_ap_flag", 0) or 0)
                    g = _ours_game_key(state_path)
                    with _ours_lock:
                        r = _OURS_TURNS.setdefault(g, {"turns": 0, "exec_turns": 0, "acts": 0, "tok": 0, "levels": 0,
                                                       "lv": {}})
                        r["turns"] += 1
                        if ap:
                            r["ap_turns"] = r.get("ap_turns", 0) + 1
                            r["ap_acts"] = r.get("ap_acts", 0) + n
                        r["exec_turns"] += 1 if n else 0
                        r["acts"] += n
                        r["tok"] += tok
                        r["levels"] += 1 if solved else 0
                        lv = r["lv"].setdefault(level, [0, 0, 0, 0])
                        lv[0] += 1; lv[1] += n; lv[2] += tok; lv[3] = max(lv[3], 1 if solved else 0)
                    if _OURS_TURNLOG >= 2:
                        print(f"ours turn {g} L{level} exec={n} tok={tok} reuse={reused}"
                              + (f" ap={ap}" if _OURS_AP_K > 0 else ""), flush=True)
            except Exception as e:
                _ours_err("an-post", e)

    ToolAgent._run_python_tool = _ours_rpt
    ToolAgent.analyze = _ours_an


def _ours_dry_run(self, state_path, valid_actions, pend, level):
    # Re-run the code that completed the previous level against the current board. action() in the sandbox reaches
    # only _rec below: it records and answers with a synthetic result. Nothing here calls self._step_env_callback (it
    # is None between analyzer turns anyway), the solver, or the guards; the sandbox is a fresh process, so the
    # snippet's variables vanish with it, and the functions it would retain are discarded.
    code = pend.get("code") or ""
    fns = pend.get("fns") or {}
    if not code.strip():
        return None
    with _ours_lock:
        _OURS_REUSE_STATS["attempts"] += 1
    frame, hist = load_runtime_state(Path(state_path))
    va = list(_normalize_valid_actions(valid_actions))
    state = {"current_frame": _ascii_frame_view_payload(frame), "history": _ascii_history_view_payload(hist),
             "valid_actions": va, "death_ledger": None, "last_action_call_result": {}}
    rec, capped = [], []

    def _rec(actions, *, stale_after=None):
        acts = self._normalize_python_actions(actions)      # pure: validation only
        if len(rec) + len(acts) > _OURS_REUSE_CAP:
            capped.append(1)
            res = {"executed": False, "level": level, "state": "NOT_FINISHED", "valid_actions": va,
                   "board_changed": False, "done": False, "level_completed": False, "game_over": False,
                   "run_complete": False, "requested_count": len(acts), "executed_count": 0, "stopped_early": True,
                   "stop_reason": "known_noop", "stop_detail": "dry run: action cap reached"}
        else:
            keys = [_ours_act_key(a) for a in acts]
            rec.extend(keys)
            res = {"executed": True, "level": level, "state": "NOT_FINISHED", "valid_actions": va,
                   "board_changed": True, "gameplay_changed": True, "done": False, "level_completed": False,
                   "game_over": False, "run_complete": False, "requested_count": len(acts),
                   "executed_count": len(acts), "stopped_early": False, "executed_actions": keys,
                   "action_display": keys[-1] if keys else ""}
        return {"action_result": res, "state": {**state, "last_action_call_result": res}}

    t = time.monotonic()
    out = run_sandboxed_python(
        code=code, timeout_seconds=_OURS_REUSE_TIMEOUT, initial_state=state, action_handler=_rec,
        animation_handler=None, kept_functions=(list(fns.values()) if _persistent_functions() else None),
        retain_imports=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False),
        repair_hints=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False))
    err = "" if capped else str((out or {}).get("error") or "").strip()
    lvl_done = pend.get("level") or max(1, level - 1)
    reads = any(w in code for w in ("current_frame", "latest_frame", "history", "transitions", "previous_frame",
                                    "last_action_frame", "valid_actions", "frame_diff", "last_action_call_result"))
    reads = reads or any(re.search(r"\b" + re.escape(n) + r"\s*\(", code) for n in fns)
    with _ours_lock:
        _OURS_REUSE_STATS["dry_ms"] += int(1000 * (time.monotonic() - t))
        _OURS_REUSE_STATS["nonempty" if rec else "empty"] += 1
        _OURS_REUSE_STATS["errors"] += 1 if err else 0
        _OURS_REUSE_STATS["capped"] += 1 if capped else 0
        _OURS_REUSE_STATS["literal"] += 0 if reads else 1
    head = (f"Your level-{lvl_done} winning code, re-run on this new board (dry run, nothing executed; the board does "
            "not change between its action calls)")
    shown = " ".join(rec[:_OURS_REUSE_SHOW]) + (" ..." if len(rec) > _OURS_REUSE_SHOW else "")
    if err:
        msg = err.splitlines()[-1] if err.splitlines() else err
        msg = msg[:160]
        text = head + (f", requested {len(rec)} actions ({shown}) and then raised: {msg}" if rec
                       else f", raised before requesting any action: {msg}")
    elif rec:
        text = head + f", would play: {shown} ({len(rec)} actions total"
        text += (f", stopped at the {_OURS_REUSE_CAP}-action cap)" if capped else ")")
    else:
        text = head + ", ran without requesting any action."
    if not reads:
        text += " That code does not read the board, so this is a literal replay of its fixed moves."
    text += " Use it only if it fits this level's layout."
    return (text, rec if rec else None)


if _OURS_REUSE:
    _ours_prev_rrf = ToolAgent._record_retained_functions
    _ours_prev_app2 = ToolAgent._append_context_message

    def _ours_rrf(self, sandbox_result, payload):
        out = _ours_prev_rrf(self, sandbox_result, payload)
        try:   # the functions as they stood after the call (the levelup scope would clear them at the transition)
            self._ours_fn_snap = dict(getattr(self, "_kept_functions", None) or {})
        except Exception:
            pass
        return out

    def _ours_app2(self, messages, message):
        note = self.__dict__.pop("_ours_reuse_note", None)
        try:
            if note:
                text, proposed = note
                c = message.get("content") if isinstance(message, dict) else None
                if isinstance(message, dict) and message.get("role") == "user" and isinstance(c, (str, list)):
                    message["content"] = (text + "\n\n" + c) if isinstance(c, str) else [{"type": "text", "text": text}, *c]
                    if proposed:
                        self._ours_reuse_check = list(proposed)
                    with _ours_lock:
                        _OURS_REUSE_STATS["injected"] += 1
                else:
                    self._ours_reuse_note = note
        except Exception as e:
            _ours_err("app", e)
        return _ours_prev_app2(self, messages, message)

    ToolAgent._record_retained_functions = _ours_rrf
    ToolAgent._append_context_message = _ours_app2


def _ours_report():
    try:
        if _OURS_TURNLOG > 0:
            tot = {"games": 0, "turns": 0, "acts": 0, "tok": 0, "levels": 0, "tok_solved": 0, "tok_solved2": 0,
                   "levels2": 0}
            for g in sorted(_OURS_TURNS):
                r = _OURS_TURNS[g]
                per = " ".join(f"L{k}{'' if v[3] else '~'}:{v[0]}t/{v[1]}a/{v[2] / 1000:.1f}k"
                               for k, v in sorted(r["lv"].items()))
                apx = f" ap={r['ap_turns']}t/{r['ap_acts']}a" if r.get("ap_turns") else ""
                print(f"ours diag: turns {g} turns={r['turns']} exec={r['exec_turns']} acts={r['acts']} "
                      f"tok={r['tok']} levels={r['levels']}{apx} | {per}", flush=True)
                tot["games"] += 1
                for k in ("turns", "acts", "tok", "levels"):
                    tot[k] += r[k]
                for k, v in r["lv"].items():
                    if v[3]:
                        tot["tok_solved"] += v[2]
                        if k >= 2:
                            tot["tok_solved2"] += v[2]
                            tot["levels2"] += 1
            lv = max(1, tot["levels"])
            print(f"ours diag: turns total {tot} tok/level={tot['tok'] // lv} "
                  f"tok/solved-level={tot['tok_solved'] // lv} "
                  f"tok/solved-level(L>=2)={tot['tok_solved2'] // max(1, tot['levels2'])} "
                  f"acts/level={tot['acts'] // lv}  ('~' = level not completed)", flush=True)
        if _OURS_REUSE:
            print("ours diag: reuse", dict(_OURS_REUSE_STATS), flush=True)
        if _OURS_AP_K > 0:
            print("ours diag: autopilot", {**_OURS_AP_STATS, "stops": dict(_OURS_AP_STATS["stops"])}, flush=True)
        if _OURS_ERRS["n"]:
            print("ours diag: turnlog/reuse/autopilot errors", _OURS_ERRS["n"], flush=True)
    except Exception as e:
        print("ours diag: report failed", repr(e), flush=True)
'''

# ---- the patch is appended to the harness after the winner's own patch is applied (cell 5)
i5 = next(i for i, c in enumerate(cells) if "harness patch applied successfully" in "".join(c["source"]))
c5 = src(i5)
anchor = "print('harness patch applied successfully')\n"
assert c5.count(anchor) == 1
c5 = c5.replace(anchor, anchor + (
    "# ours (yasunorim): A2 parts appended to the patched harness\n"
    f"os.environ['ARC_OURS_DOSSIER'] = '{dossier}'\n"
    f"os.environ['ARC_OURS_RETRY_TOK'] = '{opts.get('retry', '0')}'\n"
    f"os.environ['ARC_OURS_RETRY_WAIT_MIN'] = '{opts.get('retrywait', '0')}'\n"
    f"os.environ['ARC_OURS_REUSE'] = '{opts.get('reuse', '0')}'\n"
    f"os.environ['ARC_OURS_TURNLOG'] = '{opts.get('turnlog', '0')}'\n"
    f"os.environ['ARC_OURS_AUTOPILOT'] = '{int(opts.get('autopilot', '0'))}'\n"
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

# ---- streams=N (ours, part 5): more games decode at once. a2d2's serve.log: ~9 of 10 streams always busy, server
# queue 0, KV usage 0.4-0.77, and the harness gate kept games waiting ~25 game-hours in total. The gate, SGLang's
# running-request cap, the decode CUDA graphs and the mamba state cache (6 per stream, as the base) move together.
streams = int(opts.get("streams", "10"))
if streams != 10:
    i5s = next(i for i, c in enumerate(cells) if "'ARC3_MAX_ACTIVE_STREAMS': 10," in "".join(c["source"]))
    cells[i5s]["source"] = src(i5s).replace("'ARC3_MAX_ACTIVE_STREAMS': 10,", f"'ARC3_MAX_ACTIVE_STREAMS': {streams},  # ours")
    i13 = next(i for i, c in enumerate(cells) if "MAXREQ=10," in "".join(c["source"]))
    c13 = src(i13)
    for old, new in (("MAXREQ=10,", f"MAXREQ={streams},  # ours"), ("CUDAGRAPH_MAXBS=10,", f"CUDAGRAPH_MAXBS={streams},  # ours"),
                     ("MAMBA_CACHE=60,", f"MAMBA_CACHE={6 * streams},  # ours")):
        assert c13.count(old) == 1, old
        c13 = c13.replace(old, new)
    cells[i13]["source"] = c13

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
    print('ours diag: retry', getattr(_ta, '_OURS_RETRY_STATS', None))
    getattr(_ta, '_ours_report', lambda: None)()   # turnlog / reuse summaries (no-op when both are off)
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
print("wrote", name, "dossier", dossier, "passes", passes, "fast", opts.get("fast", "0"), "streams", opts.get("streams", "10"), "reuse", opts.get("reuse", "0"),
      "turnlog", opts.get("turnlog", "0"), "autopilot", opts.get("autopilot", "0"))

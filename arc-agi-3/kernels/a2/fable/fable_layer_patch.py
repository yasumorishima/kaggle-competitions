r"""Fable layer patch: harness-computed facts and pinned notes, applied to the Kaggle harness (tool_agent.py).

What it adds (everything off unless ARC3_FABLE_LAYER=1; every hook is wrapped so an error means "do nothing"):

  1. INDICATOR NOTES (one line per turn, only on a level that has already taken ARC3_FABLE_EVENTS_AFTER_TOKENS, default
     15000 generated tokens): when the last sequence of actions changed a small patch far from the moved objects (a legend,
     a tray mark, a counter), say which action did it; when the same patch changes BACK, say that a mark may have been lost.
  2. BUDGET ALERT (one line, when a step-budget bar runs low: <=25% of its cells or <=12 actions left at the current rate).
  3. PINNED LEVEL LOG (ARC3_FABLE_PIN_TOKENS, default 30000 tokens on one level; refreshed every
     ARC3_FABLE_PIN_REFRESH_TOKENS, default 40000): what each key did on this level (with the board state), typical moves,
     the indicator history, UNDO count, the budget, the digest of the cleared levels, and a five-line working protocol. It is
     appended to the history as a user/assistant pair and put back at the head of the history after every trim, until the
     level changes. No model call is made for it and no server seat is released.

Nothing is written by the model; the facts come from the recorded frames. Levels that are solved quickly never get the pinned
log and only get a note when a patch changed (after the early-gate tokens).

Apply to a pristine tool_agent.py; every anchor must match exactly once, otherwise 'skipped: ...' and the file is untouched.
The core (fable_layer_core.py, standard library only) is written next to tool_agent.py as inference/agent/fable_layer_core.py.

apply(agent_dir, core_src=None) returns 'applied', 'already applied' or 'skipped: <reason>'.
"""
from __future__ import annotations

import os

MARKER = "ARC3_FABLE_LAYER"

METHODS = r'''
    # ---- Fable layer (fable_layer_patch.py) -----------------------------------------------------------------------
    def _fl_on(self) -> bool:
        return _get_env_bool("ARC3_FABLE_LAYER", False)

    def _fl_core(self):
        mod = getattr(self, "_fl_mod", None)
        if mod is None:
            from inference.agent import fable_layer_core as mod  # noqa: PLC0415
            self._fl_mod = mod
        return mod

    def _fl_log(self, kind: str, **data) -> None:
        """Optional JSON-lines record of what the layer said (ARC3_FABLE_LAYER_LOG=<path>); never raises."""
        path = os.environ.get("ARC3_FABLE_LAYER_LOG", "").strip()
        if not path:
            return
        try:
            row = {"t": round(time.time(), 1), "kind": kind, "game": str(self._session_runtime_dir), **data}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _fl_level_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens) - int(getattr(self, "_tokens_at_level_start", 0) or 0))

    def _fl_reset_session(self) -> None:
        self._fl_session = self._session_runtime_dir
        self._fl_ledgers = {}
        self._fl_actions = {}
        self._fl_cleared = set()
        self._fl_done = 0
        self._fl_n_before = {}
        self._fl_pinned = None
        self._fl_pin_state = None
        self._fl_budget_cache = None
        self._fl_budget_alerted = None

    def _fl_sync(self, history_entries) -> dict:
        """Bring the per-level ledgers up to date with the recorded history; returns {level: Ledger}."""
        FL = self._fl_core()
        if getattr(self, "_fl_session", None) != self._session_runtime_dir:
            self._fl_reset_session()
        entries = list(history_entries or [])
        if len(entries) < self._fl_done:
            keep_pin = self._fl_pinned
            keep_state = self._fl_pin_state
            self._fl_reset_session()
            self._fl_pinned, self._fl_pin_state = keep_pin, keep_state
        self._fl_n_before = {lv: led.n_actions for lv, led in self._fl_ledgers.items()}
        for i in range(max(1, self._fl_done), len(entries)):
            prev, cur = entries[i - 1], entries[i]
            if prev.frame is None or cur.frame is None:
                continue
            pg, cg = prev.frame.grid, cur.frame.grid
            pl, cl = int(prev.frame.level), int(cur.frame.level)
            result = cur.result if isinstance(cur.result, dict) else {}
            name = result.get("action_display") or cur.action or ""
            if cl != pl:
                # the action that cleared level `pl`: its own frame already belongs to the next level
                led = self._fl_ledgers.setdefault(pl, FL.Ledger(pl))
                led.feed(name, result, pg, pg)
                self._fl_cleared.add(pl)
            else:
                led = self._fl_ledgers.setdefault(cl, FL.Ledger(cl))
                led.feed(name, result, pg, cg)
            acts = self._fl_actions.setdefault(pl, [])
            acts.append(FL.key_of(name))
            del acts[:-12]
        self._fl_done = len(entries)
        return self._fl_ledgers

    def _fl_attempt_frames(self, history_entries, level):
        """Grids of the current attempt on `level`: from the last RESET (or the level's first frame) to now."""
        entries = [e for e in (history_entries or []) if e.frame is not None]
        start = 0
        for i, e in enumerate(entries):
            if str(e.action or "").upper() == "RESET" or int(e.frame.level) != level:
                start = i
        frames = [e.frame.grid for e in entries[start:] if int(e.frame.level) == level]
        return frames

    def _fl_budget(self, history_entries, level):
        """Detected step-budget bar of the current attempt, recomputed at most every 6 new frames."""
        FL = self._fl_core()
        frames = self._fl_attempt_frames(history_entries, level)
        cache = getattr(self, "_fl_budget_cache", None)
        if cache is not None and cache[0] == level and 0 <= len(frames) - cache[1] < 6 and len(frames) >= cache[1]:
            return cache[2]
        b = FL.detect_budget(frames) if len(frames) >= 4 else None
        self._fl_budget_cache = (level, len(frames), b)
        return b

    def _fl_opener_lines(self, current_frame, history_entries) -> list:
        """Lines for the per-turn opener: indicator notes and a low-budget alert. [] whenever unsure."""
        if not self._fl_on() or current_frame is None:
            return []
        try:
            FL = self._fl_core()
            ledgers = self._fl_sync(history_entries)
            level = int(current_frame.level)
            led = ledgers.get(level)
            lines = []
            level_tokens = self._fl_level_tokens()
            after = _get_env_int("ARC3_FABLE_EVENTS_AFTER_TOKENS", 15000)
            if led is not None and after >= 0 and level_tokens >= after and not led.noisy:
                before_n = self._fl_n_before.get(level, 0)
                fresh = [ev for ev in led.events if ev["idx"] > before_n]
                for ev in fresh[-2:]:
                    lines.append(FL.Ledger.event_line(ev))
                    self._fl_log("note", level=level, level_tokens=level_tokens, text=lines[-1])
            if led is not None and _get_env_bool("ARC3_FABLE_BUDGET_ALERT", True) and led.n_actions >= 6:
                b = self._fl_budget(history_entries, level)
                text = FL.budget_alert(b, first_time=False) if b else ""
                if text:
                    last = getattr(self, "_fl_budget_alerted", None)
                    # one alert per attempt, then again only when about 10 actions have been used since
                    if last is None or last[0] != level or b["remaining"] <= last[1] - 10:
                        lines.append(text)
                        self._fl_budget_alerted = (level, b["remaining"])
                        self._fl_log("budget", level=level, level_tokens=level_tokens, text=text)
            return lines
        except Exception as exc:  # noqa: BLE001 - the layer is an extra; it must never stop a turn
            log.debug("fable layer opener failed: %s", exc)
            return []

    def _fl_remove_pinned(self, pinned) -> None:
        user_text = pinned[0].get("content")
        kept = []
        skip_reply = False
        for m in self._history_messages:
            if m.get("role") == "user" and m.get("content") == user_text:
                skip_reply = True
                continue
            if skip_reply and m.get("role") == "assistant" and m.get("content") == pinned[1].get("content"):
                skip_reply = False
                continue
            skip_reply = False
            kept.append(m)
        self._history_messages = kept

    def _fl_maybe_pin(self, current_frame, history_entries) -> None:
        """Append the harness-computed LEVEL LOG to the history once a level has taken long, and keep it pinned."""
        if not self._fl_on() or current_frame is None:
            return
        try:
            pin_at = _get_env_int("ARC3_FABLE_PIN_TOKENS", 30000)
            if pin_at <= 0:
                return
            refresh = max(1, _get_env_int("ARC3_FABLE_PIN_REFRESH_TOKENS", 40000))
            FL = self._fl_core()
            ledgers = self._fl_sync(history_entries)
            level = int(current_frame.level)
            led = ledgers.get(level)
            tokens = self._fl_level_tokens()
            if led is None or led.n_actions < 3 or tokens < pin_at:
                return
            state = self._fl_pin_state
            if state is not None and state[0] == level and tokens - state[1] < refresh:
                return
            history = self._history_messages
            if history and str(history[-1].get("role", "")).strip() == "user":
                return  # two user messages in a row would break the alternation; try again next turn
            b = self._fl_budget(history_entries, level)
            budget_text = FL.budget_alert(b, first_time=True) if b else ""
            earlier = []
            for lv in sorted(ledgers):
                if lv < level and lv in self._fl_cleared:
                    earlier.extend(ledgers[lv].digest_lines(self._fl_actions.get(lv)))
            text = FL.build_level_log(
                led,
                level_tokens=tokens,
                valid_actions=list(getattr(self, "_current_valid_actions", None) or []),
                budget_text=budget_text,
                earlier=earlier,
            )
            if self._fl_pinned:
                self._fl_remove_pinned(self._fl_pinned)
            user = {"role": "user", "content": text}
            reply = {"role": "assistant", "content": FL.ACK_TEXT}
            self._history_messages.append(user)
            self._history_messages.append(reply)
            self._fl_pinned = [user, reply]
            self._fl_pin_state = (level, tokens)
            log.info("fable layer: level log pinned (level %s, %s level tokens, %s chars)", level, tokens, len(text))
            self._fl_log("pin", level=level, level_tokens=tokens, actions=led.n_actions, text=text)
        except Exception as exc:  # noqa: BLE001
            log.debug("fable layer pin failed: %s", exc)

    def _fl_restore_pinned(self, history: list) -> list:
        """After a trim: put the pinned LEVEL LOG back at the head when the trim dropped it."""
        pinned = getattr(self, "_fl_pinned", None)
        if not pinned or not self._fl_on():
            return history
        try:
            text = pinned[0].get("content")
            if any(m.get("role") == "user" and m.get("content") == text for m in history):
                return history
            return [*pinned, *history]
        except Exception as exc:  # noqa: BLE001
            log.debug("fable layer restore failed: %s", exc)
            return history

    def _fl_forget_pin(self) -> None:
        self._fl_pinned = None
        self._fl_pin_state = None
        self._fl_budget_cache = None
        self._fl_budget_alerted = None

'''

EDITS = [
    # 1. methods, before the prompt builder (inside class ToolAgent)
    (
        "    def _build_user_prompt(\n        self,\n        action_num: int,\n",
        METHODS + "    def _build_user_prompt(\n        self,\n        action_num: int,\n",
    ),
    # 2. per-turn lines, right after the valid-actions line
    (
        "                f\"Valid actions right now: {_format_valid_action_line(valid_actions)}.\",\n            ]\n        )\n",
        "                f\"Valid actions right now: {_format_valid_action_line(valid_actions)}.\",\n            ]\n        )\n"
        "        lines.extend(self._fl_opener_lines(current_frame, history_entries))\n",
    ),
    # 3. a new level drops the pinned log
    (
        "            self._actions_at_level_start = _priority_action_count(summary)\n"
        "            self._tokens_at_level_start = self._session_generated_tokens\n",
        "            self._actions_at_level_start = _priority_action_count(summary)\n"
        "            self._tokens_at_level_start = self._session_generated_tokens\n"
        "            self._fl_forget_pin()\n",
    ),
    # 4. pin once the turn's own bookkeeping is done (the same place the rolling summary runs)
    (
        "        preserve_history = True\n        resuming_after_yield = bool(getattr(self, \"_resume_after_yield\", False))\n",
        "        self._fl_maybe_pin(current_frame, history_entries)\n"
        "        preserve_history = True\n        resuming_after_yield = bool(getattr(self, \"_resume_after_yield\", False))\n",
    ),
    # 5. every trim keeps the pinned log. Placed AFTER the "history was evicted" check on purpose: putting the log back
    # must not count as an eviction (an eviction hands the server seat to another game).
    (
        "        if len(history) != len(messages) - 1:\n"
        "            self._note_history_evicted()\n"
        "        return [system_message, *history]\n",
        "        if len(history) != len(messages) - 1:\n"
        "            self._note_history_evicted()\n"
        "        history = self._fl_restore_pinned(history)\n"
        "        return [system_message, *history]\n",
    ),
]


def apply(agent_dir: str, core_src: str | None = None) -> str:
    """Patch <agent_dir>/tool_agent.py and write <agent_dir>/fable_layer_core.py."""
    path = os.path.join(agent_dir, "tool_agent.py")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        return f"skipped: cannot read {path}: {exc}"
    if MARKER in text:
        return "already applied"
    for i, (old, _new) in enumerate(EDITS):
        n = text.count(old)
        if n != 1:
            return f"skipped: edit {i} anchor found {n} times (expected 1)"
    if core_src is None:
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            core_src = open(os.path.join(here, "fable_layer_core.py"), encoding="utf-8").read()
        except OSError as exc:
            return f"skipped: cannot read fable_layer_core.py: {exc}"
    try:
        compile(core_src, "fable_layer_core.py", "exec")
    except SyntaxError as exc:
        return f"skipped: core does not compile: {exc}"
    new_text = text
    for old, new in EDITS:
        new_text = new_text.replace(old, new, 1)
    try:
        compile(new_text, path, "exec")
    except SyntaxError as exc:
        return f"skipped: patched file does not compile: {exc}"
    open(os.path.join(agent_dir, "fable_layer_core.py"), "w", encoding="utf-8", newline="").write(core_src)
    open(path, "w", encoding="utf-8", newline="").write(new_text)
    return "applied"
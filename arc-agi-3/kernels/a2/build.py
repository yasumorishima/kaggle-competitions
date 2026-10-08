"""Build kernels/a2/main.ipynb = kernels/m2base/main.ipynb (the Milestone-2 winner by dfranzen, unchanged) + our A2 parts.

Ours (yasunorim):
  1. deadline: in a real rerun the per-game limit is 540 min - elapsed - 6 min at the moment the benchmark starts
     (the winner's 532 min + ~10 min server start-up leaves no slack against the 9-hour limit).
  2. level dossier (ARC_OURS_DOSSIER=1): when a level is completed, the exact action sequence of the final attempt
     is appended to the system prompt, which context trimming never removes. The winner's harness keeps only the
     (trimmed) chat history and the model's own functions across levels.

    python arc-agi-3/kernels/a2/build.py NAME [dossier=0|1] [passes=N]
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
if _ours_os.environ.get("ARC_OURS_DOSSIER", "0") == "1":
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
            self._system_prompt = (
                base + "\n\n## Solved levels of this game (exact action sequences of the winning attempts; kept "
                "even when old messages are trimmed)\n" + "\n".join(d) + "\nLater levels usually reuse the same "
                "mechanics with a bigger or changed layout: reuse what these sequences reveal instead of "
                "rediscovering the rules, and aim for fewer actions.")
        return _ours_orig_update(self)

    ToolAgent._update_summarized_knowledge_from_step_summary = _ours_update
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
cells[i17]["source"] = c17

cells[0]["source"] = ("# arc3 a2 (yasunorim)\n\nBase: the ARC-AGI-3 Milestone 2 solution by dfranzen "
                      "(https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution, built on Tufa Labs' Duck "
                      "harness), unchanged. Ours: a run deadline that respects the 9-hour limit, and a level dossier "
                      "(exact winning action sequences of solved levels kept in the system prompt). Built by "
                      "arc-agi-3/kernels/a2/build.py in github.com/yasumorishima/kaggle-competitions.\n")
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(os.path.join(HERE, "main.ipynb"), "w", encoding="utf-8"), indent=1)
meta = json.load(open(os.path.join(HERE, "..", "m2base", "kernel-metadata.json")))
meta.update(id=f"yasunorim/arc3-{name}", title=f"arc3 {name}")
json.dump(meta, open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=2)
print("wrote", name, "dossier", dossier, "passes", passes)

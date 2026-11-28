"""Build our GGA GPU notebook from the public notebook "GGA Agent 0.360+" (ozertuu/gga-agent-0-360, v10; its stack is
kozykappa's v2 with ozertuu's SGLang cache; Apache-2.0 public code). The JSON is edited as text; nothing is executed here.

Ours (v6gpu): the actor sees a smaller picture (long side 640 instead of 1024; the coach keeps 1024), to cut the reply
delay that the base's authors measured as the main cost on real-time games (off_line 0.825 at 0 s, 0.259 at 0.4 s).
Ours (v5gpu): the coach's ACT line plays when the actor is stuck. The 0.360 stack parses the coach's ACT line and drops
it; on the older stack this trigger (2 unchanged frames, a repeating actor, or a flat score) took alexchilton's notebook
from 0.326 to 0.350 (idea credited there to ozertuu and rishabhbafnaiiitd). No public notebook has it on the v2 stack.
usage: python -I patch.py public.ipynb main.ipynb
"""
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
REPL = [
    ("        self.mode = ''\n",
     "        self.mode = ''\n        self.coach_act = ''                 # [ours] coach ACT line, played once when the actor is stuck\n"),
    ("        if self.frozen >= UNSTICK_AFTER and self.frozen % 2 == 0:\n            self.mode = 'unstick'\n",
     "        stagnant = len(self.scores) >= 2 and self.scores[-1][1] == self.scores[-2][1]\n"
     "        if self.coach_act and (self.frozen >= 2 or self._looping() or stagnant):   # [ours]\n"
     "            self.mode = 'coach-act'\n            text, self.coach_act = self.coach_act, ''\n"
     "        elif self.frozen >= UNSTICK_AFTER and self.frozen % 2 == 0:\n            self.mode = 'unstick'\n"),
    ("            self.notes = '\\n'.join(notes)\n        sc = fields.get('SCORE', '').strip()\n",
     "            self.notes = '\\n'.join(notes)\n        act_line = (fields.get('ACT') or '').strip()   # [ours]\n"
     "        if act_line and parse_program(act_line):\n            self.coach_act = act_line\n"
     "        sc = fields.get('SCORE', '').strip()\n"),
    ("    def _view(self, frame):\n",
     "    def _view(self, frame, side_max=None):\n"),
    ("        im.thumbnail((IMG_SIDE, IMG_SIDE))\n        self.view_box = self.box\n",
     "        im.thumbnail((side_max or IMG_SIDE, side_max or IMG_SIDE))\n        self.view_box = self.box\n"),
    ("            self.mode = 'act'\n            img = self._view(frame)\n",
     "            self.mode = 'act'\n            img = self._view(frame, ACT_IMG_SIDE)   # [ours] smaller picture, shorter reply delay\n"),
    ("IMG_SIDE = 1024 ",
     "ACT_IMG_SIDE = 640       # [ours] the actor's picture (long side); the coach keeps IMG_SIDE\nIMG_SIDE = 1024 "),
]
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
src = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells]
for old, new in REPL:
    hits = [i for i, s in enumerate(src) if old in s]
    assert len(hits) == 1 and src[hits[0]].count(old) == 1, f"patch target not found exactly once: {old[:60]!r} {hits}"
    src[hits[0]] = src[hits[0]].replace(old, new)
for c, s in zip(cells, src): c["source"] = s
for c in nb["cells"]: c.pop("outputs", None) if c["cell_type"] != "code" else c.update(outputs=[], execution_count=None)
nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source":
    "Private build by yasunorim. Base: ozertuu/gga-agent-0-360 (v10; kozykappa v2 stack + SGLang cache), with credit. "
    "Our changes: the coach's ACT line plays when the actor is stuck; the actor sees a 640 px picture (lines marked [ours])."})
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("patched", len(REPL), "places")

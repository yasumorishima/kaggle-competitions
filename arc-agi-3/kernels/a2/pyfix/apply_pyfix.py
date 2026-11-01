"""ours (yasunorim, build.py pyfix=1): three small fixes to the harness's Python tool sandbox.

The ideas come from leoprovorov's public dataset "ARC-AGI-3 book of ideas" (entry 22, "A function that checks the
model's own simulator": "Small fixes to the Python tool ... history records had .step and .level, import time worked,
animation objects had .items()"; tool replies with an error 8.5% -> 7.3%; the book keeps the fixes and calls them free).
The book publishes no code; this file is our own implementation for the patched dfranzen / Tufa Labs Duck harness.
check_model (the other half of that entry) is not included.

Runs after the harness patch is applied. Every edit is anchored on an exact text that must occur once; if any anchor
is missing, nothing is written and the harness runs unchanged.
"""
import sys
from pathlib import Path

EDITS = [
    # 1) history entries expose .step and .level of their frame (models write h.step / h.level)
    ("        def __init__(self, *, action, frame, result=None):\n"
     "            self.action = action\n"
     "            self.frame = frame\n",
     "        def __init__(self, *, action, frame, result=None):\n"
     "            self.action = action\n"
     "            self.frame = frame\n"
     "            self.step = getattr(frame, \"step\", None)   # ours (pyfix)\n"
     "            self.level = getattr(frame, \"level\", None)  # ours (pyfix)\n"),
    # 2) `import time` no longer fails the whole snippet
    ("        \"string\",\n    }\n",
     "        \"string\",\n        \"time\",  # ours (pyfix): a stray `import time` cost the whole snippet\n    }\n"),
    # 3) the lazy animation records behave like dicts for .items() / .values()
    ("                def keys(self):\n"
     "                    return self._get().keys()\n",
     "                def keys(self):\n"
     "                    return self._get().keys()\n"
     "\n"
     "                def items(self):  # ours (pyfix)\n"
     "                    v = self._get()\n"
     "                    return v.items() if hasattr(v, \"items\") else enumerate(v)\n"
     "\n"
     "                def values(self):  # ours (pyfix)\n"
     "                    v = self._get()\n"
     "                    return v.values() if hasattr(v, \"values\") else iter(v)\n"),
]


def main(path):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, _ in EDITS:
        if s.count(old) != 1:
            print("pyfix: anchor not found once -> sandbox unchanged:", repr(old[:60]), s.count(old), flush=True)
            return 1
    for old, new in EDITS:
        s = s.replace(old, new)
    compile(s, str(p), "exec")
    p.write_text(s, encoding="utf-8")
    print("pyfix: 3 sandbox fixes applied to", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

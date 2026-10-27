"""Print the last stdout/stderr lines of a kernel log (Kaggle's JSON log) without the debugger noise (output post)."""
import json, os, sys
NOISE = ("Debugger warning", "frozen modules", "PYDEVD", "Debugging will proceed", "MissingIDFieldWarning", "validate(nb)")
for root, _, files in os.walk(sys.argv[1]):
    for f in sorted(files):
        if not f.endswith(".log"):
            continue
        raw = open(os.path.join(root, f), errors="replace").read()
        try:
            txt = "".join(x.get("data", "") for x in json.loads(raw))
        except Exception:
            txt = raw
        lines = [l for l in txt.splitlines() if l.strip() and not any(n in l for n in NOISE)]
        print(f"== {f}: {len(lines)} lines")
        for l in [l for l in lines if "ours " in l][:30] + lines[-40:]:
            print("  ", l[:300])

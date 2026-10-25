"""Print the informative lines of every *.log under a directory (used as `post` of an output request)."""
import os
import re
import sys

KEYS = re.compile(r"ready|fired up|uvicorn|error|traceback|exception|load weight|loading|took|memory|capture|killed|"
                  r"cuda graph|warmup|started|listening|startup|oom|avail", re.I)
for root, _, files in os.walk(sys.argv[1]):
    for f in sorted(files):
        if not f.endswith(".log"):
            continue
        lines = open(os.path.join(root, f), errors="replace").read().splitlines()
        print(f"== {f}: {len(lines)} lines")
        for ln in lines[:15]:
            print("  H", ln[:220])
        for i, ln in enumerate(lines[15:-40], 15):
            if KEYS.search(ln):
                print(f"  {i}", ln[:220])
        for ln in lines[-40:]:
            print("  T", ln[:220])

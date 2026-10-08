"""Build kernels/tpueval/main.py: embed agent configs for a paired local evaluation.

    python gemma4-agent/kernels/tpueval/build.py NAME=DIR [NAME=@sample ...] [--limit N] [--workers W] [--shard k/M]

DIR is a submission directory (relative to the repo root); @sample is the official sample_submission.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
args = sys.argv[1:]
opt = {"--limit": "1000", "--workers": "1", "--shard": "0/1"}
cfgs = {}
i = 0
while i < len(args):
    if args[i] in opt:
        opt[args[i]] = args[i + 1]
        i += 2
        continue
    name, d = args[i].split("=", 1)
    if d == "@sample":
        cfgs[name] = "@sample"
    else:
        base = os.path.join(ROOT, d)
        files = {}
        for dp, _, fs in os.walk(base):
            for f in fs:
                p = os.path.join(dp, f)
                files[os.path.relpath(p, base)] = open(p, encoding="utf-8").read()
        cfgs[name] = files
    i += 1
src = open(os.path.join(HERE, "template.py"), encoding="utf-8").read()
src = (src.replace("__CONFIGS__", repr(cfgs)).replace("__LIMIT__", opt["--limit"])
       .replace("__WORKERS__", opt["--workers"]).replace("__SHARD__", opt["--shard"]))
open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
print("configs", list(cfgs), opt)

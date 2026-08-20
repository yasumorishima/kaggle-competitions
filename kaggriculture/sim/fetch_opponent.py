#!/usr/bin/env python
"""Materialise a published Kaggriculture notebook as a sparring agent.

The ladder is played against other people's agents, so `starter` is a weak
yardstick. Public competition notebooks build their agent with a
`%%writefile submission.py` cell; this pulls the notebook through the Kaggle
API and writes that cell out as a runnable .py.

Nothing fetched here is committed: `opponents/` is git-ignored so other
people's code is never redistributed from this public repo.

Usage:
    python sim/fetch_opponent.py boatlee/v16-rc5-high-score-8c-4s-premium-market-lead
    -> opponents/boatlee__v16-rc5-high-score-8c-4s-premium-market-lead.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opponents")
WRITEFILE = re.compile(r"^%%writefile\s+(?:-a\s+)?(\S*?(?:submission|main)\.py)\s*$", re.M)


def extract(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        m = WRITEFILE.search(src)
        if not m:
            continue
        # Drop the magic line itself; keep everything after it.
        body = src[m.end():].lstrip("\n")
        chunks.append(body)
    return chunks


def main():
    if len(sys.argv) < 2:
        print("usage: fetch_opponent.py <owner/kernel-slug> [more...]")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    rc = 0
    for ref in sys.argv[1:]:
        tmp = tempfile.mkdtemp()
        subprocess.run(["kaggle", "kernels", "pull", ref, "-p", tmp],
                       check=True, env={**os.environ, "PYTHONUTF8": "1"})
        nb = next((os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".ipynb")), None)
        if not nb:
            print(f"NO_NOTEBOOK {ref}")
            rc = 1
            continue
        chunks = extract(nb)
        if not chunks:
            print(f"NO_WRITEFILE_CELL {ref}")
            rc = 1
            continue
        out = os.path.join(OUT_DIR, ref.replace("/", "__") + ".py")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))
        print(f"WROTE {out} ({sum(len(c) for c in chunks)} bytes, {len(chunks)} cell(s))")
    return rc


if __name__ == "__main__":
    sys.exit(main())

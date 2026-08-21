#!/usr/bin/env python
"""Materialise a published Kaggriculture notebook as a sparring agent.

The ladder is played against other people's agents, so `starter` is a weak
yardstick. Competition notebooks build their agent as `main.py`, but not in one
uniform way -- three mechanisms are in the wild:

1. a plain ``%%writefile main.py`` (or submission.py) cell;
2. a home-made cell magic (``%%agentfile``) that writes or appends the cell;
3. no file-writing cell at all: a base85+zlib blob decoded at run time into
   ``main.py`` (this is how the replay-reconstruction agents ship).

The first two are read straight out of the notebook JSON. The third needs the
notebook's own code to run, so `--exec` executes the code cells in a scratch
directory and keeps whatever `main.py` they leave behind. That flag is only
ever passed inside the GitHub Actions runner, which is a throwaway VM -- this
is other people's code and it does not run on the workstation.

Nothing fetched here is committed: `opponents/` is git-ignored so other
people's code is never redistributed from this public repo.

Usage:
    python sim/fetch_opponent.py [--exec] boatlee/v16-rc5-high-score-...
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opponents")
WRITEFILE = re.compile(r"^%%writefile\s+(?:-a\s+)?(\S*?(?:submission|main)\.py)\s*$", re.M)
AGENTFILE = re.compile(r"^%%agentfile\s*(\w*)\s*$", re.M)
# Cells that would run a whole episode: skipped when executing, they cost
# minutes and never contribute to the agent file.
HEAVY = re.compile(r"env\.run\(|make\(\s*[\"']kaggriculture|env\.render\(")


def code_cells(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            yield "".join(cell.get("source", []))


def from_writefile(cells):
    out = []
    for src in cells:
        m = WRITEFILE.search(src)
        if m:
            out.append(src[m.end():].lstrip("\n"))
    return out


def from_agentfile(cells):
    """A custom `%%agentfile [mode]` magic: bare = write, `a` = append."""
    out = []
    for src in cells:
        m = AGENTFILE.search(src)
        if not m:
            continue
        body = src[m.end():].lstrip("\n")
        if m.group(1) == "a" or out:
            out.append(body)
        else:
            out = [body]
    return out


def by_exec(cells, ref):
    """Run the notebook's code in a scratch dir and collect the main.py it writes."""
    work = tempfile.mkdtemp(prefix="nbexec_")
    script = os.path.join(work, "_run.py")
    body = []
    for src in cells:
        if HEAVY.search(src):
            continue
        lines = [ln for ln in src.splitlines()
                 if not ln.lstrip().startswith(("!", "%%", "%"))]
        body.append("try:\n    exec(" + repr("\n".join(lines)) + ", GLOBALS)\nexcept Exception as e:\n"
                    "    print('cell skipped:', type(e).__name__, e)\n")
    with open(script, "w", encoding="utf-8") as f:
        f.write("GLOBALS = {'__name__': '__main__'}\n" + "".join(body))
    subprocess.run([sys.executable, script], cwd=work, timeout=600,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    for cand in ("main.py", "submission.py"):
        for root, _dirs, files in os.walk(work):
            if cand in files:
                path = os.path.join(root, cand)
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                shutil.rmtree(work, ignore_errors=True)
                return [text]
    shutil.rmtree(work, ignore_errors=True)
    return []


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    allow_exec = "--exec" in sys.argv
    if not args:
        print("usage: fetch_opponent.py [--exec] <owner/kernel-slug> [more...]")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    rc = 0
    for ref in args:
        tmp = tempfile.mkdtemp()
        subprocess.run(["kaggle", "kernels", "pull", ref, "-p", tmp],
                       check=True, env={**os.environ, "PYTHONUTF8": "1"})
        nb = next((os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".ipynb")), None)
        if not nb:
            print(f"NO_NOTEBOOK {ref}")
            rc = 1
            continue
        cells = list(code_cells(nb))
        chunks = from_writefile(cells) or from_agentfile(cells)
        how = "writefile/agentfile"
        if not chunks and allow_exec:
            chunks = by_exec(cells, ref)
            how = "exec"
        if not chunks:
            print(f"NO_AGENT_FILE {ref} (try --exec)")
            rc = 1
            continue
        out = os.path.join(OUT_DIR, ref.replace("/", "__") + ".py")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))
        print(f"WROTE {out} ({sum(len(c) for c in chunks)} bytes, via {how})")
    return rc


if __name__ == "__main__":
    sys.exit(main())

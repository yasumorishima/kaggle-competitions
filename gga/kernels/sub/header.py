# Generalist Gaming Agent submission (yasunorim). The agent below is ours (gga/agent.py in our repo);
# this header finds the competition's harness and the footer hands the agent to the inference server.
import glob, os, sys
from pathlib import Path
_srv = sorted(glob.glob('/kaggle/input/**/kaggle_evaluation/gga/gga_inference_server.py', recursive=True))[0]
_root = str(Path(_srv).parents[2])
os.environ.setdefault('GGA_ROOT', _root)
for p in (_root,):
    if p not in sys.path:
        sys.path.insert(0, p)
print('harness', _root, flush=True)

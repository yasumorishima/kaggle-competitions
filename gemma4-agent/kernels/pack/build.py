"""Build kernels/pack/main.py: a CPU kernel that writes submission.zip from gemma4-agent/submission.

Kaggle uploads only the code file, so the submission files are embedded as strings.
A file `adapters/<name>/zero_lora.txt` holding a rank r is not shipped; instead the kernel writes
adapters/<name>/adapter_config.json + adapter_model.safetensors with train/make_zero_lora.py (a rank-r LoRA whose B
matrices are zero, for the LoRA feasibility probe).

    python gemma4-agent/kernels/pack/build.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "..", "submission")
GEN_SRC = open(os.path.join(HERE, "..", "..", "train", "make_zero_lora.py"), encoding="utf-8").read()

files, zero_loras = {}, {}
for root, _, names in os.walk(SUB):
    for n in sorted(names):
        p = os.path.join(root, n)
        rel = os.path.relpath(p, SUB)
        if n == "zero_lora.txt":
            zero_loras[os.path.dirname(rel)] = int(open(p).read().strip())
            continue
        files[rel] = open(p, encoding="utf-8").read()

src = '''"""Gemma 4 Developer Agent: write submission.zip (built by gemma4-agent/kernels/pack/build.py)."""
import os
import subprocess
import sys
import zipfile

FILES = ''' + json.dumps(files, indent=1, ensure_ascii=False) + '''
ZERO_LORAS = ''' + json.dumps(zero_loras) + '''
GEN_SRC = ''' + json.dumps(GEN_SRC) + '''

os.makedirs("/kaggle/working", exist_ok=True)
with zipfile.ZipFile("/kaggle/working/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name, text in sorted(FILES.items()):
        z.writestr(name, text)
        print(name, len(text))
    if ZERO_LORAS:
        open("/tmp/make_zero_lora.py", "w").write(GEN_SRC)
        for d, r in sorted(ZERO_LORAS.items()):
            out = os.path.join("/tmp/zl", d)
            subprocess.check_call([sys.executable, "/tmp/make_zero_lora.py", out, str(r)])
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                z.write(os.path.join(out, f), d + "/" + f)
                print(d + "/" + f, os.path.getsize(os.path.join(out, f)))
print("wrote /kaggle/working/submission.zip")
'''
open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
print("files:", sorted(files), "zero loras:", zero_loras)

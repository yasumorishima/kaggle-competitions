"""Build kernels/pack/main.py: a CPU kernel that writes submission.zip from gemma4-agent/submission.

Kaggle uploads only the code file, so the submission files are embedded as strings.

    python gemma4-agent/kernels/pack/build.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "..", "submission")

files = {}
for root, _, names in os.walk(SUB):
    for n in sorted(names):
        p = os.path.join(root, n)
        files[os.path.relpath(p, SUB)] = open(p, encoding="utf-8").read()

src = '''"""Gemma 4 Developer Agent: write submission.zip (built by gemma4-agent/kernels/pack/build.py)."""
import os
import zipfile

FILES = ''' + json.dumps(files, indent=1, ensure_ascii=False) + '''

os.makedirs("/kaggle/working", exist_ok=True)
with zipfile.ZipFile("/kaggle/working/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name, text in sorted(FILES.items()):
        z.writestr(name, text)
        print(name, len(text))
print("wrote /kaggle/working/submission.zip")
'''
open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
print("files:", sorted(files))

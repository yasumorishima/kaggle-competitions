"""Build kernels/llm/main.py = the explorer (kernels/explore/main.py, without its entry point) + agent_llm.py.

Kaggle uploads only the code file.   python arc-agi-3/kernels/llm/build.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ex = open(os.path.join(HERE, "..", "explore", "main.py"), encoding="utf-8").read()
tail = '\n\nif __name__ == "__main__":\n    main()\n'
assert ex.endswith(tail)
src = ex[:-len(tail)] + "\n" + open(os.path.join(HERE, "agent_llm.py"), encoding="utf-8").read()
open(os.path.join(HERE, "main.py"), "w", encoding="utf-8").write(src)
print("wrote main.py", len(src), "chars")

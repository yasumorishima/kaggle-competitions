import json, sys
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/pub_dwt/9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb"
nb = json.load(open(p, encoding="utf-8"))
for i in [1, 2, 5]:
    print(f"\n========== CELL {i} ==========")
    print("".join(nb["cells"][i].get("source", [])))
# also show how train_paths/test_paths are defined: search cell 4 head + tail relevant lines
print("\n========== CELL 4 (path-def lines) ==========")
src4 = "".join(nb["cells"][4].get("source", []))
for line in src4.splitlines():
    if any(k in line for k in ["train_paths", "test_paths", "glob", "_paths =", "CFG.", "horizontal_well", "typewell", "Parallel", "n_jobs"]):
        print(line)

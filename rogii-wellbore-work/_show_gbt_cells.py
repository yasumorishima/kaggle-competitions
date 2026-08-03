import json
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/pub_dwt/9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb"
nb = json.load(open(p, encoding="utf-8"))
for i in [15, 18]:
    print(f"\n========== CELL {i} ==========")
    print("".join(nb["cells"][i].get("source", [])))

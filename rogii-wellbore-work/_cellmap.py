import json
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/pub_dwt/9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb"
nb = json.load(open(p, encoding="utf-8"))
cells = nb["cells"]
print("total cells:", len(cells))
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    nlines = src.count("\n") + 1
    head = src.strip().splitlines()[0][:90] if src.strip() else "<empty>"
    # detect key markers
    marks = []
    for kw in ["build_well", "build_dataset", "train_df =", "test_df =", "features =",
               "LGBMRegressor", "lightgbm", "CatBoost", "Climber", "hill", "optuna",
               "Optuna", "submission", "to_csv", "sample_submission", "def main",
               "apply_pp", "sg_smooth", "Trainer", "joblib", "oof"]:
        if kw in src:
            marks.append(kw)
    print(f"[{i}] {c['cell_type']:8s} lines={nlines:4d} :: {head}")
    if marks:
        print(f"      MARKS: {marks}")

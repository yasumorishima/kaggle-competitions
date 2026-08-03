import json
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-ensemble/rogii-tcn-ensemble.ipynb"
nb = json.load(open(p, encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    marks = [kw for kw in ["test_preds = {}", "oof_preds = {}", "test_preds[f", "oof_preds[f",
                            "oof_preds['tcn']", "test_preds['tcn']",
                            "oof_preds = pd.DataFrame", "test_preds = pd.DataFrame",
                            "climber", "Climber(", "def train_lightgbm", "def train_catboost",
                            "[TCN]", "CFG.cv.split"] if kw in src]
    if marks:
        head = src.strip().splitlines()[0][:60] if src.strip() else ""
        print(f"[{i}] {c['cell_type'][:4]} :: {head}")
        print(f"     {marks}")

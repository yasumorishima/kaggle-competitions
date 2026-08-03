import json, pandas as pd, sys, re
d = sys.argv[1]
p = f"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/{d}/rogii-tcn-ensemble.log"
entries = json.load(open(p, encoding="utf-8"))
out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
err = "".join(e["data"] for e in entries if e.get("stream_name") == "stderr")
# print lines of interest
for line in out.splitlines():
    if re.search(r"\[TCN\]|SMOKE=|climber|Climber|best_score|hill|RMSE|score|weight|fold|submission|tcn|Optuna|trial|Best", line, re.I):
        print(line)
print("===== STDERR tail =====")
print(err[-700:])
sub = f"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/{d}/submission.csv"
try:
    df = pd.read_csv(sub)
    print("===== SUBMISSION =====", df.shape, list(df.columns), "NaN", int(df.iloc[:,1].isna().sum()))
    print(df.head(2).to_string())
except Exception as e:
    print("no submission:", e)

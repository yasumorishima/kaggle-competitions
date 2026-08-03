import json, pandas as pd, sys
d = sys.argv[1]
p = f"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/{d}/rogii-tcn-hybrid.log"
entries = json.load(open(p, encoding="utf-8"))
out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
err = "".join(e["data"] for e in entries if e.get("stream_name") == "stderr")
print("===== STDOUT (tail) =====")
print(out[-3500:])
print("===== STDERR (tail) =====")
print(err[-900:])
sub = f"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/{d}/submission.csv"
df = pd.read_csv(sub)
print("===== SUBMISSION =====")
print("shape", df.shape, "cols", list(df.columns), "NaN", int(df.iloc[:,1].isna().sum()))
print(df.head(3).to_string())
print("tvt min %.2f max %.2f mean %.2f" % (df.iloc[:,1].min(), df.iloc[:,1].max(), df.iloc[:,1].mean()))

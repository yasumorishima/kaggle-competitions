import json, pandas as pd
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/tcn_full_out/rogii-tcn-sequence.log"
entries = json.load(open(p, encoding="utf-8"))
out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
err = "".join(e["data"] for e in entries if e.get("stream_name") == "stderr")
print("===== STDOUT =====")
print(out[-3500:])
print("===== STDERR (tail) =====")
print(err[-800:])
sub = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/tcn_full_out/submission.csv"
df = pd.read_csv(sub)
print("===== SUBMISSION =====")
print("shape", df.shape, "cols", list(df.columns), "NaN", int(df.iloc[:,1].isna().sum()))
print("tvt stats: min %.2f max %.2f mean %.2f" % (df.iloc[:,1].min(), df.iloc[:,1].max(), df.iloc[:,1].mean()))

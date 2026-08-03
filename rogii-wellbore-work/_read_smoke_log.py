import json, pandas as pd
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/tcn_smoke_out/rogii-tcn-sequence.log"
entries = json.load(open(p, encoding="utf-8"))
out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
err = "".join(e["data"] for e in entries if e.get("stream_name") == "stderr")
print("===== STDOUT =====")
print(out[-4000:])
print("===== STDERR (tail) =====")
print(err[-1500:])
sub = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/tcn_smoke_out/submission.csv"
df = pd.read_csv(sub)
print("===== SUBMISSION =====")
print("shape", df.shape, "cols", list(df.columns))
print(df.head(3).to_string())
print("nan in pred col:", df.iloc[:, 1].isna().sum())

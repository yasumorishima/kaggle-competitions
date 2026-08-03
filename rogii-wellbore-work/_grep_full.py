import json, re
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/tcn_full_out/rogii-tcn-sequence.log"
entries = json.load(open(p, encoding="utf-8"))
out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
for line in out.splitlines():
    if re.search(r"wells|fold\d toe RMSE|CV toe|train=|valid=|DEVICE|GPU|SMOKE|first toe|boundary", line):
        print(line)

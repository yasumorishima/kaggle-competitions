import json, re
p = r"C:/Users/fw_ya/Desktop/Claude_code/kaggle-competitions/rogii-wellbore-work/ens_full/rogii-tcn-ensemble.log"
entries = json.load(open(p, encoding="utf-8"))
out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
lines = out.splitlines()
# print any line mentioning hill, climb, ensemble, optuna, best, weight, overall_scores, trial, postproc, final, params
pat = re.compile(r"hill|climb|ensemble|optuna|best|weight|overall|trial|post|final|params|alpha|tau|w_pf|score", re.I)
for i, line in enumerate(lines):
    if pat.search(line):
        print(line)
print("===== LAST 25 LINES OF STDOUT =====")
for line in lines[-25:]:
    print(line)

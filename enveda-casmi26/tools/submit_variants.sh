#!/bin/bash
# Submit enveda b3 variants one after another (each waits for the previous result file).
cd /home/user/kaggle-competitions || exit 1
B=claude/kaggriculture-cloud-check-hbq2ii
K=enveda-casmi26/kernels/b1/main.py
setk() {  # pow n_keep top_k radius size count
python3 - "$@" <<'PY'
import re, sys
p = "enveda-casmi26/kernels/b1/main.py"; s = open(p).read()
pw, nk, tk, r, sz, c = sys.argv[1:]
s = re.sub(r"^POW = [0-9.]+", f"POW = {pw}", s, flags=re.M)
s = re.sub(r"^N_KEEP = \d+", f"N_KEEP = {nk}", s, flags=re.M)
s = re.sub(r"^TOP_K = \d+", f"TOP_K = {tk}", s, flags=re.M)
s = re.sub(r"^FP_RADIUS, FP_SIZE, FP_COUNT = [^#]*#", f"FP_RADIUS, FP_SIZE, FP_COUNT = {r}, {sz}, {c}   #", s, flags=re.M)
open(p, "w").write(s)
PY
}
push() { git pull -q --rebase origin $B; git push -q origin $B || { sleep 5; git pull -q --rebase origin $B; git push -q origin $B; }; }
run() {  # id message pow nk tk r sz c
  id=$1; msg=$2; shift 2
  setk "$@"
  python3 -c "
import json,sys
json.dump({'id':'$id','memo':'$msg','action':'kernel','dir':'kernels/b1','submit':'submit','message':'$msg'},open('enveda-casmi26/requests/kaggle.json','w'),indent=1)"
  git add $K enveda-casmi26/requests/kaggle.json
  git commit -q -m "enveda-casmi26: submit $id ($msg)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GNS5mWzU1hvsZ2JVZ3rsPV"
  push
  for i in $(seq 1 60); do
    git fetch -q origin $B
    f=$(git grep -l "request: $id\$" origin/$B -- enveda-casmi26/requests/results 2>/dev/null | head -1)
    if [ -n "$f" ]; then git pull -q --rebase origin $B; echo "== $id"; git show "$f" | sed -n 2,3p; git show "$f" | grep -m1 "$id\|PENDING" ; break; fi
    sleep 30
  done
}
run b4a-1 "b4a: r3 counts, POW 2, 400 analogs, top-5 sum" 2.0 400 5 3 4096 True
run b4b-1 "b4b: r2 bits, POW 2, 400 analogs, top-5 sum" 2.0 400 5 2 2048 False
run b4c-1 "b4c: r3 counts, POW 3, 400 analogs, top-3 sum" 3.0 400 3 3 4096 True
run b4d-1 "b4d: r2 bits, POW 2, 400 analogs, top-10 sum" 2.0 400 10 2 2048 False
setk 2.0 400 3 3 4096 True
git add $K && git commit -q -m "enveda-casmi26: kernel back to b3 settings after the b4 LB probes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GNS5mWzU1hvsZ2JVZ3rsPV" && push
echo "all done"

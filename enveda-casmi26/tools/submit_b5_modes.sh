#!/bin/bash
# Submit b5 fusion-mode variants one after another (each waits for the previous result file).
cd /home/user/kaggle-competitions || exit 1
B=claude/kaggriculture-cloud-check-hbq2ii
push() { git pull -q --rebase origin $B; git push -q origin $B || { sleep 5; git pull -q --rebase origin $B; git push -q origin $B; }; }
wait_for() {
  for i in $(seq 1 240); do
    git fetch -q origin $B
    f=$(git grep -l "request: $1\$" origin/$B -- enveda-casmi26/requests/results 2>/dev/null | head -1)
    if [ -n "$f" ]; then git pull -q --rebase origin $B; echo "== $1"; git show "$f" | sed -n 1,4p; return; fi
    sleep 30
  done
  echo "== $1 TIMEOUT"
}
run() {  # id mode memo
  python3 enveda-casmi26/kernels/b5/build.py "$2" >/dev/null
  python3 -c "
import json
json.dump({'id':'$1','memo':'$3','action':'kernel','dir':'kernels/b5','submit':'submit','message':'$1: b5 with fusion $2'},open('enveda-casmi26/requests/kaggle.json','w'),indent=1)"
  git add enveda-casmi26/kernels/b5/main.py enveda-casmi26/requests/kaggle.json
  git commit -q -m "enveda-casmi26: submit $1 (b5 fusion $2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GNS5mWzU1hvsZ2JVZ3rsPV"
  push
  wait_for "$1"
}
[ -n "$FIRST_WAIT" ] && wait_for "$FIRST_WAIT"
for spec in "$@"; do IFS='|' read -r id mode memo <<< "$spec"; run "$id" "$mode" "$memo"; done
python3 enveda-casmi26/kernels/b5/build.py >/dev/null
git add enveda-casmi26/kernels/b5/main.py && git commit -q -m "enveda-casmi26: b5 back to the default fusion after the LB probes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GNS5mWzU1hvsZ2JVZ3rsPV" && push
echo "all done"

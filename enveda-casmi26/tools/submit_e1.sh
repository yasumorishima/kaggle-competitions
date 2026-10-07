#!/bin/bash
# Submit e1fuse variants one after another (each waits for the previous result file).
#   tools/submit_e1.sh "ID|NAME|FP2_LAM|extra build args|memo" ...
cd /home/user/kaggle-competitions || exit 1
B=claude/kaggriculture-cloud-check-hbq2ii
push() { git pull -q --rebase origin $B; git push -q origin $B || { sleep 5; git pull -q --rebase origin $B; git push -q origin $B; }; }
wait_for() {
  for i in $(seq 1 240); do
    git fetch -q origin $B
    f=$(git grep -l "request: $1\$" origin/$B -- enveda-casmi26/requests/results 2>/dev/null | head -1)
    if [ -n "$f" ]; then git pull -q --rebase origin $B; echo "== $1"; git show "$f" | sed -n 1,4p; git show "$f" | grep -iE "error|submitted|Successfully" | head -5; return; fi
    sleep 30
  done
  echo "== $1 TIMEOUT"
}
for spec in "$@"; do
  IFS='|' read -r RID NAME LAM EXTRA MEMO <<< "$spec"
  python3 enveda-casmi26/kernels/e1fuse/build.py "$NAME" "$LAM" $EXTRA >/dev/null || { echo "build failed $RID"; exit 1; }
  RID="$RID" MEMO="$MEMO" python3 -c "
import json, os
e = os.environ
json.dump({'id': e['RID'], 'memo': e['MEMO'], 'action': 'kernel', 'dir': 'kernels/e1fuse', 'submit': 'submit',
           'message': e['RID'] + ': ' + e['MEMO'][:150]}, open('enveda-casmi26/requests/kaggle.json', 'w'), indent=1)" || { echo "request not written for $RID"; exit 1; }
  git add enveda-casmi26/kernels/e1fuse enveda-casmi26/requests/kaggle.json enveda-casmi26/tools/submit_e1.sh
  git commit -q -m "enveda-casmi26: submit $RID (e1fuse $NAME FP2_LAM $LAM $EXTRA)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GNS5mWzU1hvsZ2JVZ3rsPV"
  push
  wait_for "$RID"
done

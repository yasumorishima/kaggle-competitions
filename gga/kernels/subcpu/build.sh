#!/bin/sh
# main.py = header + our agent + footer (shared with kernels/sub)
set -e
D=$(dirname "$0")
cat "$D/../sub/header.py" "$D/../../agent.py" "$D/../sub/footer.py" > "$D/main.py"

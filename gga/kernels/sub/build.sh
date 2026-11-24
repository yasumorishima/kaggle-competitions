#!/bin/sh
# main.py = header + our agent + footer
set -e
D=$(dirname "$0")
cat "$D/header.py" "$D/../../agent.py" "$D/footer.py" > "$D/main.py"

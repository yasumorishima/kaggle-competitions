#!/bin/sh
# main.ipynb = the public notebook ozertuu/gga-agent-0-360 (pulled here, never stored in this repo) + our patch.py.
set -e
D=$(dirname "$0"); T=$(mktemp -d)
kaggle kernels pull ozertuu/gga-agent-0-360 -p "$T" >/dev/null
ls "$T"
python -I "$D/patch.py" "$T"/*.ipynb "$D/main.ipynb"

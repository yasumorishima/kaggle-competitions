#!/bin/bash
# Setup Kaggle and SIGNATE credentials from env vars (for GitHub Actions or new device)
#
# GitHub Secrets required in kaggle-competitions repo:
#   KAGGLE_USERNAME  - Kaggle username
#   KAGGLE_KEY       - Kaggle API key
#   SIGNATE_TOKEN_B64 - SIGNATE token Base64 encoded: base64 < ~/.signate/signate.json
#
# Local usage (new device):
#   export KAGGLE_USERNAME=yasunorim
#   export KAGGLE_KEY=<your-key>
#   export SIGNATE_TOKEN_B64=$(base64 < ~/.signate/signate.json)
#   bash scripts/setup-credentials.sh

set -e

# Kaggle
KAGGLE_DIR="$HOME/.kaggle"
mkdir -p "$KAGGLE_DIR"

if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > "$KAGGLE_DIR/kaggle.json"
    chmod 600 "$KAGGLE_DIR/kaggle.json"
    echo "Kaggle credentials configured"
elif [ -f "$KAGGLE_DIR/kaggle.json" ]; then
    echo "Kaggle credentials already exist"
else
    echo "WARNING: No Kaggle credentials. Set KAGGLE_USERNAME + KAGGLE_KEY"
fi

# SIGNATE (Base64 encoded, same as signate-comp repo pattern)
SIGNATE_DIR="$HOME/.signate"
mkdir -p "$SIGNATE_DIR"

if [ -n "$SIGNATE_TOKEN_B64" ]; then
    echo "$SIGNATE_TOKEN_B64" | base64 -d > "$SIGNATE_DIR/signate.json"
    chmod 600 "$SIGNATE_DIR/signate.json"
    echo "SIGNATE credentials configured"
elif [ -f "$SIGNATE_DIR/signate.json" ]; then
    echo "SIGNATE credentials already exist"
else
    echo "WARNING: No SIGNATE credentials. Set SIGNATE_TOKEN_B64"
fi

echo "Done."

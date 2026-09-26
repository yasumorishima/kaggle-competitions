#!/bin/sh
# Wheels for the no-internet kernel (Kaggle runs Python 3.12). rdkit is pinned to
# the metric's version so candidate keys match the scorer's.
set -e
cd "$(dirname "$0")"
pip download -q --no-deps --only-binary=:all: --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  -d . "rdkit==2026.3.3" "ms_entropy==1.5.2"
ls -la

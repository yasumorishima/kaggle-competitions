#!/bin/sh
# Wheels for the no-internet kernels. Kaggle moved from Python 3.12 to 3.13 (2026-10-06);
# both sets are kept and each kernel installs the ones matching its interpreter. rdkit is
# pinned to the metric's version so candidate keys match the scorer's.
set -e
cd "$(dirname "$0")"
for v in 3.12 3.13; do
  pip download -q --no-deps --only-binary=:all: --python-version $v \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
    -d . "rdkit==2026.3.3" "ms_entropy==1.5.2"
done
ls -la

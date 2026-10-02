#!/bin/sh
# vLLM and every dependency as wheels, for the no-internet GPU kernel (Kaggle runs Python 3.12;
# its image has torch 2.10+cu128 and no vLLM; GPU kernels cannot have internet).
# The kernel installs them with: pip install --no-index --find-links <this dataset> vllm
set -e
cd "$(dirname "$0")"
# the hosted runner has ~14GB free; torch + CUDA libraries need room
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL || true
df -h . | tail -1
pip download -q --only-binary=:all: --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  --platform manylinux_2_31_x86_64 --platform manylinux_2_34_x86_64 --platform linux_x86_64 --platform any \
  -d . "vllm==0.30.0"
ls -la | head -200
du -sh .
df -h . | tail -1

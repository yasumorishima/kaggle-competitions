#!/bin/sh
# vLLM and every dependency as wheels, for the no-internet GPU kernel (Kaggle runs Python 3.12;
# its image has torch 2.10+cu128 and no vLLM; GPU kernels cannot have internet).
# The kernel installs them with: pip install --no-index --find-links <this dataset> vllm
set -e
cd "$(dirname "$0")"
# the hosted runner has ~14GB free; torch + CUDA libraries need room
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL || true
df -h . | tail -1
# the runner is x86_64 Linux with Python 3.12 (setup-python), the same ABI as Kaggle: resolve natively
python -V
pip download -q --only-binary=:all: -d . "vllm==0.30.0"
ls -la | head -200
du -sh .
df -h . | tail -1

#!/bin/sh
# Wheels the GGA inference server needs on images without grpc_tools (the TPU image): grpcio-tools and grpcio
# for Python 3.10-3.13, in two lines (1.62.3 for protobuf 4, 1.71.0 for protobuf 5+), and setuptools (pkg_resources).
# Fetched from PyPI at build time; nothing binary lives in git.
set -e
D=$(dirname "$0")
for py in 310 311 312 313; do
  for v in 1.62.3 1.71.0; do
    for pk in grpcio-tools grpcio; do
      pip download -q --no-deps --only-binary=:all: --python-version $py \
        --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 "$pk==$v" -d "$D" || true
    done
  done
done
pip download -q --no-deps --only-binary=:all: setuptools -d "$D"
ls "$D"

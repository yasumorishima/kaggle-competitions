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
# 11-25: the TPU image (Python 3.12, grpc 1.85.1, protobuf 7.37) also lacks polars, and maybe pyarrow, pydantic and
# opentelemetry, which kaggle_evaluation imports. grpcio-tools 1.85.1 matches its grpc (1.71 writes protobuf-5
# gencode, two majors behind protobuf 7). The kernel header installs only the modules that fail to import.
for py in 312 313; do
  pip download -q --no-deps --only-binary=:all: --python-version $py \
    --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 "grpcio-tools==1.85.1" -d "$D" || true
  pip download -q --only-binary=:all: --python-version $py \
    --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
    polars pyarrow pydantic opentelemetry-api opentelemetry-sdk -d "$D" || true
done
ls "$D"

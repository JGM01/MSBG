#!/usr/bin/env bash
# Builds build/smoothertest: runs the REAL applyChannelPdeFast in its live
# 8-color in-place path (laplTyp 1/4) for 4 iterations over a fixed polynomial
# field and prints the resulting full level-0 field. The Rust side
# (msbg-rs/tests/difftest_smoother.rs) compares against it within tolerance.
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  smoothertest.cpp -o build/smoothertest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/smoothertest"

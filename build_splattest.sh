#!/usr/bin/env bash
# Builds build/splattest: runs the REAL msbg_test_sparse pipeline (PLY load,
# placement, 8-color splat, finalize, mean-curvature smoothing) and prints the
# active count + two full level-0 u16 fields (after finalize, after smoothing).
# The Rust side (msbg-rs/tests/difftest_splat.rs) compares against it.
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  splattest.cpp -o build/splattest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/splattest"

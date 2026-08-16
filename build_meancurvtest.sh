#!/usr/bin/env bash
# Builds build/meancurvtest: runs the REAL applyChannelPdeFast (laplTyp 1/4/2)
# for one iteration over a fixed polynomial field and prints the resulting
# interior block. The Rust side (msbg-rs/tests/difftest_stencil.rs) compares
# against it within tolerance.
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

# meancurvtest links the library, so make sure it exists first.
[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  meancurvtest.cpp -o build/meancurvtest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/meancurvtest"

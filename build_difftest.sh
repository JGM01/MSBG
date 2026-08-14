#!/usr/bin/env bash
# Builds build/difftest: a fixed write/read script over SBG::SparseGrid<float>
# that prints an FNV-1a hash. The Rust side (msbg-rs/tests/difftest_cpp.rs)
# compares against this hash.
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

# difftest links the library, so make sure it exists first.
[ -f build/libmsbg.a ] || ./build_lib.sh

# -I.          resolves the `src/...` includes in difftest.cpp
# -Iexternal   resolves <vectorclass/vectorclass.h>
g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  difftest.cpp -o build/difftest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/difftest"

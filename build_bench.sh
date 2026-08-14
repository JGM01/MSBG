#!/usr/bin/env bash
# Builds build/bench_executable against libmsbg.a (blockpool + halo + PDE
# smoothing). Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

# The benchmark links the library, so make sure it exists first.
[ -f build/libmsbg.a ] || ./build_lib.sh

# -I.          resolves the `src/...` includes in benchmark.cpp
# -Iexternal   resolves <vectorclass/vectorclass.h>
g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  benchmark.cpp -o build/bench_executable \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/bench_executable"
echo "Run (small): MSBG_BENCH_SCALE=small ./build/bench_executable"
echo "Run (full):  ./build/bench_executable"

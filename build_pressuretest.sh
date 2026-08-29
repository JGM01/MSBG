#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -g -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  pressuretest.cpp -o build/pressuretest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/pressuretest"

#!/usr/bin/env bash
# Builds build/velocitytest: times SparseGrid<Vec3Float>::interpolateVec3Float
# over N random positions. The Rust side (msbg-rs/benches/velocity_bench.rs)
# runs the same field/positions through its Vec3 gather.
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  velocitytest.cpp -o build/velocitytest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/velocitytest"

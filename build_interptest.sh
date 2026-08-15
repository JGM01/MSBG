#!/usr/bin/env bash
# Builds build/interptest: prints sampled value/gradient/Hessian for a fixed
# quadratic field. The Rust side (msbg-rs/tests/difftest_interp.rs) compares
# against it within tolerance.
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

# interptest links the library, so make sure it exists first.
[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  interptest.cpp -o build/interptest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/interptest"

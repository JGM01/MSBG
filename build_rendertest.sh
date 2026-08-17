#!/usr/bin/env bash
# Builds build/rendertest: the step-10 render harness for the msbg-render
# comparison. Runs the REAL demo pipeline (PLY load, placement, 8-color splat,
# finalize, mean-curvature smoothing) then the REAL render entry points
# (MultiresSparseGrid::getSlices2D + RDR::RaymarchRenderer::render).
#
# Requires the MSBG nix-shell (see shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

[ -f build/libmsbg.a ] || ./build_lib.sh

g++ -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal \
  rendertest.cpp -o build/rendertest \
  -Lbuild -lmsbg -ltbb -lpng -ljpeg -lz -lm

echo "Built build/rendertest"

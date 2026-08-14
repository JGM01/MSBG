#!/usr/bin/env bash
# Profile one C++ benchmark scenario and emit:
#   build/profile.txt    (perf report top-N self-time, machine-readable)
#   build/flamegraph.svg (for humans)
# Requires the MSBG nix-shell (shell.nix).
set -euo pipefail
cd "$(dirname "$0")"

scenario="${1:-}"
if [ -z "$scenario" ]; then
  echo "usage: $0 <hot|contention|cold|halo|laplacian>" >&2
  exit 1
fi

[ -x build/bench_executable ] || ./build_bench.sh

mkdir -p build

# C++ has frame pointers (nixpkgs hardening), so plain -g unwinding is fine.
perf record -g -F 999 -o build/profile.data -- ./build/bench_executable small "$scenario"

# -g none    flat table, no call-tree noise
# --no-children  self time (flamegraph "width"), not inclusive
# --percent-limit 2  drop functions under 2%
perf report -i build/profile.data --stdio -g none --no-children \
  --percent-limit 2 --show-total-period > build/profile.txt

perf script -i build/profile.data | inferno-collapse-perf | inferno-flamegraph > build/flamegraph.svg

echo "Wrote build/profile.txt and build/flamegraph.svg"
echo "--- profile.txt ---"
cat build/profile.txt

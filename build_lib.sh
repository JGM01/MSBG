#!/usr/bin/env bash
# Builds build/libmsbg.a from src/ using the upstream makefile.
#
# Requires the MSBG nix-shell (see shell.nix). Plain gcc/g++ work because
# `glibc` is intentionally NOT in shell.nix (it broke the C++ include order).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build
cd build
# `mk` has a `#!/bin/bash` shebang, which does not exist on NixOS — invoke via bash.
bash ../mk libmsbg.a

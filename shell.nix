{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # build
    gcc
    gnumake
    ispc
    llvmPackages.openmp
    tbb
    libpng
    libjpeg
    zlib
    # debugging
    gdb
    strace
    perf
    pahole
    llvmPackages.llvm   # llvm-mca, llvm-objdump, llvm-dwarfdump
    valgrind
    rr
    heaptrack
  ]; # NOTE: do NOT add `glibc` here. It injects `-isystem glibc-dev/include`
     # ahead of the C++ headers and breaks `#include_next <stdlib.h>`. gcc
     # already provides glibc correctly on its own.

  shellHook = ''
    # Prevent Nix from stripping -march=native (Required for AVX/SSE4.1)
    export NIX_ENFORCE_NO_NATIVE=0

    if [ -f ./mk ]; then
      chmod +x ./mk
    fi
    export PATH="$PWD:$PATH"

    # rr needs the kernel to allow perf events; raise the limit if too strict.
    if [ "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 0)" -gt 1 ]; then
      echo "note: rr needs 'sudo sysctl kernel.perf_event_paranoid=1'"
    fi

    echo "C++ Benchmark environment loaded for NixOS!"
  '';
}

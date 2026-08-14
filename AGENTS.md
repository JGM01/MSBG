# AGENTS.md — MSBG (C++) benchmarking notes

This is the reference C++ implementation of *Multiresolution Sparse Block Grids*
(the SIGGRAPH 2025 phase-field FLIP paper). It is being used as the baseline for
the Rust port in `../msbg-rs`. The files you will care about for benchmarking are:

- `benchmark.cpp` — the benchmark harness (blockpool + halo gather + PDE smoothing)
- `difftest.cpp` — differential hash test vs the Rust port
- `shell.nix` — the build/run environment (nix-shell)
- `build_lib.sh` — builds `libmsbg.a` (via the upstream makefile)
- `build_bench.sh` — compiles `benchmark.cpp` against `libmsbg.a`
- `build_difftest.sh` — compiles `difftest.cpp` against `libmsbg.a`
- `build/` — build artifacts (`libmsbg.a`, `bench_executable`, `difftest`)

---

## 1. Environment

Everything runs inside `nix-shell` from the repo root:

```bash
cd /home/jacob/programming/MSBG
nix-shell
```

`shell.nix` provides `gcc`, `gnumake`, `ispc`, `llvmPackages.openmp`, `tbb`,
`libpng`, `libjpeg`, `zlib`, `gh` (deliberately **no** `glibc` — see §2). Its
`shellHook` sets `NIX_ENFORCE_NO_NATIVE=0` (required for `-march=native`).

**Platform: Linux x86-64 only.** The C++ sources are x86-only — `external/vectorclass`
is a header-only SIMD library built on x86 intrinsics (`__m128`/`__m256`/`__m512`),
and the build assumes `-m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX` plus `-march=native`.
It does **not** build or run on macOS, especially Apple Silicon (arm64). The Rust
port (`../msbg-rs`) has no such restriction and runs on macOS arm64 via
`portable_simd`.

## 2. IMPORTANT: keep `glibc` OUT of shell.nix

The nixpkgs `gcc` wrapper places any `-isystem <glibc-dev>/include` (which
`glibc` contributes when it is a `buildInput`) *before* the C++ include dirs.
That breaks `#include_next <stdlib.h>` in `<cstdlib>`:

```
include/c++/.../cstdlib:79: fatal error: stdlib.h: No such file or directory
```

This affects **both** the makefile library build and direct `g++` invocations.
The fix is simply to **not list `glibc` in `shell.nix`** (gcc already provides
glibc correctly on its own). With that, plain `g++`/`gcc` work — no unwrapped
compiler, no `-B`/`-L` reconstruction needed anywhere.

## 3. Building the library (`libmsbg.a`)

```bash
cd /home/jacob/programming/MSBG
nix-shell
./build_lib.sh
```

`build_lib.sh` runs `../mk libmsbg.a`, which compiles the library objects with
the upstream makefile and archives them into `build/libmsbg.a`. It rebuilds
incrementally, so re-run it after changing `src/*.cpp`.

## 4. Building the benchmark

```bash
cd /home/jacob/programming/MSBG
nix-shell
./build_bench.sh
```

`build_bench.sh` compiles `benchmark.cpp` → `build/bench_executable` and links
`-lmsbg -ltbb -lpng -ljpeg -lz -lm` (it runs `build_lib.sh` first if
`build/libmsbg.a` is missing).

### 4.1 Differential test (`difftest.cpp`)

`difftest.cpp` runs a fixed write/read script over a real
`SBG::SparseGrid<float>` and prints an FNV-1a hash of every voxel's bit
pattern. The Rust side (`msbg-rs/tests/difftest_cpp.rs`) runs the same script
and compares hashes.

```bash
./build_difftest.sh
./build/difftest
```

## 5. Running the benchmark

Two sizes are supported via `MSBG_BENCH_SCALE` (default = full):

```bash
# small (fast, for compile/debug cycles)
MSBG_BENCH_SCALE=small ./build/bench_executable

# full (stress; ~100k active blocks; watch RAM — 7 GB machines OOM at the 100k e2e case)
./build/bench_executable
```

The binary also accepts `small` as argv[1] instead of the env var. (The g++
wrapper embeds an rpath for the nix-store libs, so no `LD_LIBRARY_PATH` is
needed.)

## 6. What the benchmark measures (scenarios)

- **A `bench_hot_path_scaling`** — single-thread `BlockPool::allocBlock`/`reset`
  throughput (real `BlockPool`, monotonic `BLOCKPOOL_FAST_MONOTONIC` path).
- **B `bench_multithreaded_contention`** — parallel `allocBlock` contention
  (uses `blocks_per_seg = 4096`; note the Rust side uses 256 → not apples-to-apples).
- **C `bench_cold_extension`** — cold pool create/expand/destroy (page faults).
- **D `bench_halo_gather`** — the **real** `HaloBlockSet::fillHaloBlock_<float>`
  (18³ halo gather over an active-block shell).
- **E `bench_laplacian_smoothing_e2e`** — the **real**
  `MultiresSparseGrid::applyChannelPdeFast<float>` for `laplTyp = 1` (regular
  Laplacian smoothing) and `laplTyp = 4` (`PDE_MEAN_CURVATURE`).

The active block set is a spherical shell (same formula as the Rust side) at
~14 % occupancy. See `generate_active_blocks()` / `build_grid()`.

## 7. Library-initialization gotchas (the parts that segfault)

`benchmark.cpp` drives `MultiresSparseGrid` directly, so it must do what the
demo's `main.cpp` normally does. If you touch `build_grid()`/`main()`, keep all
of these:

1. **`int nMaxThreads = -1;`** — a global the library expects (declared
   `extern` in `src/thread.h`, normally defined in the demo's `main.cpp`).
2. **`ThrInit()` + `ThrGetMaxNumberOfThreads()` + `ThrSetMaxNumberOfTBBThreads()`**
   before creating a grid — otherwise TBB-backed `ThrRunParallel` segfaults.
3. **`sg->prepareDataAccess()`** before `sg->getBlockDataPtr(bid,1,0)` — the
   channel's `_blockmap`/`_blockPool` are NULL until this runs.
4. **`setRefinementMap` before allocating blocks.** `create()` initializes every
   block's `bi->level` to `getNumLevels()-1` (coarsest). The smoothing/halo code
   only operates on `bi->level == 0` blocks. `build_grid()` sets
   `blockLevels[bid] = 0` for shell blocks and `getNumLevels()-1` elsewhere,
   calls `regularizeRefinementMap()` then `setRefinementMap(..., doResetChannels=false)`,
   then re-reads the active set from `getBlockInfo(bid)->level == 0`.

Without (4), `fillHaloBlock_` dereferences a NULL channel (level-2 channels
don't exist under `OPT_SINGLE_LEVEL`) and you get a silent segfault right after
`create:` prints "number of resolution levels = 3".

## 8. Comparing against the Rust port

The matching Rust benchmark is `../msbg-rs/benches/allocator_benches.rs`, run with:

```bash
cd ../msbg-rs && nix-shell && MSBG_BENCH_SCALE=small cargo bench --bench allocator_benches
```

It mirrors scenarios A–E: `blockpool_*`, `halo_gather`, `laplacian_compute_only`,
`laplacian_smoothing_e2e`. Both sides use the same shell occupancy generator, so
the active-block sets match. Known asymmetries to keep in mind when reading
numbers:

- Rust `HaloBlock::fill` is a simplified 7-neighbor copy; C++ `fillHaloBlock_`
  does full multires/boundary handling (Rust gather is therefore faster).
- Rust `kernel_laplacian_simd_16` applies a per-voxel **fluid mask**; the C++
  `laplTyp=1` smoothing path does not mask (masking lives in MSBG's multigrid
  path). This makes the Rust e2e look slower than C++.
- C++ `laplTyp=1` computes `phi += dt·L·|∇φ|`; the Rust kernel computes only `L`.

For an automated side-by-side comparison (IPC, cache misses, code size, layout),
see `.agents/skills/compare-cpp-rust/SKILL.md` — it describes spawning one
subagent per repo and aggregating their output.

## 9. ISPC

`ispc` (1.28) is available in the shell, but the repo contains **no `.ispc`
source** — `src/kernels_ispc.h` is a generated header from the original author's
machine, and `TEST_ISPC_LAPSM` is not defined in the makefile, so the library is
built without ISPC. The ISPC functions (`ispc_laplacian_smooth_halo_block`,
`ispc_meancurv_smooth_halo_block`) are therefore not linked; the fallback for a
"compute-only" C++ number is `e2e − gather` (derived).

## 10. Diagnostics

- `LogLevel = 2;` (global from `src/log.h`) enables `TRC` output; `TRCP` prints
  at level ≥ 1. Set it early in `main()` when debugging `create()`.
- `gdb -batch -ex run -ex bt ./build/bench_executable`, or `coredumpctl debug
  <pid>` / `coredumpctl list` for post-mortem backtraces. Crash addresses can be
  resolved with `addr2line -e build/bench_executable -f -C <offset>`.
- The shell also ships `strace`, `perf`, `pahole`, `llvm-mca`, `valgrind`, `rr`,
  `heaptrack` — see `.agents/skills/debugging/SKILL.md` for what each is for and
  example invocations (including instruction-peeping comparisons vs the Rust
  port).
- `./profile.sh <hot|contention|cold|halo|laplacian>` profiles one scenario and
  writes `build/profile.txt` (flat top-N self-time) + `build/flamegraph.svg`.
  See `.agents/skills/compare-cpp-rust/SKILL.md` for the Rust side and the
  side-by-side comparison flow.

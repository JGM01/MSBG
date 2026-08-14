---
name: debugging
description: Debug and profile the MSBG C++ benchmark/binary, and compare instruction-level behavior with the Rust port. Use when the benchmark segfaults, when chasing a performance discrepancy between C++ and msbg-rs, or when inspecting codegen/data layout.
---

# MSBG debugging

All tools live in the nix-shell (see `shell.nix`). Build first with
`./build_lib.sh` (then `./build_bench.sh` / `./build_difftest.sh` as needed).
Use `MSBG_BENCH_SCALE=small` for fast repro cycles.

## Tool cheat-sheet

| Tool | Purpose |
|------|---------|
| `gdb` | backtraces, breakpoints, core dumps |
| `strace` | syscalls, mmap, page faults, thread creation |
| `valgrind` | memory errors (`memcheck`), cache/call simulation |
| `rr` | record & replay a crash deterministically |
| `perf` | hardware counters, sampling, `annotate`, false sharing (`c2c`) |
| `objdump` / `llvm-objdump` | disassemble a function |
| `llvm-mca` | static instruction-throughput/latency of a loop |
| `pahole` | struct layout from DWARF (parity vs Rust) |
| `heaptrack` | heap allocations (find the OOM cases) |
| `gcc -fopt-info` | why a loop did/didn't vectorize |

The ELF tools (`objdump`, `perf`, `llvm-mca`, `pahole`) also work on the Rust
`msbg-rs` binaries, so they cover comparisons in both directions.

## Examples

### Crash / backtrace
```bash
gdb -batch -ex run -ex bt ./build/bench_executable
coredumpctl debug <pid>            # post-mortem from a systemd core dump
```

### Replay a flaky crash
```bash
# one-time: sudo sysctl kernel.perf_event_paranoid=1
rr record ./build/bench_executable
rr replay                          # deterministic; can step backwards
```

### Memory errors / leak
```bash
valgrind --leak-check=full --track-origins=yes ./build/difftest
```

### Hardware counters (IPC / cache misses)
```bash
perf stat -d ./build/bench_executable
```

### Hot-spot sampling -> annotated asm
```bash
perf record -g ./build/bench_executable
perf report                        # or: perf annotate <func>
```

### False sharing (common Rust-vs-C++ discrepancy)
```bash
perf c2c record ./build/bench_executable
perf c2c report
```

### Disassemble one function (works on the Rust binary too)
```bash
nm -C ./build/bench_executable | grep -i <name>   # find the mangled symbol
objdump -d -M intel --disassemble=<sym> ./build/bench_executable
```

### Static throughput of a hot loop
```bash
g++ -O3 -march=native -S src/halo.cpp -o /tmp/halo.s   # AT&T syntax (llvm-mca's default)
llvm-mca -mcpu=native -iterations=100 /tmp/halo.s
```
`llvm-mca` analyzes the first basic block by default; extract a single function
or loop with `-masm=att` and `sed -n '/<name>:/,/ret/p'` to target a specific one.

### Why gcc did/didn't vectorize a loop
```bash
g++ -O3 -march=native -fopt-info-vec -fopt-info-vec-missed -c src/halo.cpp
```

### Struct layout parity vs Rust
Objects built by `build_lib.sh` have no debug info; rebuild one with `-g`
(same flags as `build_difftest.sh`, plus `-g`) first:
```bash
g++ -g -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal -c src/blockpool.cpp -o /tmp/blockpool_dbg.o
pahole -C BlockPool /tmp/blockpool_dbg.o
pahole -C Block /tmp/blockpool_dbg.o
```

### Heap profile (the 7 GB OOM cases)
```bash
heaptrack ./build/bench_executable
heaptrack_print heaptrack.bench_executable.*.zst | head -30
```

### Syscall / thread tracing
```bash
strace -f -e trace=mmap,munmap,clone,futex ./build/bench_executable
```

## Notes

- `perf` is built for nixpkgs' kernel, which may be newer than the running one
  (`uname -r`); `perf stat`/`record` still work in practice.
- `pahole` and `perf annotate` source lines need `-g`; `build_lib.sh` objects
  are `-O3` only, though symbols remain for `objdump`/`nm`.
- `valgrind` reports "uninitialised value" in `MM_Str2UserId` (block-name
  hashing in `BlockPool::extend_`) — a known, benign-looking finding.

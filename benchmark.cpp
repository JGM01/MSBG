#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <omp.h>

#include "src/globdef.h"
#include "src/util.h"
#include "src/blockpool.h"
#include "src/msbg.h"
#include "src/halo.h"

using namespace MSBG;
using namespace SBG;

// Defined by the MSBG demo's main.cpp (extern in src/thread.h). We define it
// here and initialize it from the OpenMP thread count before benchmarking.
int nMaxThreads = -1;

// Emulates Criterion's black_box to prevent the compiler from optimizing away computations
template <typename T>
inline void black_box(T& value) {
    asm volatile("" : "+r,m"(value) : : "memory");
}

// Memory Layout & Domain Constants
const size_t BLOCK_SIZE = 16448; // 4096 * 4 bytes + 2 flags + 62 padding
const int BSX = 16;
const int N = 4096;               // 16^3 voxels per block

// Approx. fraction of the domain a surface shell occupies (mirrors Rust bench).
const double SHELL_OCCUPANCY = 0.14;

static bool gSmall = false;

static std::vector<int> hot_counts() {
    if (gSmall) return {1000, 10000, 50000};
    return {10000, 100000, 250000};
}

static std::vector<int> active_targets() {
    if (gSmall) return {1000, 5000, 10000};
    return {10000, 50000, 100000};
}

// ---------------------------------------------------------------------------
// Scenario A/B/C: BlockPool allocation (uses the REAL MSBG BlockPool)
// ---------------------------------------------------------------------------

void bench_hot_path_scaling() {
    std::cout << "--- Scenario A: Hot Path Scaling ---\n";
    for (int count : hot_counts()) {
        int blocks_per_seg = 4096;
        int max_segments = (count / blocks_per_seg) + 2;
        int max_blocks = max_segments * blocks_per_seg;

        BlockPool *pool = BlockPool::create("bench_hot", BLOCK_SIZE, max_blocks, 0, 0, 0);
        int iters = (count > 100000) ? 100 : 1000;

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < iters; run++) {
            for (int i = 0; i < count; i++) {
                void *ptr = pool->allocBlock(i);
                black_box(ptr);
            }
            pool->reset();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[C++] single_thread_" << count << "_blocks avg time: "
                  << (duration.count() / iters) << " ms\n";

        BlockPool::destroy(&pool, 0);
    }
}

void bench_multithreaded_contention() {
    std::cout << "\n--- Scenario B: Multithreaded Contention ---\n";
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    int blocks_per_thread = 4096;
    int total_blocks = num_threads * blocks_per_thread;

    int blocks_per_seg = 4096;
    int max_segments = (total_blocks / blocks_per_seg) + 16;
    int max_blocks = max_segments * blocks_per_seg;

    BlockPool *pool = BlockPool::create("bench_contention", BLOCK_SIZE, max_blocks, 0, 0, 0);
    int iters = 500;

    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < iters; run++) {
        #pragma omp parallel for schedule(static, 1)
        for (int t = 0; t < num_threads; t++) {
            for (int i = 0; i < blocks_per_thread; i++) {
                void *ptr = pool->allocBlock(i);
                black_box(ptr);
            }
        }
        pool->reset();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[C++] " << num_threads << "_threads_" << total_blocks << "_blocks avg time: "
              << (duration.count() / iters) << " ms\n";

    BlockPool::destroy(&pool, 0);
}

void bench_cold_extension() {
    std::cout << "\n--- Scenario C: Cold Allocations (OS Page Faults) ---\n";
    int count = gSmall ? 1000 : 10000;
    int max_blocks = 16 * 4096;
    int iters = 100;

    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < iters; run++) {
        BlockPool *pool = BlockPool::create("bench_cold", BLOCK_SIZE, max_blocks, 0, 0, 0);
        for (int i = 0; i < count; i++) {
            void *ptr = pool->allocBlock(i);
            black_box(ptr);
        }
        BlockPool::destroy(&pool, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[C++] cold_alloc_" << count << "_blocks avg time: "
              << (duration.count() / iters) << " ms\n";
}

// ---------------------------------------------------------------------------
// Shared helpers for the real-kernel smoothing / gather benchmarks
// ---------------------------------------------------------------------------

// Deterministic sparse occupancy: blocks whose center lies inside a spherical
// shell. Mirrors the Rust side exactly.
static std::vector<int> generate_active_blocks(int nbx, int nby, int nbz) {
    double cx = nbx * 0.5, cy = nby * 0.5, cz = nbz * 0.5;
    double r_out = std::min(std::min(nbx, nby), nbz) * 0.5;
    double r_in = r_out * 0.9;

    std::vector<int> active;
    for (int bz = 0; bz < nbz; bz++) {
        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                double dx = bx + 0.5 - cx;
                double dy = by + 0.5 - cy;
                double dz = bz + 0.5 - cz;
                double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d >= r_in && d <= r_out) {
                    active.push_back(bx + by * nbx + bz * nbx * nby);
                }
            }
        }
    }
    return active;
}

static MultiresSparseGrid *build_grid(int active_target, std::vector<int> &active) {
    int bpd = (int)std::ceil(std::cbrt(active_target / SHELL_OCCUPANCY));
    bpd = std::max(bpd, 8);
    int sx = bpd * BSX, sy = bpd * BSX, sz = bpd * BSX;

    MultiresSparseGrid *msg = MultiresSparseGrid::create(
        "bench", sx, sy, sz, BSX, -1, 0, -1,
        OPT_SINGLE_LEVEL | OPT_SINGLE_CHANNEL_FLOAT);

    SparseGrid<float> *sg = msg->getFloatChannel(CH_FLOAT_1, 0);
    active = generate_active_blocks(sg->nbx(), sg->nby(), sg->nbz());

    // Set the refinement map so active blocks live at the finest level (0)
    // and everything else at the coarsest level, exactly like the demo does
    // before allocating/initializing data blocks.
    std::vector<int> blockLevels(msg->nBlocks(), msg->getNumLevels() - 1);
    for (int bid : active) blockLevels[bid] = 0;
    msg->regularizeRefinementMap(blockLevels.data());
    msg->setRefinementMap(blockLevels.data(), NULL, -1, NULL, false);

    // Re-derive the active set from the (possibly regularized) refinement map.
    active.clear();
    for (int bid = 0; bid < msg->nBlocks(); bid++) {
        if (msg->getBlockInfo(bid)->level == 0) active.push_back(bid);
    }

    sg->prepareDataAccess();
    sg->setEmptyValue(0.0f);
    sg->setFullValue(1.0f);

    // Materialize active blocks with a deterministic (non-denormal) value.
    for (int bid : active) {
        float *d = sg->getBlockDataPtr(bid, 1, 0);
        for (int i = 0; i < N; i++) d[i] = 0.5f;
    }

    return msg;
}

// ---------------------------------------------------------------------------
// Scenario D: Halo gather (REAL HaloBlockSet::fillHaloBlock_)
// ---------------------------------------------------------------------------

void bench_halo_gather() {
    std::cout << "\n--- Scenario D: Halo Gather Throughput (real fillHaloBlock_) ---\n";

    for (int target : active_targets()) {
        std::vector<int> active;
        MultiresSparseGrid *msg = build_grid(target, active);
        int nThreads = omp_get_max_threads();

        HaloBlockSet haloBlocks(msg, 1, nThreads);

        int iters = (target >= 50000) ? 30 : 50;

        // Warmup
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)active.size(); i++) {
            int tid = omp_get_thread_num();
            float **hb = haloBlocks.fillHaloBlock_<float>(CH_FLOAT_1, active[i], 0, tid, OPT_BC_NEUMANN);
            black_box(hb);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < iters; run++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)active.size(); i++) {
                int tid = omp_get_thread_num();
                float **hb = haloBlocks.fillHaloBlock_<float>(CH_FLOAT_1, active[i], 0, tid, OPT_BC_NEUMANN);
                black_box(hb);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_ms = duration.count() / iters;
        double active_voxels = (double)active.size() * N;
        double gvoxels_per_sec = (active_voxels / (avg_ms / 1000.0)) / 1e9;

        std::cout << "[C++] shell_fill_" << target << " active=" << active.size()
                  << " avg time: " << avg_ms << " ms (" << gvoxels_per_sec << " Gvoxels/s)\n";

        MultiresSparseGrid::destroy(msg);
    }
}

// ---------------------------------------------------------------------------
// Scenario E: End-to-end smoothing (REAL applyChannelPdeFast)
// ---------------------------------------------------------------------------

void bench_laplacian_smoothing_e2e() {
    std::cout << "\n--- Scenario E: Laplacian Smoothing Throughput (real applyChannelPdeFast) ---\n";

    for (int target : active_targets()) {
        std::vector<int> active;
        MultiresSparseGrid *msg = build_grid(target, active);

        for (int laplTyp : {1, 4}) {
            const char *name = laplTyp == 1 ? "laplacian" : "mean_curvature";
            int numIter = (target >= 50000) ? 10 : 20;

            // Warmup
            msg->applyChannelPdeFast<float>(
                CH_FLOAT_1, CH_FLOAT_2, CH_FLOAT_3, &active, laplTyp,
                0, 0.0f, 0.0f, 0.0f, numIter, 1.0f, 0.025f,
                0.0f, NULL, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, NULL, 0);

            auto start = std::chrono::high_resolution_clock::now();
            msg->applyChannelPdeFast<float>(
                CH_FLOAT_1, CH_FLOAT_2, CH_FLOAT_3, &active, laplTyp,
                0, 0.0f, 0.0f, 0.0f, numIter, 1.0f, 0.025f,
                0.0f, NULL, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, NULL, 0);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> duration = end - start;
            double per_iter_ms = duration.count() / numIter;
            double active_voxels = (double)active.size() * N;
            double gvoxels_per_sec = (active_voxels / (per_iter_ms / 1000.0)) / 1e9;

            std::cout << "[C++] " << name << "_" << target << " active=" << active.size()
                      << " avg time/iter: " << per_iter_ms << " ms ("
                      << gvoxels_per_sec << " Gvoxels/s)\n";
        }

        MultiresSparseGrid::destroy(msg);
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "small") gSmall = true;
    const char *env = getenv("MSBG_BENCH_SCALE");
    if (env && std::string(env) == "small") gSmall = true;

    nMaxThreads = omp_get_max_threads();
    omp_set_num_threads(nMaxThreads);
    LogLevel = 2; // enable TRC trace to diagnose setup

    // Initialize MSBG's threading backend (TBB) like the demo's main.cpp does.
    ThrInit();
    int nMaxOmpThreadsAct = 0, nMaxTBBThreadsAct = 0;
    ThrGetMaxNumberOfThreads(&nMaxOmpThreadsAct, &nMaxTBBThreadsAct);
    int nMaxThreadsAct = std::max(nMaxOmpThreadsAct, nMaxTBBThreadsAct);
    ThrSetMaxNumberOfTBBThreads(nMaxThreads);
    nMaxThreads = std::max(1, std::min(nMaxThreadsAct, nMaxThreads));

    std::cout << "Starting C++ MSBG Benchmarks (real kernels)...\n\n";

    bench_hot_path_scaling();
    bench_multithreaded_contention();
    bench_cold_extension();
    bench_halo_gather();
    bench_laplacian_smoothing_e2e();

    return 0;
}

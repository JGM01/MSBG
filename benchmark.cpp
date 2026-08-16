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

        std::cout << "[C++] shell_fill_full_" << target << " active=" << active.size()
                  << " avg time: " << avg_ms << " ms (" << gvoxels_per_sec << " Gvoxels/s)\n";

        // Faces-only (do1stOrderOnly=1): used by the 7-point Laplacian.
        start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < iters; run++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)active.size(); i++) {
                int tid = omp_get_thread_num();
                float **hb = haloBlocks.fillHaloBlock_<float, 1, 1>(CH_FLOAT_1, active[i], 0, tid, OPT_BC_NEUMANN);
                black_box(hb);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        avg_ms = duration.count() / iters;
        gvoxels_per_sec = (active_voxels / (avg_ms / 1000.0)) / 1e9;

        std::cout << "[C++] shell_fill_faces_" << target << " active=" << active.size()
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

// ---------------------------------------------------------------------------
// Scenario F: Density quantization (REAL renderDens* SIMD)
// ---------------------------------------------------------------------------

void bench_density_quantization() {
    std::cout << "\n--- Scenario F: Density Quantization Throughput (real renderDens* SIMD) ---\n";

    const size_t large = 4ull * 1024 * 1024;

    for (size_t n : { (size_t)N, large }) {
        // dequant: u16 -> f32
        std::vector<uint16_t> src(n);
        for (size_t i = 0; i < n; i++) src[i] = (uint16_t)((i * 26543 + 1) & 0xFFFF);
        std::vector<float> dst(n, 0.0f);

        size_t iters = (n >= large) ? 8 : 2000;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t r = 0; r < iters; r++) {
            for (size_t i = 0; i < n; i += 8) {
                Vec8f v = renderDensToFloat_simd8<uint16_t>(&src[i]);
                v.store(&dst[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        black_box(dst[0]);
        black_box(dst[n - 1]);
        std::chrono::duration<double, std::milli> duration = end - start;
        double gvox = (double)(n * iters) / (duration.count() / 1000.0) / 1e9;
        std::cout << "[C++] density_dequant_" << n << " avg: "
                  << (duration.count() / iters) << " ms (" << gvox << " Gvoxels/s)\n";

        // quantize: f32 -> u16
        std::vector<float> fsrc(n);
        for (size_t i = 0; i < n; i++) fsrc[i] = (float)(i % 1000) / 999.0f;
        std::vector<uint16_t> qdst(n, 0);

        start = std::chrono::high_resolution_clock::now();
        for (size_t r = 0; r < iters; r++) {
            for (size_t i = 0; i < n; i += 8) {
                Vec8f vf;
                vf.load(&fsrc[i]);
                renderDensFromFloat_storeSimd8<uint16_t>(vf, &qdst[i]);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        black_box(qdst[0]);
        black_box(qdst[n - 1]);
        duration = end - start;
        gvox = (double)(n * iters) / (duration.count() / 1000.0) / 1e9;
        std::cout << "[C++] density_quantize_" << n << " avg: "
                  << (duration.count() / iters) << " ms (" << gvox << " Gvoxels/s)\n";
    }
}

// ---------------------------------------------------------------------------
// Scenario G: Field sampling (REAL interpolate* kernels)
// ---------------------------------------------------------------------------

static float bench_field(float x, float y, float z) {
    return 0.002f * x * x + 0.003f * y * y + 0.004f * z * z
         + 0.005f * x * y + 0.006f * x * z + 0.007f * y * z
         + 0.1f * x + 0.05f * y - 0.02f * z + 0.75f;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static void report_gsamples(const char *name, long long samples, double ms) {
    double gs = (double)samples / (ms / 1000.0) / 1e9;
    std::cout << "[C++] " << name << ": " << gs << " Gsamples/s\n";
}

void bench_interpolation() {
    std::cout << "\n--- Scenario G: Field Sampling (real interpolate*) ---\n";

    const int sx = 96, sy = 96, sz = 96, bsx = 16;
    SparseGrid<float> *sg =
        SparseGrid<float>::create("interp", sx, sy, sz, bsx, 0, -1, OPT_NO_BORDPAD);
    sg->prepareDataAccess();
    sg->setEmptyValue(0.0f);
    sg->setFullValue(1.0f);

    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
                sg->setValue(x, y, z, bench_field((float)x, (float)y, (float)z));

    const int NS = 100000;
    uint32_t seed = 12345;
    auto rng = [&seed]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return (float)seed / (float)UINT32_MAX;
    };
    std::vector<Vec4f> pos(NS);
    for (int i = 0; i < NS; i++)
        pos[i] = Vec4f(2.0f + rng() * 92.0f, 2.0f + rng() * 92.0f, 2.0f + rng() * 92.0f, 0.0f);

    // linear value
    {
        float acc = 0;
        int iters = 50;
        double t0 = now_ms();
        for (int r = 0; r < iters; r++)
            for (int i = 0; i < NS; i++) acc += sg->interpolateFloatFast(pos[i], OPT_IPCORNER);
        black_box(acc);
        report_gsamples("linear_value", (long long)NS * iters, now_ms() - t0);
    }

    // linear value + gradient
    {
        float acc = 0;
        int iters = 50;
        double t0 = now_ms();
        for (int r = 0; r < iters; r++)
            for (int i = 0; i < NS; i++) {
                float v, g[3];
                sg->interpolateLinearWithGradient<1, 1>(pos[i], OPT_IPCORNER, &v, g);
                acc += v + g[0] + g[1] + g[2];
            }
        black_box(acc);
        report_gsamples("linear_grad", (long long)NS * iters, now_ms() - t0);
    }

    // cubic value + gradient
    {
        float acc = 0;
        int iters = 20;
        double t0 = now_ms();
        for (int r = 0; r < iters; r++)
            for (int i = 0; i < NS; i++) {
                float v, g[4];
                Vec4f cg;
                sg->interpolateWithDerivs<0>(IP_BSPLINE_CUBIC, pos[i], 0, &v, &cg);
                cg.store(g);
                acc += v + g[0] + g[1] + g[2];
            }
        black_box(acc);
        report_gsamples("cubic_grad", (long long)NS * iters, now_ms() - t0);
    }

    // cubic value + gradient + Hessian
    {
        float acc = 0;
        int iters = 10;
        double t0 = now_ms();
        for (int r = 0; r < iters; r++)
            for (int i = 0; i < NS; i++) {
                float v, g[3], h[6];
                sg->interpolateWithSecondDerivs<0>(pos[i], 0, &v, g, h);
                acc += v + h[0] + h[1] + h[2] + h[3] + h[4] + h[5];
            }
        black_box(acc);
        report_gsamples("cubic_hess", (long long)NS * iters, now_ms() - t0);
    }
}

int main(int argc, char **argv) {
    std::string scenario;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "small") gSmall = true;
        else scenario = a;
    }
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

    bool all = scenario.empty();
    bool known = scenario == "hot" || scenario == "contention" ||
                 scenario == "cold" || scenario == "halo" || scenario == "laplacian" ||
                 scenario == "density" || scenario == "interp";
    if (!all && !known) {
        std::cerr << "unknown scenario '" << scenario
                  << "' (use: hot contention cold halo laplacian density interp)\n";
        return 1;
    }

    std::cout << "Starting C++ MSBG Benchmarks (real kernels)...\n\n";

    if (all || scenario == "hot")        bench_hot_path_scaling();
    if (all || scenario == "contention") bench_multithreaded_contention();
    if (all || scenario == "cold")       bench_cold_extension();
    if (all || scenario == "halo")       bench_halo_gather();
    if (all || scenario == "laplacian")  bench_laplacian_smoothing_e2e();
    if (all || scenario == "density")    bench_density_quantization();
    if (all || scenario == "interp")     bench_interpolation();

    return 0;
}

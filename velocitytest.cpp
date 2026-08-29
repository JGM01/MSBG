// Velocity-gather benchmark harness for the Rust step-13 G2P comparison.
//
// Fills a single-level SBG::SparseGrid<Vec3Float> with a deterministic field and
// times the REAL `interpolateVec3Float<IP_LINEAR>` (cell-centered) over N
// deterministic random interior positions. The Rust side
// (msbg-rs/benches/velocity_bench.rs) runs the same field/positions through its
// `SampleVec3` gather (and its step-13 staggered `MacSampler`), so the two
// throughputs are directly comparable.
//
// Usage: velocitytest <nthreads> <res> <nsamples>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <vectorclass/vectorclass.h>
#include "src/globdef.h"
#include "src/mtool.h"
#include "src/fastmath.h"
#include "src/util.h"
#include "src/thread.h"
#include "src/blockpool.h"
#include "src/sbg.h"

using namespace SBG;

int nMaxThreads = -1;

int main(int argc, char **argv) {
    ThrInit();
    nMaxThreads = argc > 1 ? atoi(argv[1]) : 1;
    ThrSetMaxNumberOfTBBThreads(nMaxThreads);
    const int sx = argc > 2 ? atoi(argv[2]) : 128;
    const int sy = sx, sz = sx, bsx = 16;
    const int n = argc > 3 ? atoi(argv[3]) : 1000000;

    SparseGrid<Vec3Float> *sg =
        SparseGrid<Vec3Float>::create("vel", sx, sy, sz, bsx, 0, -1, OPT_NO_BORDPAD);
    sg->prepareDataAccess();
    sg->setEmptyValue(Vec3Float(0.0f));
    sg->setFullValue(Vec3Float(0.0f));

    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++) {
                float fx = (float)x, fy = (float)y, fz = (float)z;
                sg->setValue(x, y, z,
                             Vec3Float(0.002f * fx + 0.003f * fy + 0.004f * fz + 0.1f,
                                       0.001f * fx + 0.005f * fy + 0.002f * fz - 0.2f,
                                       0.004f * fx + 0.001f * fy + 0.006f * fz + 0.3f));
            }

    // Deterministic interior positions (LCG), identical on the Rust side.
    float *pos = new float[3 * n];
    uint32_t seed = 12345;
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        float rx = (float)seed / 4294967295.0f;
        seed = seed * 1664525u + 1013904223u;
        float ry = (float)seed / 4294967295.0f;
        seed = seed * 1664525u + 1013904223u;
        float rz = (float)seed / 4294967295.0f;
        pos[3 * i + 0] = 2.0f + rx * (sx - 4.0f);
        pos[3 * i + 1] = 2.0f + ry * (sy - 4.0f);
        pos[3 * i + 2] = 2.0f + rz * (sz - 4.0f);
    }

    const int iters = 5;
    float acc = 0.0f;
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        acc = 0.0f;
        for (int i = 0; i < n; i++) {
            Vec3Float v = sg->interpolateVec3Float<IP_LINEAR>(
                1, &pos[3 * i], -1, OPT_IP_NO_BORDER_CHECK);
            acc += v[0] + v[1] + v[2];
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        if (sec < best) best = sec;
    }
    double mpairs = (double)n / best / 1e6;
    printf("[C++] velocity_gather: %d samples in %.3f ms (%.2f Msamples/s, acc=%.6f)\n",
           n, best * 1000.0, mpairs, acc);

    delete[] pos;
    return 0;
}

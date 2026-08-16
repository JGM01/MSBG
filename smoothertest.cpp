// Differential test harness for the Rust step-8 8-color PDE smoother.
//
// Builds a single-level MultiresSparseGrid filled with a deterministic
// polynomial field, runs the REAL `applyChannelPdeFast` in its *live* path —
// `-(laplTyp + OPT_8_COLOR_SCHEME)` (8-color in-place, matching
// msbg_demo.cpp:716) — for 4 iterations, and prints the resulting full level-0
// field. The Rust side (msbg-rs/tests/difftest_smoother.rs) runs its `Sweeper`
// on the same field and compares within tolerance (f32 + FMA -> not bit-exact).
//
// numIter is EVEN (4) to sidestep the C++ `nMaxIter++` even-izing quirk that
// leaks into the colored path.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <vectorclass/vectorclass.h>
#include "src/globdef.h"
#include "src/mtool.h"
#include "src/fastmath.h"
#include "src/util.h"
#include "src/thread.h"
#include "src/blockpool.h"
#include "src/msbg.h"

using namespace MSBG;

int nMaxThreads = -1;

static float field(int x, int y, int z) {
    // Smooth bounded field (values O(1), small gradients) so 4 iterations of
    // the 8-color sweep stay well-conditioned for an absolute-tolerance diff.
    float fx = (float)x, fy = (float)y, fz = (float)z;
    return sinf(0.1f * fx) * cosf(0.08f * fy) * sinf(0.06f * fz);
}

static MultiresSparseGrid *build_grid() {
    const int sx = 64, sy = 64, sz = 64, bsx = 16;

    MultiresSparseGrid *msg = MultiresSparseGrid::create(
        "smoothertest", sx, sy, sz, bsx, -1, 0, -1,
        OPT_SINGLE_LEVEL | OPT_SINGLE_CHANNEL_FLOAT);

    SparseGrid<float> *sg = msg->getFloatChannel(CH_FLOAT_1, 0);

    std::vector<int> blockLevels(msg->nBlocks(), 0);
    msg->regularizeRefinementMap(blockLevels.data());
    msg->setRefinementMap(blockLevels.data(), NULL, -1, NULL, false);

    std::vector<int> active(msg->nBlocks());
    for (int i = 0; i < (int)active.size(); i++) active[i] = i;

    sg->prepareDataAccess();
    sg->setEmptyValue(0.0f);
    sg->setFullValue(1.0f);

    int nbx = sg->nbx(), nby = sg->nby();
    for (int bid : active) {
        float *d = sg->getBlockDataPtr(bid, 1, 0);
        int bx = bid % nbx;
        int by = (bid / nbx) % nby;
        int bz = bid / (nbx * nby);
        for (int z = 0; z < bsx; z++)
            for (int y = 0; y < bsx; y++)
                for (int x = 0; x < bsx; x++)
                    d[x + y * bsx + z * bsx * bsx] =
                        field(bx * bsx + x, by * bsx + y, bz * bsx + z);
    }

    return msg;
}

int main() {
    ThrInit();
    nMaxThreads = 1;
    ThrSetMaxNumberOfTBBThreads(nMaxThreads);
    LogLevel = 0; // silence TRC/TRCP traces so stdout is pure data

    const float dt = 0.01f;
    const int numIter = 4;

    for (int laplTyp : {1, 4}) {
        MultiresSparseGrid *msg = build_grid();
        std::vector<int> active(msg->nBlocks());
        for (int i = 0; i < (int)active.size(); i++) active[i] = i;

        msg->applyChannelPdeFast<float>(
            CH_FLOAT_1, CH_NULL, CH_NULL, &active,
            -(laplTyp + OPT_8_COLOR_SCHEME), // 8-color in-place (the live path)
            0, 0.0f, 0.0f, 0.0f, numIter, 1.0f, dt,
            0.0f, NULL, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, NULL, 0);

        SparseGrid<float> *sg = msg->getFloatChannel(CH_FLOAT_1, 0);

        printf("%d\n", laplTyp);
        for (int bid = 0; bid < msg->nBlocks(); bid++) {
            float *d = sg->getBlockDataPtr(bid, 0, 0);
            for (int i = 0; i < 16 * 16 * 16; i++) printf("%.9g\n", d[i]);
        }

        MultiresSparseGrid::destroy(msg);
    }

    return 0;
}

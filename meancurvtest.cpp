// Differential test harness for the Rust step-6 stencil kernels.
//
// Builds a single-level MultiresSparseGrid filled with a deterministic
// polynomial field, runs the REAL `applyChannelPdeFast` for laplTyp 1
// (Laplacian), 4 (mean-curvature) and 2 (bi-Laplacian) for exactly one
// Jacobi iteration (numIter=1 -> one smoothing pass + copy-back), and prints
// the resulting values of the interior block (1,1,1). The Rust side
// (msbg-rs/tests/difftest_stencil.rs) gathers the same block's halo, applies
// its kernel once, and compares within tolerance (f32 + FMA -> not bit-exact).
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

// Deterministic polynomial field. Values stay O(10) so an absolute 1e-4
// comparison is meaningful; has non-trivial gradient, Hessian and 4th
// derivative (so all three kernels do real work). Quadratic + quartic,
// evaluated in f32 identically on both sides (tolerance absorbs ulp noise).
static float field(int x, int y, int z) {
    float u = (float)(x - 24);
    float v = (float)(y - 24);
    float w = (float)(z - 24);
    float q = 0.01f * (u * u + 2.0f * v * v + 3.0f * w * w + u * v);
    float r = 0.001f * (u * u * u * u + v * v * v * v + w * w * w * w);
    return q + r;
}

static MultiresSparseGrid *build_grid() {
    const int sx = 64, sy = 64, sz = 64, bsx = 16;

    MultiresSparseGrid *msg = MultiresSparseGrid::create(
        "meancurvtest", sx, sy, sz, bsx, -1, 0, -1,
        OPT_SINGLE_LEVEL | OPT_SINGLE_CHANNEL_FLOAT);

    SparseGrid<float> *sg = msg->getFloatChannel(CH_FLOAT_1, 0);

    // Every block active at the finest (only) level.
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

    // Interior block (1,1,1) of the 4^3 block grid.
    const int center_bid = 1 + 1 * 4 + 1 * 16;

    for (int laplTyp : {1, 4, 2}) {
        MultiresSparseGrid *msg = build_grid();
        std::vector<int> active(msg->nBlocks());
        for (int i = 0; i < (int)active.size(); i++) active[i] = i;

        msg->applyChannelPdeFast<float>(
            CH_FLOAT_1, CH_FLOAT_2, CH_FLOAT_3, &active, laplTyp,
            0, 0.0f, 0.0f, 0.0f, 1, 1.0f, dt,
            0.0f, NULL, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, NULL, 0);

        SparseGrid<float> *sg = msg->getFloatChannel(CH_FLOAT_1, 0);
        float *d = sg->getBlockDataPtr(center_bid, 0, 0);

        printf("%d\n", laplTyp);
        for (int i = 0; i < 16 * 16 * 16; i++) printf("%.9g\n", d[i]);

        MultiresSparseGrid::destroy(msg);
    }

    return 0;
}

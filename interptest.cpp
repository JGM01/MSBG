// Differential test harness for the Rust step-4 interpolation port.
//
// Builds a single-level SBG::SparseGrid<float> filled with a deterministic
// polynomial field and prints, per sample position, the linear value+gradient,
// cubic value+gradient, and cubic Hessian. The Rust side
// (msbg-rs/tests/difftest_interp.rs) runs the same field/positions natively and
// compares within tolerance (not bit-exact: the Rust port uses f32 + FMA).
#include <cstdint>
#include <cstdio>
#include <cstring>

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

static float field(int x, int y, int z) {
    float fx = (float)x, fy = (float)y, fz = (float)z;
    // Quadratic with non-trivial cross terms so the Hessian off-diagonals
    // (fxy, fxz, fyz) are meaningfully non-zero.
    return 0.002f * fx * fx + 0.003f * fy * fy + 0.004f * fz * fz
         + 0.005f * fx * fy + 0.006f * fx * fz + 0.007f * fy * fz
         + 0.1f * fx + 0.05f * fy - 0.02f * fz + 0.75f;
}

int main() {
    ThrInit();
    nMaxThreads = 1;
    ThrSetMaxNumberOfTBBThreads(nMaxThreads);

    const int sx = 48, sy = 48, sz = 48, bsx = 16;

    SparseGrid<float> *sg =
        SparseGrid<float>::create("interp", sx, sy, sz, bsx, 0, -1, OPT_NO_BORDPAD);
    sg->prepareDataAccess();
    sg->setEmptyValue(0.0f);
    sg->setFullValue(1.0f);

    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
                sg->setValue(x, y, z, field(x, y, z));

    static const float positions[][3] = {
        {5.3f, 7.7f, 3.2f},
        {10.5f, 12.25f, 20.75f},
        {15.1f, 30.9f, 10.4f},
        {20.5f, 21.25f, 40.75f},
        {25.25f, 8.1f, 33.3f},
        {30.9f, 15.4f, 25.6f},
        {35.5f, 40.2f, 12.8f},
        {40.1f, 25.5f, 30.2f},
    };
    const int npos = (int)(sizeof(positions) / sizeof(positions[0]));

    for (int i = 0; i < npos; i++) {
        Vec4f pos(positions[i][0], positions[i][1], positions[i][2], 0.0f);

        float lv, lg[3];
        sg->interpolateLinearWithGradient<1, 1>(pos, OPT_IPCORNER, &lv, lg);

        float cv, cg4[4];
        Vec4f cgrad;
        sg->interpolateWithDerivs<0>(IP_BSPLINE_CUBIC, pos, 0, &cv, &cgrad);
        cgrad.store(cg4);

        float hv, hg[3], hh[6];
        sg->interpolateWithSecondDerivs<0>(pos, 0, &hv, hg, hh);

        printf("%.9g %.9g %.9g %.9g  %.9g %.9g %.9g %.9g  %.9g %.9g %.9g %.9g %.9g %.9g\n",
               lv, lg[0], lg[1], lg[2],
               cv, cg4[0], cg4[1], cg4[2],
               hh[0], hh[1], hh[2], hh[3], hh[4], hh[5]);
    }

    return 0;
}

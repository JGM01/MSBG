#include <cmath>
#include <cstdio>
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

static float pressure_value(int x, int y, int z) {
    return 0.002f * x * x + 0.003f * y * y + 0.004f * z * z
         + 0.01f * x - 0.02f * y + 0.03f * z + 0.25f;
}

static float rhs_value(int x, int y, int z) {
    return 0.1f + 0.001f * x - 0.0005f * y + 0.00025f * z;
}

int main() {
    ThrInit();
    nMaxThreads = 1;
    ThrSetMaxNumberOfTBBThreads(nMaxThreads);
    LogLevel = 0;

    const int size = 32;
    const int block_size = 16;
    MultiresSparseGrid *grid = MultiresSparseGrid::create(
        "pressuretest", size, size, size, block_size, -1, 0, -1, 0);

    std::vector<int> levels(grid->nBlocks(), 0);
    std::vector<unsigned> block_flags(grid->nBlocks(), BLK_EXISTS | BLK_ONLY_FLUID);
    grid->regularizeRefinementMap(levels.data());
    grid->setRefinementMap(levels.data(), block_flags.data(), -1, NULL, true);

    std::vector<int> active(grid->nBlocks());
    for (int bid = 0; bid < grid->nBlocks(); ++bid) active[bid] = bid;

    SparseGrid<PSFloat> *pressure = grid->getPSFloatChannel(CH_PRESSURE, 0);
    SparseGrid<PSFloat> *rhs = grid->getPSFloatChannel(CH_DIVERGENCE, 0);
    SparseGrid<PSFloat> *diagonal = grid->getPSFloatChannel(CH_DIAGONAL, 0);
    SparseGrid<PSFloat> *output = grid->getPSFloatChannel(CH_CG_Q, 0);
    pressure->prepareDataAccess();
    rhs->prepareDataAccess();
    diagonal->prepareDataAccess();
    output->prepareDataAccess();
    pressure->setEmptyValue(0.0f);
    pressure->setFullValue(0.0f);
    rhs->setEmptyValue(0.0f);
    rhs->setFullValue(0.0f);
    diagonal->setEmptyValue(1.0f / 6.0f);
    diagonal->setFullValue(1.0f / 6.0f);
    output->setEmptyValue(0.0f);
    output->setFullValue(0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        SparseGrid<float> *face_area = grid->getFaceAreaChannel(axis, 0);
        face_area->prepareDataAccess();
        face_area->setEmptyValue(1.0f);
        face_area->setFullValue(1.0f);
    }
    const int nbx = pressure->nbx();
    const int nby = pressure->nby();

    for (int bid : active) {
        PSFloat *p = pressure->getBlockDataPtr(bid, 1, 0);
        PSFloat *b = rhs->getBlockDataPtr(bid, 1, 0);
        PSFloat *d = diagonal->getBlockDataPtr(bid, 1, 0);
        output->getBlockDataPtr(bid, 1, 0);
        int bx = bid % nbx;
        int by = (bid / nbx) % nby;
        int bz = bid / (nbx * nby);
        for (int z = 0; z < block_size; ++z) {
            for (int y = 0; y < block_size; ++y) {
                for (int x = 0; x < block_size; ++x) {
                    int i = x + block_size * (y + block_size * z);
                    int gx = bx * block_size + x;
                    int gy = by * block_size + y;
                    int gz = bz * block_size + z;
                    p[i] = pressure_value(gx, gy, gz);
                    b[i] = rhs_value(gx, gy, gz);
                    d[i] = 1.0f / 6.0f;
                }
            }
        }
    }

    grid->multiplyLaplacianMatrixOpt(
        0, 0, CH_PRESSURE, CH_NULL, 1.0f, CH_CG_Q, NULL, &active);

    const int samples[][3] = {
        {2, 3, 4}, {7, 11, 5}, {14, 9, 13}, {15, 8, 8},
        {16, 8, 8}, {20, 17, 12}, {25, 24, 21}, {29, 28, 27}
    };
    std::puts("MATVEC 8");
    for (const auto &s : samples) {
        std::printf("%.9g\n", (double)output->getValueGen_(s[0], s[1], s[2]));
    }

    grid->relaxBlockList(
        &active, 0, 0, CH_PRESSURE, CH_DIVERGENCE, 0, CH_PRESSURE);
    std::puts("RELAX 8");
    for (const auto &s : samples) {
        std::printf("%.9g\n", (double)pressure->getValueGen_(s[0], s[1], s[2]));
    }

    MultiresSparseGrid::destroy(grid);
    return 0;
}

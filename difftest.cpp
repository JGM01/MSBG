// Differential test harness: runs a fixed write/read script over a real
// SBG::SparseGrid<float> and prints an FNV-1a hash of every voxel's bit
// pattern. The Rust side (msbg-rs/tests/difftest_cpp.rs) runs the same script
// natively and compares hashes.
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

static uint64_t fnv1a(uint64_t h, const uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)buf[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main() {
    ThrInit();
    nMaxThreads = 1;
    ThrSetMaxNumberOfTBBThreads(nMaxThreads);

    const int sx = 17, sy = 33, sz = 5, bsx = 16;

    SparseGrid<float> *sg =
        SparseGrid<float>::create("diff", sx, sy, sz, bsx, 0, -1, OPT_NO_BORDPAD);
    sg->prepareDataAccess();
    sg->setEmptyValue(0.0f);
    sg->setFullValue(1.0f);

    // Writes (avoid blocks (1,1,0) and (0,1,0), marked full/empty below)
    sg->setValue(0, 0, 0, 42.0f);
    sg->setValue(16, 0, 0, 7.0f);
    sg->setValue(0, 32, 0, 3.0f);
    sg->setValue(16, 32, 4, 99.0f);
    sg->setValue(5, 5, 2, 123.5f);
    sg->setValue(3, 3, 3, 0.25f);

    sg->setFullBlock(sg->getBlockIndex(1, 1, 0));
    sg->setEmptyBlock(sg->getBlockIndex(0, 1, 0));

    uint64_t h = 0xcbf29ce484222325ULL;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                float v = sg->getValue(x, y, z);
                uint32_t bits;
                memcpy(&bits, &v, sizeof(bits));
                uint8_t b[4] = {
                    (uint8_t)(bits),
                    (uint8_t)(bits >> 8),
                    (uint8_t)(bits >> 16),
                    (uint8_t)(bits >> 24),
                };
                h = fnv1a(h, b, 4);
            }
        }
    }

    printf("%016llx\n", (unsigned long long)h);
    return 0;
}

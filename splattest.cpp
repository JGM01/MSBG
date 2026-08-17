// Differential test harness for the Rust step-9 surface-reconstruction
// pipeline (splat + finalize + mean-curvature smoothing).
//
// Replicates the REAL demo path from msbg_test_sparse (msbg_demo.cpp:228):
// readVerticesFromPLY -> placement (bunny-of-bunnies) -> active blocks ->
// setRefinementMap -> 8-color lock-free splat -> finalize -> applyChannelPdeFast
// (mean curvature, 8-color in-place).
//
// Output (stdout): a header line "n_active n_total n_particles" followed by
// two full level-0 u16 fields (n_total * 4096 values each), raw binary:
//   1) after splat+finalize
//   2) after splat+finalize+nIter mean-curvature sweeps
// The Rust side (msbg-rs/tests/difftest_splat.rs) compares both within a
// budgeted tolerance.
//
// Usage: splattest <ply> <sx> <sy> <sz> <nInst(0=auto)> <scale_factor>
//                  <rParticle> <nbDist> <nIter> <dt>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>

#include <vectorclass/vectorclass.h>
#include "src/globdef.h"
#include "src/mtool.h"
#include "src/fastmath.h"
#include "src/util.h"
#include "src/thread.h"
#include "src/blockpool.h"
#include "src/msbg.h"

using namespace MSBG;
using namespace SBG;

int nMaxThreads = -1;

typedef uint32_t ParticleIdx;

typedef struct
{
  Vec3Float pos;
  ParticleIdx idxNext;
}
Particle;

static Particle *particles = NULL;
static uint64_t nMaxParticles = 0;

inline static ParticleIdx getParticleIdx(int nInst, int iInst, int i)
{
  return nInst * (ParticleIdx)(iInst) + (ParticleIdx)(i) + 1; // slot 0 reserved
}

inline static Particle *getParticle(ParticleIdx ixp)
{
  UT_ASSERT2(ixp > 0 && ixp < nMaxParticles + 1);
  return &particles[ixp];
}

inline static void pushToListMonotonic(Particle *particle, ParticleIdx idxParticle,
                                       ParticleIdx *listRoot)
{
  ParticleIdx idx = InterlockedExchange((LONG volatile *)(listRoot), idxParticle);
  particle->idxNext = idx;
}

static std::tuple<std::vector<Vec3Float>*, Vec3Float, Vec3Float>
readVerticesFromPLY(const char *filename)
{
  std::ifstream file(filename);
  std::string line;
  std::vector<Vec3Float>* vertices = new std::vector<Vec3Float>();
  Vec3Float bbMin(1e20), bbMax(-1e20);

  bool parsingHeader = true;
  int vertexCount = 0, verticesParsed = 0;

  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    if (parsingHeader)
    {
      if (line.rfind("element ve", 0) == 0)
      {
        iss >> line >> line >> vertexCount;
      }
      else if (line == "end_header")
      {
        parsingHeader = false;
      }
    }
    else
    {
      if (verticesParsed < vertexCount)
      {
        Vec3Float vertex;
        iss >> vertex.x >> vertex.y >> vertex.z;
        vertices->push_back(vertex);
        for (int k = 0; k < 3; k++)
        {
          bbMin[k] = std::min(bbMin[k], vertex[k]);
          bbMax[k] = std::max(bbMax[k], vertex[k]);
        }
        verticesParsed++;
      }
    }
  }
  return std::make_tuple(vertices, bbMin, bbMax);
}

int main(int argc, char **argv)
{
  if (argc != 11)
  {
    fprintf(stderr,
            "usage: %s <ply> <sx> <sy> <sz> <nInst(0=auto)> <scale_factor> "
            "<rParticle> <nbDist> <nIter> <dt>\n",
            argv[0]);
    return 1;
  }

  const char *plyFile = argv[1];
  int sx = atoi(argv[2]), sy = atoi(argv[3]), sz = atoi(argv[4]);
  int nInstArg = atoi(argv[5]);
  float scaleFactor = atof(argv[6]);
  float rParticle = atof(argv[7]);
  float nbDist = atof(argv[8]);
  int nIter = atoi(argv[9]);
  float dt = atof(argv[10]);

  ThrInit();  // Machine default thread count (matches the demo: nMaxThreads starts -1 and
  // is resolved to the detected count after ThrInit; ThrSetMaxNumberOfTBBThreads
  // is only called for positive user values).
  nMaxThreads = omp_get_max_threads();  LogLevel = 0; // silence TRC/TRCP so stdout is pure data

  int bsx = 16;
  const float rScan = rParticle + nbDist;

  auto [basePoints, basePointsBMin, basePointsBMax] = readVerticesFromPLY(plyFile);
  Vec3Float basePointsSpan;
  float basePointsSpanMax = -1e20;
  for (int k = 0; k < 3; k++)
  {
    basePointsSpan[k] = basePointsBMax[k] - basePointsBMin[k];
    basePointsSpanMax = std::max(basePointsSpanMax, basePointsSpan[k]);
  }
  if (!(basePointsSpanMax > 0.f) || !std::isfinite(basePointsSpanMax))
  {
    fprintf(stderr, "degenerate point cloud (zero span)\n");
    return 1;
  }

  int nBasePoints = (int)basePoints->size();
  int nInst = nInstArg == 0 ? nBasePoints : nInstArg;
  uint64_t nTotalParticles = (uint64_t)nInst * (uint64_t)nBasePoints;
  nMaxParticles = nTotalParticles;

  // Single-level MSBG grid like the demo (OPT_SINGLE_LEVEL).
  MSBG::MultiresSparseGrid *msbg = MSBG::MultiresSparseGrid::create(
      "SPLATTEST", sx, sy, sz, bsx, -1, 0, -1,
      MSBG::OPT_SINGLE_LEVEL | MSBG::OPT_SINGLE_CHANNEL_FLOAT);
  LONG *blockActive = NULL;
  ParticleIdx *particlesPerBlock = NULL;
  ALLOCARR0_(blockActive, LONG, msbg->nBlocks());
  ALLOCARR0_(particlesPerBlock, ParticleIdx, msbg->nBlocks());

  int chan = CH_UINT16_1;
  SparseGrid<uint16_t> *sg0 = msbg->getUint16Channel(chan, 0);

  // ---- Placement + active blocks + per-block particle lists ---------------
  float rScanBsx = rScan / (float)sg0->bsx();
  int bxMax = sg0->nbx() - 1, byMax = sg0->nby() - 1, bzMax = sg0->nbz() - 1;

  ALLOCARR_(particles, Particle, nTotalParticles + 1);
  uint64_t nActParticles = 0;

  {
    const float scale2DestBlockGrid = 1.0f / (float)sg0->bsx();
    const float baseScale = 1.0f / basePointsSpanMax;
    const float sxyzMax = (float)sg0->sxyzMax();
    const float sxyzMin = (float)sg0->sxyzMin();
    const float scale = sxyzMin * scaleFactor;

    using ThreadLocals = struct { size_t nActParticles; };    ThrRunParallel<ThreadLocals>(nInst, nullptr,
      [&](ThreadLocals &tls, int tid, size_t iInst)
      {        Vec4f baseMin(basePointsBMin.x, basePointsBMin.y, basePointsBMin.z, 0);
        Vec3Float &pos_ = (*basePoints)[iInst];
        Vec4f pos = sxyzMax * baseScale * (Vec4f(pos_.x, pos_.y, pos_.z, 0.f) - baseMin);
        Vec4f pos0 = sxyzMax * 0.2f + 0.6f * pos;

        for (int i = 0; i < nBasePoints; i++)
        {
          Vec3Float &bp = (*basePoints)[i];
          Vec4f p = baseScale * (Vec4f(bp.x, bp.y, bp.z, 0.f) - baseMin);
          p = pos0 + scale * p;
          if (!msbg->isInDomainRange(p)) continue;
          Vec4i ipos = truncate_to_int(p);
          if (!sg0->inRange(ipos)) continue;

          ParticleIdx ixp = getParticleIdx(nInst, iInst, i);
          Particle *particle = getParticle(ixp);
          particle->pos = Vec3Float(p);

          Vec4f bpos = p * scale2DestBlockGrid;
          Vec4i bpos1 = max(truncate_to_int(bpos - rScanBsx), 0),
                bpos2 = min(truncate_to_int(bpos + rScanBsx),
                            Vec4i(bxMax, byMax, bzMax, INT32_MAX));
          int bx1 = vget_x(bpos1), by1 = vget_y(bpos1), bz1 = vget_z(bpos1);
          int bx2 = vget_x(bpos2), by2 = vget_y(bpos2), bz2 = vget_z(bpos2);
          for (int bz = bz1; bz <= bz2; bz++)
            for (int by = by1; by <= by2; by++)
              for (int bx = bx1; bx <= bx2; bx++)
              {
                int bid = sg0->getBlockIndex(bx, by, bz);
                InterlockedIncrement((volatile LONG *)(&blockActive[bid]));
              }

          int bid = sg0->getBlockIndex(sg0->getBlockCoords(ipos));
          pushToListMonotonic(particle, ixp, &particlesPerBlock[bid]);
          tls.nActParticles++;
        }
      },
      [&](ThreadLocals &tls, int tid) { nActParticles += tls.nActParticles; });
  }

  // ---- Refinement map + allocate active value blocks (fill 65535) ----------
  LongInt nActiveBlocks = 0;
  std::unique_ptr<int[]> blockLevels(new int[msbg->nBlocks()]);
  for (int i = 0; i < msbg->nBlocks(); i++)
  {
    int level = msbg->getNumLevels() - 1;
    if (blockActive[i]) { level = 0; nActiveBlocks++; }
    blockLevels[i] = level;
  }  msbg->regularizeRefinementMap(blockLevels.get());  msbg->setRefinementMap(blockLevels.get(), NULL, -1, NULL, false);

  std::vector<int> activeBlocks;
  sg0->reset();  sg0->prepareDataAccess(chan);
  sg0->setEmptyValue(renderDensFromFloat_<uint16_t>(0.0f));
  sg0->setFullValue(renderDensFromFloat_<uint16_t>(1.0f));

  {
    using ThreadLocals = struct { std::vector<int> activeBlocks; };
    ThrRunParallel<ThreadLocals>(sg0->nBlocks(),
      [&](ThreadLocals &tls, int tid) { tls.activeBlocks.reserve(256); },
      [&](ThreadLocals &tls, int tid, int bid)
      {
        BlockInfo *bi = msbg->getBlockInfo(bid);
        FLAG_RESET(bi->flags, BLK_EXISTS);
        if (bi->level > 0) { sg0->setEmptyBlock(bid); return; }
        tls.activeBlocks.push_back(bid);
        FLAG_SET(bi->flags, BLK_EXISTS);
        uint16_t *data = sg0->getBlockDataPtr(bid, 1, 0);
        for (int i = 0; i < sg0->nVoxelsInBlock(); i++)
          data[i] = renderDensFromFloat_<uint16_t>(1.0f);
      },
      [&](ThreadLocals &tls, int tid)
      {
        UT_VECTOR_APPEND(activeBlocks, tls.activeBlocks);
      },
      0, /*doSerial=*/true);
  }
  UT_ASSERT0((size_t)nActiveBlocks == activeBlocks.size());  // ---- 8-color lock-free splat ---------------------------------------------
  const float distSqMax = rScan * rScan, distSqMaxInv = 1.0f / distSqMax;

  std::vector<int> activeBlocksPerCol[8];
  for (int i = 0; i < (int)activeBlocks.size(); i++)
  {
    int bid = activeBlocks[i];
    Vec4i bpos = sg0->getBlockCoordsById(bid);
    int icol = getBlockColor8(vget_x(bpos), vget_y(bpos), vget_z(bpos));
    activeBlocksPerCol[icol].push_back(bid);
  }

  for (int icol = 0; icol < 8; icol++)
  {
    std::vector<int> *blockList = &activeBlocksPerCol[icol];
    ThrRunParallel<int>(blockList->size(), nullptr,
      [&](int &tls, int tid, int ibid)
      {
        int bid = (*blockList)[ibid];
        Vec4i bpos = sg0->getBlockCoordsById(bid);
        int bx = vget_x(bpos), by = vget_y(bpos), bz = vget_z(bpos);

        Particle *p = NULL;
        for (ParticleIdx ixp = particlesPerBlock[bid]; ixp; ixp = p->idxNext)
        {
          p = getParticle(ixp);
          Vec4f pos0(p->pos[0], p->pos[1], p->pos[2], 0.f);

          Vec4i ipos1 = max(truncate_to_int(ceil(pos0 - rScan - 0.5f)), 0),
                ipos2 = min(truncate_to_int(floor(pos0 + rScan - 0.5f)),
                            sg0->v4iDomMax());
          int ix1 = vget_x(ipos1), ix2 = vget_x(ipos2),
              iy1 = vget_y(ipos1), iy2 = vget_y(ipos2),
              iz1 = vget_z(ipos1), iz2 = vget_z(ipos2);
          Vec4f pshift = pos0 - 0.5f;
          float x0 = vfget_x(pshift), y0 = vfget_y(pshift), z0 = vfget_z(pshift);

          const int bsxLog2 = sg0->bsxLog2(), bsx2Log2 = sg0->bsx2Log2(),
                    bsxMask = sg0->bsx() - 1, nx = sg0->nbx(), nxy = sg0->nbxy();

          for (int iz = iz1; iz <= iz2; iz++)
          {
            float dz = iz - z0;
            int ibz = (iz >> bsxLog2) * nxy, ivz = (iz & bsxMask) << bsx2Log2;
            for (int iy = iy1; iy <= iy2; iy++)
            {
              float dy = iy - y0, distSqZY = dz * dz + dy * dy;
              int ibzy = ibz + (iy >> bsxLog2) * nx,
                  ivzy = ivz + ((iy & bsxMask) << bsxLog2);
              for (int ix = ix1; ix <= ix2; ix++)
              {
                float dx = ix - x0, distSq = distSqZY + dx * dx;
                if (distSq > distSqMax) continue;
                int ib = ibzy + (ix >> bsxLog2),
                    iv = ivzy + (ix & bsxMask);
                SBG::Block<uint16_t> *block = sg0->getBlock(ib);
                if (!sg0->isValueBlock(block)) continue;
                uint16_t val = renderDensFromFloat_<uint16_t>(distSq * distSqMaxInv);
                block->_data[iv] = std::min(block->_data[iv], val);
              }
            }
          }
        }
      });
  }

  FREEMEM(particlesPerBlock);
  FREEMEM(particles);  // ---- Finalize ------------------------------------------------------------
  ThrRunParallel<int>(activeBlocks.size(), nullptr,
    [&](int &tls, int tid, int ibid)
    {
      int bid = activeBlocks[ibid];
      uint16_t *data = sg0->getBlockDataPtr(bid);
      for (int vid = 0; vid < sg0->nVoxelsInBlock(); vid++)
      {
        float distSq = renderDensToFloat_(data[vid]);
        uint16_t uiVal = 0;
        if (distSq < 0.999f)
        {
          distSq *= distSqMax;
          float dist = sqrtf(distSq) - rParticle;
          float f = 1.0f - MT_LINSTEPF(-rParticle, (rParticle + nbDist), dist);
          uiVal = renderDensFromFloat_<uint16_t>(f);
        }
        data[vid] = uiVal;
      }
    });  // ---- Output --------------------------------------------------------------
  printf("%lld %d %llu\n", (long long)nActiveBlocks, (int)msbg->nBlocks(),
         (unsigned long long)nActParticles);

  auto dumpField = [&]()
  {
    for (int bid = 0; bid < msbg->nBlocks(); bid++)
    {
      uint16_t *d = sg0->getBlockDataPtr(bid, 0, 0);
      fwrite(d, sizeof(uint16_t), sg0->nVoxelsInBlock(), stdout);
    }
  };

  // Field A: after splat + finalize.
  dumpField();
  // Field B: after splat + finalize + mean-curvature smoothing.
  msbg->applyChannelPdeFast<uint16_t>(
      chan, CH_NULL, CH_NULL, &activeBlocks,
      -(PDE_MEAN_CURVATURE + OPT_8_COLOR_SCHEME),
      1 /*doConstrZeroOne*/, 0, 0, 0, nIter, 1.0f, dt,
      0, NULL, 0, 0, 0, 0, 0, 0, NULL, 0);  dumpField();
  MSBG::MultiresSparseGrid::destroy(msbg);
  FREEMEM(blockActive);
  return 0;
}

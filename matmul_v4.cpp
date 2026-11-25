#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr int32_t kTilingInts = 256;   // TCubeTiling 字段总大小（int32 计）

// ---- 核间网格固定 4x4 = 16 块，tile 尺寸随问题规模自适应 ----
// 之前固定 tile=256x256 时，tile 数 = (M/256)*(N/256) 随规模线性增长，
// 复用放大 = A 读 N/256 遍、B 读 M/256 遍，4096^3 时流量冲到 1 GB，
// 已接近带宽墙（约 1.26 TB/s），所以大尺寸只跑到官方 62%。
// 改成固定 4x4 后：每核正好 1 块，复用放大恒为 4x。
//   1024^3: 流量 16 MB  （不变）
//   2048^3: 128 MB -> 64 MB
//   4096^3: 1 GB   -> 256 MB
// L0C 不受影响：singleShape 变大时 API 内部按 baseM/baseN 循环，
// L0C 单块仍然是 baseM*baseN*4（auto 配置下 128*256*4 = 128 KB）。
constexpr uint32_t kGrid = 4;
}

// V4：AscendC::MatmulImpl（MDL） + kernel 内自行核间切分。
//
// 支持任意 M/N/K（M/N 需能被 kGrid 整除）。问题尺寸从 tiling 的 M/N/Ka 读出，
// host 按实际尺寸生成 tiling（SetShape/SetOrgShape 传给全量 M/N/K）。
//
// 为什么自己在 kernel 里切核：本 CANN 版本的 MatmulApiTiling 没有 SetDim /
// SetSingleShape / MultiCoreMatmulTiling，host 侧无法设定核数（usedCoreNum 恒为 1）。
extern "C" __global__ __aicore__ void matmul_v4(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    // 官方 API 内部会把 tiling 指针转成普通指针，直接传 __gm__ 指针编译不过。
    // 等价于官方 GET_TILING_DATA：先把 tiling 从 GM 读进核内局部变量再传入。
    AscendC::tiling::TCubeTiling tilingData;
    {
        GlobalTensor<int32_t> tGm;
        tGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tiling), kTilingInts);
        int32_t *dst = reinterpret_cast<int32_t *>(&tilingData);
        for (int32_t i = 0; i < kTilingInts; ++i) {
            dst[i] = tGm.GetValue(i);
        }
    }

    const uint32_t M = static_cast<uint32_t>(tilingData.M);
    const uint32_t N = static_cast<uint32_t>(tilingData.N);
    const uint32_t K = static_cast<uint32_t>(tilingData.Ka);

    const uint32_t tileM = M / kGrid;
    const uint32_t tileN = N / kGrid;
    constexpr uint32_t totalTiles = kGrid * kGrid;

    TPipe pipe;
    AscendC::MatmulImpl<MatmulType<TPosition::GM, CubeFormat::ND, half>,
                        MatmulType<TPosition::GM, CubeFormat::ND, half>,
                        MatmulType<TPosition::GM, CubeFormat::ND, half>,
                        MatmulType<TPosition::GM, CubeFormat::ND, half>> mm;
    mm.Init(&tilingData, &pipe);

    GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), M * K);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), K * N);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), M * N);

    const uint32_t blkNum = AscendC::GetBlockNum();

    // 16 块 / 16 核 = 每核 1 块；blockDim 小于 16 时轮转多块
    for (uint32_t idx = AscendC::GetBlockIdx(); idx < totalTiles; idx += blkNum) {
        const uint32_t mt = idx / kGrid;
        const uint32_t nt = idx % kGrid;

        mm.SetOrgShape(M, N, K);              // 原始矩阵
        mm.SetSingleShape(tileM, tileN, K);   // 本核这块的计算量
        mm.SetTensorA(aGm[mt * tileM * K]);   // A 的第 mt 个 M 带
        mm.SetTensorB(bGm[nt * tileN]);       // B 的第 nt 个 N 带
        mm.IterateAll(cGm[mt * tileM * N + nt * tileN]);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v4_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                  uint8_t *tiling)
{
    matmul_v4<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr int32_t kTilingInts = 256;   // TCubeTiling 字段总大小（int32 计）

// ---- 每核单次计算的输出块尺寸（不随问题规模变）----
// 256x256 在 1024^3 上实测最优；M/N 变大时改为"每核循环处理多个 tile"。
constexpr uint32_t kTileM = 256;
constexpr uint32_t kTileN = 256;
}

// V4：AscendC::MatmulImpl（MDL） + kernel 内自行核间切分。
//
// 支持任意 M/N/K（M/N 需为 256 的倍数，K 需满足 tiling 的对齐要求）：
//   - 问题尺寸从 tiling 的 M/N/Ka 读出（host 按实际尺寸生成 tiling）
//   - 输出空间切成 (M/256) x (N/256) 个 tile，各核按 idx += blockDim 轮转领取
//   - 每个 tile 用 SetOrgShape + SetSingleShape 声明"原始矩阵"与"本核计算量"
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

    const uint32_t gridN = N / kTileN;
    const uint32_t gridM = M / kTileM;
    const uint32_t totalTiles = gridM * gridN;
    const uint32_t blkNum = AscendC::GetBlockNum();

    // 各核轮转领取 tile；1024^3 时 16 tile / 16 核 = 每核 1 块（与之前一致），
    // 2048^3 时 64/16 = 每核 4 块，4096^3 时 256/16 = 每核 16 块。
    for (uint32_t idx = AscendC::GetBlockIdx(); idx < totalTiles; idx += blkNum) {
        const uint32_t mt = idx / gridN;
        const uint32_t nt = idx % gridN;

        mm.SetOrgShape(M, N, K);                 // 原始矩阵
        mm.SetSingleShape(kTileM, kTileN, K);    // 本核这块的计算量
        mm.SetTensorA(aGm[mt * kTileM * K]);     // A 的第 mt 个 M 带
        mm.SetTensorB(bGm[nt * kTileN]);         // B 的第 nt 个 N 带
        mm.IterateAll(cGm[mt * kTileM * N + nt * kTileN]);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v4_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                  uint8_t *tiling)
{
    matmul_v4<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
#endif

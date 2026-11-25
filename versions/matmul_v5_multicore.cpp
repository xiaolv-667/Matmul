#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t M = 1024, N = 1024, K = 1024;
constexpr int32_t kTilingInts = 256;   // TCubeTiling 字段总大小（int32 计）

// ---- 核间切分参数（本版由 kernel 自行接管，不依赖 tiling 生成器）----
constexpr uint32_t kTileM = 256;              // 每核负责的输出块 M
constexpr uint32_t kTileN = 256;              // 每核负责的输出块 N
constexpr uint32_t kGridN = N / kTileN;       // 4 -> 4x4 = 16 核
constexpr uint32_t kGridM = M / kTileM;
}

// V4：AscendC::MatmulImpl（MDL 实现） + kernel 内自行核间切分。
//
// 为什么自己在 kernel 里切核：
//   本 CANN 版本的 MatmulApiTiling 没有 SetDim / SetSingleShape /
//   MultiCoreMatmulTiling，host 侧无法设定核数（usedCoreNum 恒为 1）。
//   实测 SetShape(256,256,1024) 只让 singleCoreM/N 变小、核数不变，
//   手动覆盖 usedCoreNum=16 则 16 个核都算同一块（cos=0.2518=1/sqrt(16)）。
// 做法：
//   tiling 仍按单核生成（参数正确），kernel 里用 GetBlockIdx 算出本核的
//   (mt, nt) 子块，再通过 MatmulImplBase 的运行期 SetOrgShape/SetSingleShape
//   声明"原始矩阵"与"本核计算量"，并把 A/B/C 的 GlobalTensor 偏移到子块起点。
extern "C" __global__ __aicore__ void matmul_v4(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using A_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using B_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using C_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;

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

    TPipe pipe;
    AscendC::MatmulImpl<A_T, B_T, C_T, C_T> mm;
    mm.Init(&tilingData, &pipe);

    // ---- 自行核间切分 ----
    const uint32_t blk = AscendC::GetBlockIdx();
    const uint32_t mt = blk / kGridN;
    const uint32_t nt = blk % kGridN;
    if (mt >= kGridM) {
        return;   // blockDim 超出实际子块数时，多余核直接退出
    }

    GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), M * K);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), K * N);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), M * N);

    // 原始矩阵形状（用于地址换算）
    mm.SetOrgShape(M, N, K);
    // 本核计算量：256 x 256，K 方向全量累加
    mm.SetSingleShape(kTileM, kTileN, K);

    // A 取第 mt 个 M 带（A 是 [M,K]，行偏移 = mt*kTileM*K）
    // B 取第 nt 个 N 带（B 是 [K,N]，列偏移 = nt*kTileN）
    // C 写到 (mt, nt) 子块起点
    mm.SetTensorA(aGm[mt * kTileM * K]);
    mm.SetTensorB(bGm[nt * kTileN]);
    mm.IterateAll(cGm[mt * kTileM * N + nt * kTileN]);
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v4_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                  uint8_t *tiling)
{
    matmul_v4<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
#endif

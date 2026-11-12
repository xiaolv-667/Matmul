#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t M = 1024, N = 1024, K = 1024;
constexpr int32_t kTilingInts = 256;   // TCubeTiling 字段总大小（int32 计）
}

// V4：AscendC::MatmulImpl（MDL 实现，非 KFC）。
//
// 为什么不用官方别名 AscendC::Matmul：
//   在 __NPU_ARCH__==2201(910B) 上它映射到 KFC 模式的 MatmulClient，
//   依赖 AIC(Server) + AIV(Client) 握手以及 system workspace。
//   实测无论声明 KERNEL_TYPE_AIC_ONLY 还是 mix，进程都会死锁（timeout 124）。
// 直接用 MatmulImpl + Init(&tiling, &pipe) 走 MDL 路径：无核间握手、不需要 workspace。
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

    GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), M * K);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), K * N);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), M * N);

    mm.SetTensorA(aGm);
    mm.SetTensorB(bGm);
    mm.IterateAll(cGm);
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v4_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                  uint8_t *tiling)
{
    matmul_v4<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
#endif

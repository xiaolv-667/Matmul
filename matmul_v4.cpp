#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t M = 1024, N = 1024, K = 1024;
}

// V4：改用 AscendC::Matmul 高阶 API。
// 手搓 Mmad/LoadData/Fixpipe 的路线已归档（Fixpipe 输出环节只写入 6.2%，
// 卡在 mSize/srcStride 的硬件语义上，本地无法验证）。官方 API 内部封装了
// 分形转换、L1/L0 多级缓冲和 Fixpipe 参数，只要求外部提供正确的 tiling。
extern "C" __global__ __aicore__ void matmul_v4(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    // 走 Cube 单元，声明为 AIC 算子
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using A_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using B_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using C_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;

    TPipe pipe;
    Matmul<A_T, B_T, C_T, C_T> mm;
    // tiling 由 host 侧填好放在 GM，字段布局与 kernel_tiling.h 的 TCubeTiling 一致。
    // AI core 上 GM 指针必须带 __gm__ 地址空间限定，且类型在 AscendC::tiling 命名空间内。
    mm.Init(reinterpret_cast<__gm__ AscendC::tiling::TCubeTiling *>(tiling));

    GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), M * K);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), K * N);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), M * N);

    mm.SetTensorA(aGm);
    mm.SetTensorB(bGm);
    mm.IterateAll(cGm);
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v4_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *tiling)
{
    matmul_v4<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
#endif

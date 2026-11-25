#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;   // 行数
constexpr uint32_t N = 1024;   // 每行长度

// ---- UB 地址布局（字节）----
constexpr uint32_t XH_ADDR = 0;                     // half[N]  2048 B
constexpr uint32_t YH_ADDR = XH_ADDR + N * 2;       // half[N]  2048 B
constexpr uint32_t XF_ADDR = YH_ADDR + N * 2;       // float[N] 4096 B
// 共 8 KB，UB 有 192 KB，余量充足
}

// ===================== softmax v0：朴素实现 =====================
// 最朴素的做法：单核、逐行处理、**三遍独立遍历**，每遍都从 GM 重新读原始数据。
//   pass1: 求本行最大值 rowMax
//   pass2: 重读 → exp(x-rowMax) → 求和 rowSum
//   pass3: 再重读 → exp(x-rowMax) → 除以 rowSum → 写回
//
// 这份代码的意义是提供一个"什么都没优化"的基线，代价一眼可见：
//   - GM 流量 = 3 次读 + 1 次写 = 8 MB（理论下限是 1 读 1 写 = 4 MB）
//   - exp 被算了两遍（pass2 / pass3 各一次）
//   - 单核，40 个 AIV 只用了 1 个
//   - 每遍之间 SetFlag/WaitFlag 全串行，搬运与计算零重叠
//
// 关于归约：本来想直接用 AscendC 的 Level-2 ReduceMax/ReduceSum，但在这套环境下
// 无论怎么给 sharedTmpBuffer 都会触发 AIV 越界异常
// （plog: `[vector core exception] ... retCode=0x31`，aclError 507035）。
// 隔离测试确认去掉归约后 kernel 正常，所以这里改为**显式树形归约**，
// 只使用 Add / Max 这类基础原语：每次把后半段并到前半段，log2(N) 步完成。
// 归约就地破坏 xF 不影响正确性——每一遍都会重新从 GM 读并重新计算。
namespace {
// 行内树形归约求最大值，结果留在 src[0]
__aicore__ inline void RowMax(AscendC::LocalTensor<float> &src)
{
    for (uint32_t stride = N / 2; stride >= 1; stride >>= 1) {
        AscendC::Max(src, src, src[stride], static_cast<int32_t>(stride));
    }
}

// 行内树形归约求和，结果留在 src[0]
__aicore__ inline void RowSum(AscendC::LocalTensor<float> &src)
{
    for (uint32_t stride = N / 2; stride >= 1; stride >>= 1) {
        AscendC::Add(src, src, src[stride], static_cast<int32_t>(stride));
    }
}
}

extern "C" __global__ __aicore__ void softmax_v0(GM_ADDR x, GM_ADDR y)
{
    // 纯 Vector 算子，必须显式声明为 AIV-only。
    // 否则工具链会编成 mix 核（每块 1 AIC + 2 AIV），blockDim 语义翻倍。
    // 这个坑在 Matmul 项目里踩过（导致 1.6x 性能回退），这里提前避开。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::InitSocState();

    AscendC::GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, M * N);
    yGm.SetGlobalBuffer((__gm__ half *)y, M * N);

    AscendC::LocalTensor<half>  xHalf(TPosition::VECCALC, XH_ADDR, N);
    AscendC::LocalTensor<half>  yHalf(TPosition::VECCALC, YH_ADDR, N);
    AscendC::LocalTensor<float> xF   (TPosition::VECCALC, XF_ADDR, N);

    for (uint32_t i = 0; i < M; ++i) {
        // ---------------- 第 1 遍：求本行最大值 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        RowMax(xF);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float rowMax = xF.GetValue(0);

        // ---------------- 第 2 遍：exp(x - max) 并求和 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);      // 重读 GM
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::Adds(xF, xF, -rowMax, N);
        AscendC::Exp(xF, xF, N);
        RowSum(xF);                                   // 就地求和，破坏 xF 无妨
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float rowSum = xF.GetValue(0);

        // ---------------- 第 3 遍：归一化并写回 ----------------
        AscendC::DataCopy(xHalf, xGm[i * N], N);      // 第三次重读
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(xF, xHalf, AscendC::RoundMode::CAST_NONE, N);
        AscendC::Adds(xF, xF, -rowMax, N);
        AscendC::Exp(xF, xF, N);                      // exp 第二次计算
        AscendC::Muls(xF, xF, 1.0f / rowSum, N);
        AscendC::Cast(yHalf, xF, AscendC::RoundMode::CAST_NONE, N);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGm[i * N], yHalf, N);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_v0_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    softmax_v0<<<blockDim, nullptr, stream>>>(x, y);
}
#endif

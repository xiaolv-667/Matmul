#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;
constexpr uint32_t N = 1024;
constexpr uint32_t K = 1024;

// ================== v2 步骤一：加大单次搬运粒度 ==================
// v0：每次 DataCopy 只搬 1 行 B（2 KB）→ MTE2 单次约 129 ns，
//     其中固定延迟（约 109 ns）完全摊不掉，活跃带宽只有 ~16 GB/s。
// v2：一次搬 Kb=16 行（32 KB）→ 固定延迟摊到 16 行上，理论活跃带宽 ~60 GB/s。
//
// 注意：本版本仍是"单 buffer + 全串行"，和 v0 的同步结构完全一致
// （每个 SetFlag 后面紧跟配对的 WaitFlag），所以没有引入新的同步风险。
// 双缓冲（ping-pong）会作为步骤二单独加，避免一次改两个变量。
//
// 参数化 Kb 便于后续扫参：1 / 4 / 8 / 16 / 32
constexpr uint32_t Kb = 16;
constexpr uint32_t NK_BLOCK = K / Kb;   // K 方向分块数，Kb=16 时为 64

constexpr uint32_t HALF_BYTES  = 2;
constexpr uint32_t FLOAT_BYTES = 4;

// ---- UB 地址布局（字节）----
// 单 buffer 时 B 只占 32 KB，总计 48 KB / 192 KB
constexpr uint32_t B0_ADDR  = 0;
constexpr uint32_t BF_ADDR  = B0_ADDR + Kb * N * HALF_BYTES;
constexpr uint32_t TMP_ADDR = BF_ADDR + N * FLOAT_BYTES;
constexpr uint32_t ACC_ADDR = TMP_ADDR + N * FLOAT_BYTES;
constexpr uint32_t CH_ADDR  = ACC_ADDR + N * FLOAT_BYTES;
constexpr uint32_t AH_ADDR  = CH_ADDR + N * HALF_BYTES;
}

extern "C" __global__ __aicore__ void matmul_v2(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    AscendC::InitSocState();

    AscendC::GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer((__gm__ half *)a, M * K);
    bGm.SetGlobalBuffer((__gm__ half *)b, K * N);
    cGm.SetGlobalBuffer((__gm__ half *)c, M * N);

    AscendC::LocalTensor<half>  aHalf (TPosition::VECCALC, AH_ADDR, K);
    AscendC::LocalTensor<half>  bBuf  (TPosition::VECCALC, B0_ADDR, Kb * N);
    AscendC::LocalTensor<half>  cHalf (TPosition::VECCALC, CH_ADDR, N);
    AscendC::LocalTensor<float> bFloat(TPosition::VECCALC, BF_ADDR, N);
    AscendC::LocalTensor<float> tmp   (TPosition::VECCALC, TMP_ADDR, N);
    AscendC::LocalTensor<float> acc   (TPosition::VECCALC, ACC_ADDR, N);

    const uint32_t blk    = AscendC::GetBlockIdx();   // 当前使用的 core
    const uint32_t blkNum = AscendC::GetBlockNum();   // 获取全部 core
    const uint32_t rowBegin = blk * M / blkNum;
    const uint32_t rowEnd   = (blk + 1) * M / blkNum; // 数据平均分配给每个核
    // rowBegin -> rowEnd，该核处理的所有数据

    for (uint32_t i = rowBegin; i < rowEnd; ++i) {
        AscendC::DataCopy(aHalf, aGm[i * K], K);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Duplicate(acc, 0.0f, N);

        // 外层按 K 方向分块，一次搬 Kb 行 B（32 KB）进来
        for (uint32_t kb = 0; kb < NK_BLOCK; ++kb) {
            AscendC::DataCopy(bBuf, bGm[kb * Kb * N], Kb * N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // 计算这一块的 Kb 行：acc += a[i, kb*Kb+kk] * b[kb*Kb+kk, :]
            for (uint32_t kk = 0; kk < Kb; ++kk) {
                const float av = (float)aHalf.GetValue(kb * Kb + kk);
                AscendC::Cast(bFloat, bBuf[kk * N], AscendC::RoundMode::CAST_NONE, N);
                AscendC::Muls(tmp, bFloat, av, N);
                AscendC::Add(acc, acc, tmp, N);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }

        AscendC::Cast(cHalf, acc, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(cGm[i * N], cHalf, N);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v2_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c)
{
    // blockDim：核数
    // a,b,c 三个张量
    matmul_v2<<<blockDim, nullptr, stream>>>(a, b, c);
}
#endif

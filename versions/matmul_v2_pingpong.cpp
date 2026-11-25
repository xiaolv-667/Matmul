#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;
constexpr uint32_t N = 1024;
constexpr uint32_t K = 1024;

// ================== v2：分块搬运 + 双缓冲 ==================
// 改动一：Kb —— 一次 DataCopy 搬多少行 B
//   1 行（2 KB）时 MTE2 单次约 129 ns，其中固定延迟约 109 ns 摊不掉，
//   活跃带宽只有 ~16 GB/s；16 行（32 KB）时固定延迟被摊薄。
constexpr uint32_t Kb = 16;
constexpr uint32_t NK_BLOCK = K / Kb;   // 64

// 改动二：双缓冲（ping-pong），采用"预取 2 块 + 自己回填自己"的结构：
//
//   预取: buf0 <- chunk0, buf1 <- chunk1
//   第 kb 轮:  等 buf[cur] 到齐 -> 算 buf[cur] -> 释放 buf[cur]
//              -> 立即等自己释放 -> 把 chunk(kb+2) 填回 buf[cur]
//
// 这样安排的好处：WaitFlag<V_MTE2> 永远紧跟在自己的 SetFlag 后面，同一条指令流内
// 成对出现，不存在"等一个上一轮设下的释放旗标"这种跨迭代依赖——那正是手写 ping-pong
// 最容易死锁的地方。
//
// 事件号：
//   EVENT_ID0 / EVENT_ID1 -> B 的双 buffer（cur==0 用 0，cur==1 用 1）
//   EVENT_ID2             -> A 行搬运
//   EVENT_ID3             -> C 写回
// 注意：EVENT_ID* 是全局 event_t 枚举，不在 AscendC 命名空间里，必须裸写。
// ==========================================================

constexpr uint32_t HALF_BYTES  = 2;
constexpr uint32_t FLOAT_BYTES = 4;

// ---- UB 地址布局（字节），总计 80 KB / 192 KB ----
constexpr uint32_t B0_ADDR  = 0;
constexpr uint32_t B1_ADDR  = B0_ADDR + Kb * N * HALF_BYTES;
constexpr uint32_t BF_ADDR  = B1_ADDR + Kb * N * HALF_BYTES;
constexpr uint32_t TMP_ADDR = BF_ADDR + N * FLOAT_BYTES;
constexpr uint32_t ACC_ADDR = TMP_ADDR + N * FLOAT_BYTES;
constexpr uint32_t CH_ADDR  = ACC_ADDR + N * FLOAT_BYTES;
constexpr uint32_t AH_ADDR  = CH_ADDR + N * HALF_BYTES;
}

extern "C" __global__ __aicore__ void matmul_v2(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    // 本算子只用到 Vector 单元，显式声明为纯 AIV 算子。
    // 不加这一句，工具链会把它编成 mix 核（1 AIC + 2 AIV 每块），
    // blockDim=40 就变成 80 个 AIV 任务压到 40 个物理 AIV 上，实测慢 1.6 倍。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::InitSocState();

    AscendC::GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer((__gm__ half *)a, M * K);
    bGm.SetGlobalBuffer((__gm__ half *)b, K * N);
    cGm.SetGlobalBuffer((__gm__ half *)c, M * N);

    AscendC::LocalTensor<half>  aHalf (TPosition::VECCALC, AH_ADDR, K);
    AscendC::LocalTensor<half>  bBuf0 (TPosition::VECCALC, B0_ADDR, Kb * N);
    AscendC::LocalTensor<half>  bBuf1 (TPosition::VECCALC, B1_ADDR, Kb * N);
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
        // ---- A 行搬入：一次 2 KB，不是瓶颈，用独立事件号 ----
        AscendC::DataCopy(aHalf, aGm[i * K], K);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);

        AscendC::Duplicate(acc, 0.0f, N);

        // ---- 预取前两块 ----
        AscendC::DataCopy(bBuf0, bGm[0], Kb * N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::DataCopy(bBuf1, bGm[Kb * N], Kb * N);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        for (uint32_t kb = 0; kb < NK_BLOCK; ++kb) {
            const uint32_t cur = kb & 1u;   // 0 = buf0, 1 = buf1
            AscendC::LocalTensor<half> bCur = (cur == 0u) ? bBuf0 : bBuf1;

            // 等当前块到齐（本块的搬运在上一轮/预取时已发起）
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                (cur == 0u) ? EVENT_ID0 : EVENT_ID1);

            // 计算这一块的 Kb 行：acc += a[i, kb*Kb+kk] * b[kb*Kb+kk, :]
            for (uint32_t kk = 0; kk < Kb; ++kk) {
                const float av = (float)aHalf.GetValue(kb * Kb + kk);
                AscendC::Cast(bFloat, bCur[kk * N], AscendC::RoundMode::CAST_NONE, N);
                AscendC::Muls(tmp, bFloat, av, N);
                AscendC::Add(acc, acc, tmp, N);
            }

            // 本块已用完，释放
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                (cur == 0u) ? EVENT_ID0 : EVENT_ID1);

            // 把 chunk(kb+2) 填回刚用完的这块 buffer。
            // 等待紧跟在自己上面的 Set，同一指令流内成对，不会有跨迭代死锁。
            if (kb + 2 < NK_BLOCK) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    (cur == 0u) ? EVENT_ID0 : EVENT_ID1);
                AscendC::DataCopy(bCur, bGm[(kb + 2) * Kb * N], Kb * N);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                    (cur == 0u) ? EVENT_ID0 : EVENT_ID1);
            }
        }

        // 收尾：最后两块没有再回填，它们各留下一个 V_MTE2 旗标未消费。
        // 这里各消费一次，保证跨行迭代时旗标计数严格配平。
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

        AscendC::Cast(cHalf, acc, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::DataCopy(cGm[i * N], cHalf, N);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
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

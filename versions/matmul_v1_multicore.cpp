#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024;
constexpr uint32_t N = 1024;
constexpr uint32_t K = 1024;

constexpr uint32_t A_ADDR  = 0;
constexpr uint32_t BH_ADDR = A_ADDR  + K * sizeof(half);
constexpr uint32_t CH_ADDR = BH_ADDR + N * sizeof(half);
constexpr uint32_t BF_ADDR = CH_ADDR + N * sizeof(half);
constexpr uint32_t TMP_ADDR = BF_ADDR + N * sizeof(float);
constexpr uint32_t ACC_ADDR = TMP_ADDR + N * sizeof(float);
}

extern "C" __global__ __aicore__ void matmul_v0(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    AscendC::InitSocState();

    AscendC::GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer((__gm__ half *)a, M * K);
    bGm.SetGlobalBuffer((__gm__ half *)b, K * N);
    cGm.SetGlobalBuffer((__gm__ half *)c, M * N);

    AscendC::LocalTensor<half>  aHalf (TPosition::VECCALC, A_ADDR,  K);
    AscendC::LocalTensor<half>  bHalf (TPosition::VECCALC, BH_ADDR, N);
    AscendC::LocalTensor<half>  cHalf (TPosition::VECCALC, CH_ADDR, N);
    AscendC::LocalTensor<float> bFloat(TPosition::VECCALC, BF_ADDR, N);
    AscendC::LocalTensor<float> tmp   (TPosition::VECCALC, TMP_ADDR, N);
    AscendC::LocalTensor<float> acc   (TPosition::VECCALC, ACC_ADDR, N);

    const uint32_t blk    = AscendC::GetBlockIdx();//当前使用的core
    const uint32_t blkNum = AscendC::GetBlockNum();//获取全部core
    const uint32_t rowBegin = blk * M / blkNum;
    const uint32_t rowEnd   = (blk + 1) * M / blkNum;//数据平均分配给每个核
    //rowBegin->rowEnd，该核处理的所有数据

    for (uint32_t i = rowBegin; i < rowEnd; ++i) {
        AscendC::DataCopy(aHalf, aGm[i * K], K);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Duplicate(acc, 0.0f, N);

        for (uint32_t k = 0; k < K; ++k) {
            const float av = (float)aHalf.GetValue(k);

            AscendC::DataCopy(bHalf, bGm[k * N], N);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Cast(bFloat, bHalf, AscendC::RoundMode::CAST_NONE, N);
            AscendC::Muls(tmp, bFloat, av, N);
            AscendC::Add(acc, acc, tmp, N);

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
void matmul_v0_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c)
{
    //blockDim：核数
    //a,b,c 三个张量
    matmul_v0<<<blockDim, nullptr, stream>>>(a, b, c);
}
#endif
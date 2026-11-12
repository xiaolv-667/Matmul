#include "kernel_operator.h"

using AscendC::TPosition;

namespace {
constexpr uint32_t M = 1024, N = 1024, K = 1024;

// ---- tiling：本版先固定 128x128x128，整除 1024，无尾块 ----
constexpr uint32_t BM = 128;   // baseM：一次 Mmad 的 M
constexpr uint32_t BN = 128;   // baseN：一次 Mmad 的 N
constexpr uint32_t BK = 128;   // baseK：一次 Mmad 的 K
constexpr uint32_t MT = M / BM;   // 8
constexpr uint32_t NT = N / BN;   // 8
constexpr uint32_t KT = K / BK;   // 8
constexpr uint32_t TILES = MT * NT;   // 64 个输出 tile，按核轮转分配

// L1 = 512KB：A1 32KB + B1 32KB，远没用满（留给后续双缓冲）
constexpr uint32_t A1_ADDR = 0;
constexpr uint32_t B1_ADDR = A1_ADDR + BM * BK * 2;
// L0A/L0B/L0C 是独立存储，地址各自从 0 起
}

extern "C" __global__ __aicore__ void matmul_v4(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    // 本版走 Cube 单元，必须声明为 AIC 算子。
    // 注意：不能像 v2/v3 那样写 AIV_ONLY，Mmad/LoadData(load_cbuf_to_ca) 只会在 AIC 上执行。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    AscendC::InitSocState();

    AscendC::GlobalTensor<half> aGm, bGm, cGm;
    aGm.SetGlobalBuffer((__gm__ half *)a, M * K);
    bGm.SetGlobalBuffer((__gm__ half *)b, K * N);
    cGm.SetGlobalBuffer((__gm__ half *)c, M * N);

    AscendC::LocalTensor<half>  a1(TPosition::A1, A1_ADDR, BM * BK);   // L1
    AscendC::LocalTensor<half>  b1(TPosition::B1, B1_ADDR, BK * BN);   // L1
    AscendC::LocalTensor<half>  a2(TPosition::A2, 0, BM * BK);         // L0A
    AscendC::LocalTensor<half>  b2(TPosition::B2, 0, BK * BN);         // L0B
    AscendC::LocalTensor<float> c1(TPosition::CO1, 0, BM * BN);        // L0C (fp32)

    const uint32_t blk    = AscendC::GetBlockIdx();
    const uint32_t blkNum = AscendC::GetBlockNum();

    // ---- GM(ND 行主序) -> L1(NZ 分形) ----
    // A: 源是 [M,K] 行主序，取 [BM,BK] 子块，行内连续，行间距 K
    AscendC::Nd2NzParams ndA;
    ndA.ndNum = 1;
    ndA.nValue = BM;
    ndA.dValue = BK;
    ndA.srcNdMatrixStride = 0;
    ndA.srcDValue = K;
    ndA.dstNzC0Stride = BM;
    ndA.dstNzNStride = 1;
    ndA.dstNzMatrixStride = 0;

    // B: 源是 [K,N] 行主序，取 [BK,BN] 子块。Cube 需要的是 B 的转置分形，
    //    这里先按原样搬进 L1，转置交给 LoadData 的 ifTranspose。
    AscendC::Nd2NzParams ndB;
    ndB.ndNum = 1;
    ndB.nValue = BK;
    ndB.dValue = BN;
    ndB.srcNdMatrixStride = 0;
    ndB.srcDValue = N;
    ndB.dstNzC0Stride = BK;
    ndB.dstNzNStride = 1;
    ndB.dstNzMatrixStride = 0;

    // L1 -> L0A：不转置
    AscendC::LoadData2DParams pA;
    pA.startIndex = 0;
    pA.repeatTimes = BM / 16;
    pA.srcStride = 1;
    pA.sid = 0;
    pA.dstGap = 0;
    pA.ifTranspose = false;
    pA.addrMode = 0;

    // L1 -> L0B：转置（B 在 L1 是 [BK,BN]，L0B 要 [BN,BK] 分形）
    AscendC::LoadData2DParams pB;
    pB.startIndex = 0;
    pB.repeatTimes = BK / 16;
    pB.srcStride = 1;
    pB.sid = 0;
    pB.dstGap = 0;
    pB.ifTranspose = true;
    pB.addrMode = 0;

    for (uint32_t t = blk; t < TILES; t += blkNum) {
        const uint32_t mt = t / NT;
        const uint32_t nt = t % NT;

        for (uint32_t kt = 0; kt < KT; ++kt) {
            // 1) GM -> L1（ND 转 NZ）
            AscendC::DataCopy(a1, aGm[(mt * BM) * K + kt * BK], ndA);
            AscendC::DataCopy(b1, bGm[(kt * BK) * N + nt * BN], ndB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

            // 2) L1 -> L0A / L0B
            AscendC::LoadData(a2, a1, pA);
            AscendC::LoadData(b2, b1, pB);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

            // 3) Mmad：L0A x L0B -> L0C，K 方向在 L0C 里累加
            AscendC::MmadParams mp;
            mp.m = BM;
            mp.n = BN;
            mp.k = BK;
            mp.unitFlag = 0;
            mp.cmatrixInitVal = (kt == 0);   // 第一个 K 块清零，之后累加
            AscendC::Mmad(c1, a2, b2, mp);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        }

        // 4) L0C -> GM，顺便 fp32 转 fp16
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = BN;
        fp.mSize = BM;
        fp.srcStride = 1;
        // dstStride 单位是 32B 块，且是相邻 burst（行）之间的"间隔"，实现里会再补 burstLen。
        //
        // TODO（当前唯一阻塞点）：L0C 是 NZ 分形，C 在 GM 是行主序，两者布局不一致。
        //   内部实现里靠 intriParams.nz2ndParams.nz2ndEn 开关（开启后 dstStride 单位变成元素、
        //   并读 originalNSize），但用户侧 FixpipeParamsV220 没有 nz2ndParams 子结构，
        //   只暴露了 ndNum / srcNdStride / dstNdStride。下一步应参照
        //   impl/adv_api/detail/matmul/stage/copy/ 的 store 阶段，照抄官方构造方式。
        fp.dstStride = (N - BN) * 2 / 32;
        fp.reluEn = false;
        fp.quantPre = QuantMode_t::F322F16;   // 全局枚举，不在 AscendC 命名空间内
        fp.reluEn = false;
        fp.quantPre = QuantMode_t::F322F16;   // 全局枚举，不在 AscendC 命名空间内
        AscendC::Fixpipe(cGm[(mt * BM) * N + nt * BN], c1, fp);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_v4_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c)
{
    matmul_v4<<<blockDim, nullptr, stream>>>(a, b, c);
}
#endif

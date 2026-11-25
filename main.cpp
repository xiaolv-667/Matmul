#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"

#include <cstdlib>

#include "adv_api/matmul/matmul_tiling.h"
#include "adv_api/kernel_tiling.h"
#include "utils/tiling/platform/platform_ascendc.h"

extern void matmul_v4_do(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

namespace {
// 用官方 host 侧 tiling 生成器算 TCubeTiling。
// 手填结构实测精度不过（fro_err≈0.97），官方生成器会自动决定 baseM/baseN/baseK、
// L1/L0 深度，并处理 M/N 不能被核数整除时的尾块切分。
void ComputeTiling(AscendC::tiling::TCubeTiling &tiling)
{
    platform_ascendc::PlatformAscendC *plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MatmulApiTiling api(*plat);
    api.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT16, false);
    api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT16, false);
    api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT16);
    // 多核尝试结论（已实测，失败）：
    //   1) SetShape(256,256,1024) -> singleCoreM/N 变 256，但 usedCoreNum 仍为 1，
    //      只算 1/16 输出，cos=0.2518（正好 = 1/sqrt(16)）。
    //   2) 再手动 tiling.usedCoreNum=16 -> 仍是 cos=0.2518，说明 16 个核算的是同一块。
    // 即本 CANN 版本的 MatmulApiTiling 不支持 host 侧核数切分（无 SetDim/SetSingleShape）。
    // 因此回到单核形状，保证结果正确。
    api.SetShape(1024, 1024, 1024);
    api.SetOrgShape(1024, 1024, 1024);
    api.SetBias(false);
    // L1 的 A/B 双缓冲
    api.SetDoubleBuffer(true, true, false, false);
    // ---- base / 缓冲 扫描结论（16 核，全部实测）----
    //   auto 128/256/64, db=(T,T,F,F)  : 24.377 us / 88.10 TFLOPS   <-- 最优，采用
    //   fix  128/128/128, db=(T,T,F,F) : 27.523 us / 78.03 TFLOPS
    //   fix  128/128/64,  db=(T,T,T,F) : 27.483 us / 78.14 TFLOPS（L0C 双缓冲实测为负收益）
    //   单核时 auto 198.4us/10.83 vs fix128^3 237.6us/9.04
    // 注意：SetFixSplit 的 baseK 会被生成器钳制（传 64 实际给 128），
    //       所以 baseK 并非自由变量，这也解释了 base 扫描为何难有正收益。
    // 结论：交回生成器自动选择。
    api.SetDoubleBuffer(true, true, false, false);
    api.GetTiling(tiling);
}

// 从环境变量读正整数，读不到或非法则用默认值
int EnvInt(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    const int n = atoi(v);
    return (n > 0) ? n : fallback;
}
}

int32_t main(int32_t argc, char **argv)
{
    // blockDim 由命令行第 1 个参数指定，默认 1（保持 v0 原始行为）
    uint32_t blockDim = 1;
    if (argc > 1) {
        const int v = atoi(argv[1]);
        if (v > 0) {
            blockDim = static_cast<uint32_t>(v);
        }
    }
    // 重复次数 / 预热次数走环境变量，方便脚本扫描而不必重编
    const int repeat = EnvInt("BENCH_REPEAT", 100);
    const int warmup = EnvInt("BENCH_WARMUP", 5);

    const uint32_t M = 1024, N = 1024, K = 1024; // 数据处理的规模 M*N*K 规模的数据乘

    size_t aBytes = (size_t)M * K * 2; // 数据类型 fp16
    size_t bBytes = (size_t)K * N * 2;
    size_t cBytes = (size_t)M * N * 2;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *aHost = nullptr, *bHost = nullptr, *cHost = nullptr;
    CHECK_ACL(aclrtMallocHost((void **)&aHost, aBytes));
    CHECK_ACL(aclrtMallocHost((void **)&bHost, bBytes));
    CHECK_ACL(aclrtMallocHost((void **)&cHost, cBytes));
    ReadFile("./input/a.bin", aBytes, aHost, aBytes); // 加载矩阵规模
    ReadFile("./input/b.bin", bBytes, bHost, bBytes);

    uint8_t *aDev = nullptr, *bDev = nullptr, *cDev = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&aDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&bDev, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&cDev, cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    // 诊断用：把 C 显存清零后再跑。
    // 结果若全 0 -> kernal 根本没写进来；若仍是 NaN -> 写进来了但值是错的。
    CHECK_ACL(aclrtMemset(cDev, cBytes, 0, cBytes));
    // 拷贝到设备端
    CHECK_ACL(aclrtMemcpy(aDev, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDev, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- tiling：官方生成器算好，拷到 device，作为内核第 4 个参数 ----
    constexpr size_t kTilingBytes = 1024;   // 留足余量，超过 sizeof(TCubeTiling)
    uint8_t *tilingDev = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&tilingDev, kTilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(tilingDev, kTilingBytes, 0, kTilingBytes));
    AscendC::tiling::TCubeTiling cubeTiling;
    ComputeTiling(cubeTiling);
    CHECK_ACL(aclrtMemcpy(tilingDev, kTilingBytes, &cubeTiling, sizeof(cubeTiling),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    // 核数以 kernel 侧约定为准：4x4 = 16 核（tiling 生成器在本版本不支持设核数）
    blockDim = 16;
    printf("[TILING] usedCoreNum=%d baseM=%d baseN=%d baseK=%d singleCoreM=%d singleCoreN=%d "
           "depthA1=%d depthB1=%d stepM=%d stepN=%d\n",
           cubeTiling.usedCoreNum, cubeTiling.baseM, cubeTiling.baseN, cubeTiling.baseK,
           cubeTiling.singleCoreM, cubeTiling.singleCoreN, cubeTiling.depthA1, cubeTiling.depthB1,
           cubeTiling.stepM, cubeTiling.stepN);

    aclrtEvent startEv = nullptr, endEv = nullptr;
    CHECK_ACL(aclrtCreateEvent(&startEv));
    CHECK_ACL(aclrtCreateEvent(&endEv));

    // 预热：首次 launch 会包含 kernel 库加载、UB/栈初始化等一次性开销，
    // 这些开销计入平均值会污染结果，尤其当 kernel 优化到 10us 量级时。
    for (int i = 0; i < warmup; ++i) {
        matmul_v4_do(blockDim, stream, aDev, bDev, cDev, tilingDev);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 计时：用 event 夹住连续 repeat 次 launch。
    // 不用 host 侧 wall clock，是因为 kernel launch 是异步的，host 计时会把
    // launch 开销算进去；连续发射再取平均可以把这部分摊薄。
    CHECK_ACL(aclrtRecordEvent(startEv, stream));
    for (int it = 0; it < repeat; ++it) {
        matmul_v4_do(blockDim, stream, aDev, bDev, cDev, tilingDev);
    }
    CHECK_ACL(aclrtRecordEvent(endEv, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream)); // 只同步一次

    float totalMs = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&totalMs, startEv, endEv));
    const double avgMs = static_cast<double>(totalMs) / repeat;
    // 2*M*N*K 次浮点运算；GFLOP/ms 在数值上等于 TFLOPS
    const double gflop = 2.0 * M * N * K / 1e9;
    const double tflops = (avgMs > 0.0) ? gflop / avgMs : 0.0;

    printf("[BENCH] block=%u repeat=%d warmup=%d avg_ms=%.6f total_ms=%.3f gflop=%.3f tflops=%.4f\n",
           blockDim, repeat, warmup, avgMs, static_cast<double>(totalMs), gflop, tflops);

    CHECK_ACL(aclrtMemcpy(cHost, cBytes, cDev, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cBytes); // 输出结果到 output.bin 中去

    CHECK_ACL(aclrtDestroyEvent(startEv));
    CHECK_ACL(aclrtDestroyEvent(endEv));
    CHECK_ACL(aclrtFree(aDev));
    CHECK_ACL(aclrtFree(bDev));
    CHECK_ACL(aclrtFree(cDev));
    CHECK_ACL(aclrtFree(tilingDev));
    CHECK_ACL(aclrtFreeHost(aHost));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFreeHost(cHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
#else
int32_t main() { return 0; }
#endif

#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"

extern void matmul_v0_do(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);

int32_t main()
{
    const uint32_t blockDim = 1; //启动处理AI Core
    const uint32_t M = 1024, N = 1024, K = 1024; //数据处理的规模 M*N*K规模的数据乘

    size_t aBytes = (size_t)M * K * 2; //数据类型fp16
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
    ReadFile("./input/a.bin", aBytes, aHost, aBytes); //加载矩阵规模
    ReadFile("./input/b.bin", bBytes, bHost, bBytes);

    uint8_t *aDev = nullptr, *bDev = nullptr, *cDev = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&aDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&bDev, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&cDev, cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    //拷贝到设备端
    CHECK_ACL(aclrtMemcpy(aDev, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDev, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    //调用朴素版本的的函数
    constexpr int kRepeat = 10;
    for (int it = 0; it < kRepeat; ++it) {
        matmul_v0_do(blockDim, stream, aDev, bDev, cDev);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));   // 只同步一次

    CHECK_ACL(aclrtMemcpy(cHost, cBytes, cDev, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cBytes);//输出结果到output.bin中去

    CHECK_ACL(aclrtFree(aDev));
    CHECK_ACL(aclrtFree(bDev));
    CHECK_ACL(aclrtFree(cDev));
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
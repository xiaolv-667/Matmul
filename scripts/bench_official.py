#!/usr/bin/env python3
"""
bench_official.py —— 用 CANN 官方 Matmul 实现（aclnn，经 torch_npu）做对照测试

目的：验证「官方实测 157.8 TFLOPS」这个数是否成立，并给出与本项目 1024^3
fp16 同口径的官方算力。

口径说明（与本项目 bench.sh 对齐）：
  - 计算量 2*M*N*K，只算矩阵乘本身
  - 用 device 侧 Event 计时（不是 host wall clock），排除 launch 抖动
  - 预热若干次，取 N 次平均
"""
import time
import torch
import torch_npu  # noqa: F401  必须导入才能启用 npu 后端


def bench(size, iters, dtype=torch.float16):
    m = n = k = size
    a = torch.randn(m, k, dtype=dtype, device="npu")
    b = torch.randn(k, n, dtype=dtype, device="npu")

    for _ in range(5):                      # warmup
        torch.matmul(a, b)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = torch.matmul(a, b)
    end.record()
    torch.npu.synchronize()

    ms = start.elapsed_time(end) / iters    # 单次平均耗时（device 侧）
    tflops = 2.0 * m * n * k / (ms * 1e-3) / 1e12
    print(f"  {m}x{n}x{k}  dtype={str(dtype).split('.')[-1]:8s}"
          f"  {ms * 1000:8.2f} us  {tflops:8.2f} TFLOPS")
    del a, b, c
    return tflops


def main():
    print("=== CANN 官方 Matmul（aclnn via torch_npu）===")
    print(f"torch {torch.__version__} / torch_npu {torch.npu.get_device_name(0)}")
    print("")
    # 与本项目同口径：1024^3
    print("[同口径对比] 1024^3 fp16")
    bench(1024, 200)

    # 大尺寸：用来判断「整卡峰值」量级
    print("")
    print("[大尺寸：寻找官方实现的吞吐上限]")
    for s in (2048, 4096, 8192):
        bench(s, 50 if s <= 4096 else 20)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bench_official.py —— 用 CANN 官方 Matmul 实现（aclnn，经 torch_npu）做对照测试

目的：给出与本项目同口径（1024^3 fp16）的官方算力，作为对照基准；
      并通过大尺寸测量确定官方实现能达到的吞吐上限（用作峰值口径的实证）。

用途说明（避免误解）：本脚本只调用 torch.matmul 一行 API 取对照数据，
不含任何模型/训练逻辑。本项目的算子实现是纯 C++（Ascend C），不使用 PyTorch。

实测结论（详见 results.md）：
    1024^3  69.30 TFLOPS   <- 与 v5-multicore 的 87.67 对比为 1.26x
    2048^3 230.38 TFLOPS
    4096^3 273.16 TFLOPS
    8192^3 273.69 TFLOPS   <- 约为推导峰值 294.9 的 92.8%，印证峰值口径

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


def bench_softmax(m, n, iters, dtype=torch.float16):
    """官方 Softmax（aclnnSoftmax，经 torch.softmax）对照。

    口径与本项目 softmax 子项目一致：
      - 沿最后一维做 softmax
      - 必要流量 = 读 1 次 + 写 1 次 = 2*m*n*2 字节
      - device 侧 Event 计时
    """
    x = torch.randn(m, n, dtype=dtype, device="npu")

    for _ in range(5):
        torch.softmax(x, dim=-1)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = torch.softmax(x, dim=-1)
    end.record()
    torch.npu.synchronize()

    ms = start.elapsed_time(end) / iters
    min_bytes = 2.0 * m * n * 2.0            # 读 + 写
    bw = min_bytes / (ms * 1e-3) / 1e9       # GB/s
    print(f"  softmax {m}x{n}  {ms * 1000:8.2f} us  eff_bw={bw:8.2f} GB/s")
    del x, y
    return bw


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

    print("")
    print("=== CANN 官方 Softmax（aclnn via torch_npu）===")
    print("[同口径对比] 1024x1024 fp16，沿最后一维")
    bench_softmax(1024, 1024, 200)
    print("")
    print("[大尺寸]")
    for s in (2048, 4096):
        bench_softmax(s, s, 100)


if __name__ == "__main__":
    main()

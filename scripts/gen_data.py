import os
import numpy as np

# 问题规模由 MMA_SIZE 环境变量控制（默认 1024），与 main.cpp 的 MMA_SIZE 保持一致
S = int(os.environ.get("MMA_SIZE", "1024"))
M = N = K = S
DT = np.float16                      # 和官方基准同口径

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

rng = np.random.default_rng(42)
a = rng.standard_normal((M, K)).astype(DT)
b = rng.standard_normal((K, N)).astype(DT)

a.tofile("input/a.bin")
b.tofile("input/b.bin")

# fp32 累加再转 fp16，作为 golden
golden = (a.astype(np.float32) @ b.astype(np.float32)).astype(DT)
golden.tofile("output/golden.bin")

print(f"gen {M}x{N}x{K} {DT.__name__}, a={a.nbytes} bytes")

# v0 · 朴素 Vector 实现（基线）

> 提交：`v0-naive` ｜ 代码：`versions/matmul_v0_naive.cpp` ｜ 单元：Vector，1 核

## 1. 实现方式

矩阵乘拆成逐元素的向量运算。外层按行遍历 A，内层遍历 K：

```cpp
for (i = rowBegin; i < rowEnd; ++i) {              // 该核负责的行
    DataCopy(aHalf, aGm[i * K], K);                // A 的一行 → UB
    Duplicate(acc, 0.0f, N);                       // fp32 累加器清零
    for (k = 0; k < K; ++k) {
        float av = aHalf.GetValue(k);              // 取 A 的一个标量
        DataCopy(bHalf, bGm[k * N], N);            // B 的一行 2KB → UB
        Cast(bFloat, bHalf, CAST_NONE, N);         // fp16 → fp32
        Muls(tmp, bFloat, av, N);                  // × 标量
        Add(acc, acc, tmp, N);                     // 累加到 fp32
    }
    Cast(cHalf, acc, CAST_NONE, N);                // fp32 → fp16
    DataCopy(cGm[i * N], cHalf, N);
}
```

精度路线正确：**fp32 累加**，所以 `fro_err = 0`，与 golden 逐元素一致。

## 2. 基线数据

| 项 | 值 |
|---|---|
| 耗时（1 核） | **204.97 ms** |
| 算力 | 0.0105 TFLOPS |
| 精度 | `fro_err = 0`，`cos = 1.0` |

## 3. 瓶颈诊断（msprof）

40 核版本（`docs/prof/v0-naive/`）的最慢核拆解：

| 流水线 | 时间 | 占比 |
|---|---|---|
| Vector | 1835 µs | 33.2% |
| MTE2（GM→UB） | 3435 µs | 62.2% |
| MTE3（UB→GM） | 3 µs | 0.06% |
| **Vec + MTE2 之和** | **5270 µs** | **95.4%** |

**关键读数**：`ratio` 是「该流水线占用的墙钟时间比例」。两者之和 ≈ 100% 说明它们的**时间区间互不重叠 → 完全串行**。（若真正并行，两者可以同时接近 90%，和会到 180%。）

进一步把耗时拆到单次内层迭代：

```
迭代总数 = 1024 行 × 1024 次 K = 1,048,576
204.97 ms ÷ 1,048,576 = 195.5 ns/迭代
```

而这一轮的实际工作量（2 KB 搬运 + 3 条 1024 长向量指令）约 50 cycles ≈ 28 ns。
**即每轮 195 ns 里有约 170 ns 是纯停顿**（`SetFlag/WaitFlag` 配对 + MTE2 往返延迟 + `GetValue` 标量依赖）。

同时 MTE2 的活跃带宽只有 **14.8 GB/s**（2 KB / 129 ns），远低于核间公平份额——说明不是带宽不够，而是**每次搬得太小、固定延迟摊不掉**。

## 4. 这一版没解决的问题

| 问题 | 量化 |
|---|---|
| 只用 1 个核 | 910B3 有 40 个 AIV 可用 |
| 搬运与计算零重叠 | `vec + mte2 = 95.4%` |
| B 被重复读 1024 遍 | 实际数据 2 MB → 流量 2 GB |
| 搬运粒度 2 KB | 有效带宽只有 15 GB/s |

不过分核框架（`GetBlockIdx` / `rowBegin~rowEnd`）在 v0 就已经写好了，只是 `blockDim` 固定为 1——这为 v1 留下了空间。

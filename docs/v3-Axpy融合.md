# v3 · Axpy 指令融合

> 提交：`v3-axpy` ｜ 代码：`versions/matmul_v3_axpy.cpp` ｜ diff：`docs/patches/04_v2pingpong_to_v3axpy.diff`
> 单元：Vector，40 核

## 1. 判断依据

v2-pingpong 之后，msprof 显示 **Vector 占 90.3%**，MTE2 已降到 24.3% 且被完全掩盖 —— 瓶颈彻底转移到计算侧。

再看每个 k 步做了什么：

```cpp
Cast(bFloat, bCur[kk*N], CAST_NONE, N);   // ① half → float（精度需要）
Muls(tmp,   bFloat, av, N);               // ② × 标量   ┐ 这两条可以合成一条
Add (acc,   acc, tmp, N);                 // ③ + 累加器 ┘
```

真正有用的 GEMM 运算只需要 ②+③，而 **②③ 可以合并成一条 `Axpy`**。

## 2. 改动

```cpp
// v2：3 条指令
Cast(bFloat, bCur[kk * N], AscendC::RoundMode::CAST_NONE, N);
Muls(tmp,   bFloat, av, N);
Add (acc,   acc, tmp, N);

// v3：2 条指令
Cast(bFloat, bCur[kk * N], AscendC::RoundMode::CAST_NONE, N);
Axpy(acc,   bFloat, av, N);      // acc = bFloat*av + acc
```

API 语义已核对（`kernel_operator_vec_ternary_scalar_intf.h:63`）：`dst[i] = src[i]*scalar + dst[i]`，与原来的 `Muls + Add` 等价，**不改变累加顺序**。

顺带省掉 `tmp` 这块 4 KB UB（80 → 76 KB）。

## 3. 结果

| 指标 | v2-pingpong | v3-axpy |
|---|---|---|
| 耗时 | 1.9973 ms | **1.4796 ms** |
| 算力 | 1.0752 | **1.4514 TFLOPS** |
| Vector 时间 | 1806 µs | **1347 µs（−25.4%）** |
| Vector 占比 | 90.3% | 91.0% |
| `vec + mte2` | 114.6% | 121.0% |
| 精度 | `fro_err = 0` | **`fro_err = 0`**（依旧逐元素精确） |

## 4. 预测偏差复盘（值得记）

我原本预测 Vector 时间会降到 `1806 × 2/3 = 1204 µs`（−33%），实测 **1347 µs（−25.4%）**，差 12%。

原因是有两个模型可以算，**都没完全对上**：

| 模型 | 预测倍数 | 预测 Vector 时间 | 对实测 1347 µs |
|---|---|---|---|
| 指令数模型（3 条 → 2 条） | 0.667 | 1204 µs | 偏乐观 12% |
| UB 流量加权模型 | 0.692 | 1249 µs | 偏乐观 7.8% |
| **实测** | **0.746** | 1347 µs | — |

UB 流量模型推导（每 k 步的 UB 读写字节数）：

```
v2: Cast(读2K+写4K=6K) + Muls(4K+4K=8K) + Add(4K+4K读+4K写=12K) = 26 KB
v3: Cast(6K) + Axpy(4K+4K读+4K写=12K)                            = 18 KB
                                                        18/26 = 0.692
```

流量模型比指令数模型准，但仍偏乐观 7.8%。剩下的差额说明：**向量指令存在不随数据量变化的固定开销**（发射、掩码设置、pipe 排队）。

**教训**：做融合类优化时，不能按「少了 1/3 指令 = 快 1/3」推，要按 UB 流量加权，再打一个固定开销折扣。

## 5. 这一版摸到的墙

做完 Axpy 后，每个 k 步只剩 **2 条指令（`Cast` + `Axpy`）**：

- `Cast` 去不掉：fp16 必须转 fp32 才保精度（换 fp16 累加会让 `fro_err` 从 0 跳到 1e-2 量级，精度判据直接挂）
- `Axpy` 去不掉：累加是矩阵乘的定义

**2 条指令就是这种做法的物理下限。** 由此得到 Vector 路线的理论上限：

```
理想总时长 = Vector 时间 = 1347 µs  →  2.147 / 1.347 = 1.594 TFLOPS
当前效率   = 1.4514 / 1.594 = 91.0%   ← 与 Vector 占比一致（必然）
```

也就是说 Vector 路线上最多还能再拿 9%，把 1.48 ms 压到 1.33 ms。而对照 Cube 路径的目标（20–40 µs）差 **37–74 倍** —— 这个差距不是微调能补的，所以下一步是换计算单元。

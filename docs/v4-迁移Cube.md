# v4 · 迁移到 Cube 矩阵单元

> 提交：`v4-cube` ｜ 代码：`versions/matmul_v4_cube.cpp` ｜ diff：`docs/patches/05_v3_to_v4cube.diff`
> 单元：**Cube（AIC）**，单核

## 1. 为什么必须换单元

v3 已经把 Vector 路线做到了物理下限（每 k 步 2 条指令），理论上限 1.594 TFLOPS，当前已达 91.0%。而矩阵乘天生是 Cube 的工作：

| 单元 | 一次能做什么 |
|---|---|
| Vector | 一条指令处理 1024 个元素的逐元素运算 |
| **Cube** | 一条 `Mmad` 完成 **16×16×16 的矩阵乘累加** |

用 Vector 做矩阵乘等于把矩阵乘手工展开成上百万次逐元素乘加。这条路线的天花板和 Cube 相差一个数量级。

## 2. 先走的手写 Mmad 路线（已归档，精度未通）

按最底层的方式实现：`DataCopy + Nd2NzParams`（GM→L1 的 ND→NZ 转换）→ `LoadData`（L1→L0A/L0B，B 用 `ifTranspose`）→ `Mmad`（L0C 内累加）→ `Fixpipe`（L0C→GM，fp32 转 fp16）。

**结果：20 核跑到 61.5 µs，但 `fro_err = NaN`。**

诊断过程（这部分比结果更有价值）：

| 步骤 | 观察 | 结论 |
|---|---|---|
| msprof | `Cube 14.3%`、`Fixpipe 66.2%` 但只有 2.68 GB/s | Fixpipe 在忙，却搬得极慢 |
| 清零 C 显存后再跑 | 输出只有 **6.2% 非零**，其中 94% 是 NaN | 写入位置/内容错 |
| 对照官方 `GetCBCopyOutParams` | 地址公式、`nSize`、`dstStride` 都一致 | 问题在 `mSize` / `srcStride` 的单位语义 |

`FixpipeParamsV220` 的 `mSize` / `srcStride` 单位属于硬件层，本地无文档可验证，只能试错。**该版本的性能数据已作废**（见 `results.md` 中的 `[作废]` 标注与 `docs/prof/hand-written-cube-INVALID/`）。

## 3. 改走官方 Matmul API（最终采用）

```cpp
TPipe pipe;
AscendC::MatmulImpl<A_T, B_T, C_T, C_T> mm;   // 注意不是 AscendC::Matmul
mm.Init(&tilingData, &pipe);
mm.SetTensorA(aGm);
mm.SetTensorB(bGm);
mm.IterateAll(cGm);
```

### 路上踩的四个坑

| 坑 | 现象 | 解法 |
|---|---|---|
| 用了 `AscendC::Matmul` | 进程死锁（`timeout` 124），`AIC_ONLY` 和 mix 都挂 | 该别名在 910B 上映射到 KFC 模式的 `MatmulClient`，依赖 AIC/AIV 握手；改用 **`MatmulImpl`** 走 MDL 路径，无握手无 workspace |
| 直接传 tiling 指针 | 编译错误：`__gm__` 指针不能转普通指针 | 先把 tiling 从 GM 读进核内局部变量（等价官方 `GET_TILING_DATA`） |
| `PlatformAscendC::GetInstance` | 编译错误：无此成员 | 单例在 **`PlatformAscendCManager`** 上 |
| host 侧找不到 tiling 头 | 编译错误 | 需 `-I${CANN}/aarch64-linux/asc/include` |
| 手填 `TCubeTiling` | 精度 `fro_err≈0.97` | 改用官方 **`MatmulApiTiling`** 生成器，立刻正确 |

## 4. 结果

| 指标 | v3-axpy | v4-cube |
|---|---|---|
| 单元 / 核数 | Vector / 40 | **Cube / 1** |
| 耗时 | 1.4796 ms | **0.198421 ms** |
| 算力 | 1.4514 | **10.82 TFLOPS** |
| 精度 | `fro_err = 0` | **1.588e-05** |

**只用 1 个 Cube 核，就超过 40 核 Vector 的版本 7.46 倍。**

msprof（`docs/prof/v4-cube-1core/`）：

| 流水线 | 占比 |
|---|---|
| Cube | **69.1%** |
| MTE2 | **93.1%**（已饱和） |
| MTE1 | 47.3% |
| Fixpipe | 34.5% |

把算力拆成两个百分比，避免误读：

```
单核效率  = 10.82 ÷ (294.9/20 = 14.75) = 73.4%   ← 实现质量
整卡占比  = 10.82 ÷ 294.9               = 3.67%   ← 只用了 1/20 个核
```

3.67% 低不是因为实现差，是因为只开了一个核。

## 5. 试过但没成功的三件事

| 尝试 | 结果 |
|---|---|
| `SetFixSplit(128,128,128)` | 237.6 µs，比生成器 auto 的 198.4 µs 慢 16% |
| `SetFixSplit(128,128,64)` | 被生成器**钳制成 baseK=128**（输出仍是 128）→ baseK 不是自由变量 |
| `SetShape(256,256,1024)` 想切 16 核 | `usedCoreNum` 仍为 1，只算 1/16 输出，`cos = 0.2518 = 1/√16`；再手动覆盖 `usedCoreNum=16` 仍是 0.2518 → 16 个核算的是同一块 |

结论：**本 CANN 版本的 `MatmulApiTiling` 不支持 host 侧核数切分**（无 `SetDim` / `SetSingleShape` / `MultiCoreMatmulTiling`）。这直接引出 v5——把核间切分移到 kernel 里做。

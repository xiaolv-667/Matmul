# 昇腾 910B3 矩阵乘算子性能优化

从零手写 Ascend C 矩阵乘算子，用 7 个版本迭代把 1024³ fp16 矩阵乘从 **5.39 ms** 优化到 **24.5 µs**（**220×**），算力 0.40 → **87.7 TFLOPS**；同尺寸下超过 CANN 官方 `aclnnMatmul` **26%**。

整个演进过程、每一版的代码快照、以及支撑结论的 profiling 原始数据都在本仓库里。

---

## 一、结果

固定条件：`M = N = K = 1024`，fp16 输入 / **fp32 累加** / fp16 输出，时长只计 kernel（不含 H2D/D2H），`aclrtEvent` 夹 100 次连发取平均。

| 版本 | 计算单元 | 核数 | 耗时 | 算力 | 相对上一版 | 精度 `fro_err` |
|---|---|---|---|---|---|---|
| `v0-naive` | Vector | 1 AIV | 204.97 ms | 0.0105 TFLOPS | — | 0 |
| `v1-multicore` | Vector | 40 AIV | 5.3919 ms | 0.3983 TFLOPS | 38.0× | 0 |
| `v2-chunk` | Vector | 40 AIV | 2.4663 ms | 0.8707 TFLOPS | 2.19× | 0 |
| `v2-pingpong` | Vector | 40 AIV | 1.9973 ms | 1.0752 TFLOPS | 1.23× | 0 |
| `v3-axpy` | Vector | 40 AIV | 1.4796 ms | 1.4514 TFLOPS | 1.35× | 0 |
| `v4-cube` | **Cube** | 1 AIC | 0.1984 ms | 10.82 TFLOPS | 7.46× | 1.59e-05 |
| **`v5-multicore`** | **Cube** | **16 AIC** | **0.0245 ms** | **87.67 TFLOPS** | **8.14×** | **1.59e-05** |

两个最有说服力的对比：

- **核数相同时换计算单元**：单核 Vector `204.97 ms` → 单核 Cube `0.1984 ms`，**1033×**
- **整体**：40 核 Vector `5.3919 ms` → 16 核 Cube `0.0245 ms`，**220×**

> 第 7 版只用 16 个 Cube 核，就打败了第 5 版用满 40 个 Vector 核的实现 60 倍——**换计算单元的收益远大于在同一单元上做工程优化**。

---

## 二、瓶颈转移链（方法论）

每一版都先用 `msprof op` 定位瓶颈，再针对性优化，并用同一工具验证优化是否真的生效。这是本项目贯穿始终的方法。

| 版本 | 上一版的瓶颈 | 证据 | 手段 | 优化后瓶颈 |
|---|---|---|---|---|
| v0 → v1 | 只用 1 核；每轮迭代 195 ns，85% 在空转 | `vec+mte2 = 95.4%`（时间互斥=串行） | 开满 40 核 | **每轮迭代成本没变**（195→202 ns），只是横向复制 |
| v1 → v2 | 搬运粒度太小：MTE2 活跃带宽仅 14.8 GB/s | 2 KB/次，固定延迟摊不掉 | 一次搬 16 行（32 KB） | MTE2 带宽 → 104.5 GB/s |
| v2 阶段 | 搬运与计算零重叠 | `vec+mte2` 时间占比 114.6% | 双缓冲 ping-pong | 重叠发生 |
| v2 → v3 | Vector 指令数：每 k 步 3 条（Cast/Muls/Add） | Vector 占比 90.3% | `Axpy` 融合 Muls+Add | Vector 时间 1806 → 1347 µs |
| v3 → v4 | 2 条指令已是 Vector 路线物理下限 | Vector 仍占 91% | **迁移 Cube 单元** | Cube 69.1% / MTE2 93.1% |
| v4 → v5 | 单核 MTE2 已 93% 饱和 | `aic_time` 225 µs | kernel 内自行核间切分，16 核 | Cube 34.2% / Scalar 61.5% |

**过程中被数据推翻的两个判断**（记录在案，因为它们同样有价值）：

1. `Kb=32` 实验：把单次搬运从 32 KB 加到 64 KB，MTE2 时间从 486 → 346 µs，但**总时长不变**（1997 → 2008 µs）。说明瓶颈已转移后，继续优化非瓶颈项零收益。
2. 4096³ 流量减 4×（1 GB → 256 MB），算力只提升 **2.6%**。证明大尺寸**不是带宽受限**（实测仅 322 GB/s），此前"撞带宽墙"的判断是错的。

---

## 三、与 CANN 官方实现对比

官方数据由 `scripts/bench_official.py`（`torch_npu` → `aclnnMatmul`）实测，口径相同（device 侧 Event 计时、只算 kernel）。

| 尺寸 | 官方 TFLOPS | 本项目 TFLOPS | 对比 |
|---|---|---|---|
| **1024³** | 69.30 | **87.67** | **1.26×（本项目更快）** |
| 2048³ | 230.38 | 150.70 | 0.65× |
| 4096³ | 273.16 | 172.81 | 0.63× |
| 8192³ | **273.69** | —（kernel 只支持到 4096） | — |

峰值口径核对：`294.9 TFLOPS = 20 核 × 4096 MAC/cycle × 2 FLOP × 1.8 GHz`，官方实现在 8192³ 达到 **273.69（92.8%）**，印证了该推导。

---

## 三之二、Roofline 定位（算力轴 vs 带宽轴）

两个轴都有实测值，用来判断"该优化计算还是该优化搬运"：

| 项 | 数值 | 定义 / 来源 |
|---|---|---|
| 算术强度 | **512 FLOP/B** | `2·M·N·K ÷ (A+B 字节数)`，是问题固有属性，不随实现变 |
| 机器平衡点 | **184 FLOP/B** | 峰值 294.9 TFLOPS ÷ HBM 1.6 TB/s |
| **峰值算力利用率** | **29.7%** | `87.67 TFLOPS ÷ 294.9`，整卡口径 |
| **Cube 流水线占空比** | **35.12%** | msprof `aic_cube_fp16_ratio`，16 核平均（`docs/prof/v5-multicore-roofline/`） |
| **GM→L1 带宽利用率** | **18.38%** | msprof `GM_to_L1_bw_usage_rate`（`docs/prof/v5-multicore-L2Memory/`） |

**两个"利用率"不是一回事，混用会被追问**：

- `29.7%` = 整卡算力用掉多少（achieved TFLOPS ÷ chip peak）
- `35.12%` = Cube 这一条流水线在核内的占空比（时间维度）

因为 **512 FLOP/B >> 184 FLOP/B ⇒ compute-bound**，所以主线是让 Cube 别停下来等数据（v4 迁移 Cube、v5 多核）。
同期带宽利用率仅 18.38%，也反向印证了"4096³ 不是带宽受限"那个结论（见 v5 文档第 5 节）。

### 剩余空间评估（ratio 的分母决定结论）

"还有多少优化空间"要用 **ratio = 实测 ÷ 该强度下的可达上界** 衡量，但**分母选谁结论能差 3 倍**，引用时必须写明分母：

| 分母（可达峰值口径） | 数值 | 本项目 ratio | 折算还能快 |
|---|---|---|---|
| 理论峰值（20 核 × 4096 MAC × 2 × 1.8 GHz） | 294.9 TFLOPS | **29.7%** | 3.4× |
| 本实现实际可用核数（16/20，受 16 对齐限制） | 235.9 TFLOPS | **37.2%** | 2.7× |
| 官方实现在大尺寸的可达峰值（8192³） | 273.7 TFLOPS | **32.0%** | 3.1× |
| 官方实现在**同尺寸**（1024³） | 69.3 TFLOPS | **126.5%** | 已超越 |

**最后两行并排看是关键**：

```
官方在 1024³ 的 ratio = 69.3 / 273.7 = 25.3%
我们在 1024³ 的 ratio = 87.7 / 273.7 = 32.0%     ← 同等尺寸约束下高 1.27 倍
```

即：剩下的约 3 倍空间**主要来自"1024³ 这个尺寸太小、流水线填充/排空摊不开"**，而不是实现缺陷。
这与 `docs/v5-多核Cube.md` 里"每核只有 9.1 µs 的 Cube 工作量、实际墙钟 28.9 µs"的结论一致。

## 四、仓库导航

```
.
├── README.md                  ← 你在这里
├── results.md                 ← 全部实测台账（含作废数据标注）
├── matmul_v4.cpp              ← 当前版本源码（构建用）
├── main.cpp                   ← host：数据准备、tiling 生成、计时
├── CMakeLists.txt / build.sh / bench.sh
├── scripts/
│   ├── gen_data.py            ← 按 MMA_SIZE 生成输入与 golden
│   ├── verify_result.py       ← fro_err / cos 判定
│   ├── prof.sh                ← msprof 采集封装
│   ├── prof_summary.sh        ← profiling 结果汇总
│   └── bench_official.py      ← CANN 官方实现对照测试
├── versions/                  ← ★ 7 个版本的代码快照，可直接对比
│   ├── matmul_v0_naive.cpp
│   ├── matmul_v1_multicore.cpp
│   ├── matmul_v2_chunk.cpp
│   ├── matmul_v2_pingpong.cpp
│   ├── matmul_v3_axpy.cpp
│   ├── matmul_v4_cube.cpp
│   └── matmul_v5_multicore.cpp
├── docs/
│   ├── v0-朴素Vector.md        ← 各版本的瓶颈分析（v0~v5）
│   ├── v1-多核.md
│   ├── v2-分块与双缓冲.md
│   ├── v3-Axpy融合.md
│   ├── v4-迁移Cube.md
│   ├── v5-多核Cube.md
│   ├── patches/                ← ★ 各版本之间的 diff（每步只改十几行）
│   └── prof/                   ← ★ 精选的 msprof 原始 CSV
└── git tag: v0-naive … v5-multicore   ← 7 个版本，均可检出复现
```

**阅读建议**：先看 `versions/` 里的两个相邻文件差异（或 `docs/patches/` 的 diff），再看 `docs/` 里对应那一版的分析，最后到 `docs/prof/` 核对数据。

> **另一个算子已拆分为独立仓库**：[xiaolv-667/Softmax](https://github.com/xiaolv-667/Softmax)（Vector 路线 / memory-bound，与 Matmul 的 compute-bound 互补）。
> 0.8619 ms → 0.0147 ms（**58.6×**），同尺寸比 CANN 官方 `aclnnSoftmax` 快 **2.25×**，含 v0~v6 六版逐版 msprof 归因。
> 本仓库只保留 Matmul，不再包含 Softmax 的代码。

---

## 五、复现

环境：Ascend 910B3，CANN 9.0.0，`ai_core_cnt=20 / vector_core_cnt=40 / l1_size=512KB / l2_size=192MB / ub_size=192KB`。

```bash
# 1. 编译（自动 source CANN 环境，默认 Ascend910B3 + Release）
./build.sh -c

# 2. 跑当前版本（1024³，16 核）
./bench.sh 16

# 3. 换问题规模（1024 / 2048 / 4096）
MMA_SIZE=2048 ./bench.sh 16

# 4. profile（输出目录会打印在终端）
./scripts/prof.sh 16 PipeUtilization

# 5. 与官方实现对照
python3 scripts/bench_official.py
```

`bench.sh` 每次运行会往 `results.md` 追加一行（含版本号、耗时、算力、精度），日志落在 `logs/`。

返回 `test pass` 表示 `fro_err < 1e-3` 且 `cos > 0.9999`。

检出历史版本：

```bash
git checkout v3-axpy && ./build.sh -c && ./bench.sh 40
```

---

## 六、已知边界（不回避）

| 边界 | 说明 |
|---|---|
| **只用 16/20 个 Cube 核** | Cube 要求每维 16 元素对齐，而 `1024/5 = 204.8` 不是 16 的倍数，5 路切分无法均分。16 核零失配（实测 4.65%）优于 20 核的 4/3 失配。 |
| **大尺寸为官方的 63%** | 差距分解：只用 16/20 核（−20%）+ 单核效率 73%（官方 92.6%）。 |
| **kernel 只支持 M/N 为 4 的倍数** | tile = M/4，未做尾块处理。 |
| **手写 Mmad 版本精度未通** | 20 核可跑到 61.5 µs，但 `Fixpipe` 的 `mSize/srcStride` 未校准，输出仅写入 6.2%、`fro_err = NaN`。该版本性能数据**作废**，未计入结果表（见 `results.md` 中的 `[作废]` 标注）。 |

---

## 七、这个项目里值得一提的两个坑

**1. 工具链把算子误判成 mix 核，导致性能回退 1.6×**

纯 Vector 算子若不加 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，工具链会编成 mix 核（每块 1 AIC + 2 AIV），`blockDim=40` 实际启动 80 个 AIV 任务压到 40 个物理核上。不报错、精度正常，只是悄悄慢 1.6 倍（3996 µs vs 1997 µs）。诊断线索是 `msprof` 里的 `Op Type: mix` 与 `build/auto_gen/.../aiv_config.cmake` 中的 `MIX_SOURCES`。

**2. 陈旧输出会伪装成"测试通过"**

进程被 `timeout` 杀掉时不会写 `output.bin`，`verify_result.py` 会去校验上一次的旧文件并报 `test pass`。**必须先判退出码再判精度**，本仓库的验证流程已按此修正。

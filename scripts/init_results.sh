#!/bin/bash
###############################################################################
# init_results.sh —— 初始化性能实验台账 results.md
# 由 bench.sh 在发现台账不存在时自动调用
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}")/.. && pwd)
RESULTS="${CURRENT_DIR}/results.md"

cat > "${RESULTS}" <<'EOF'
# Matmul 性能实验台账

**固定条件**（所有数据必须在此口径下才有可比性）

| 项 | 值 |
|---|---|
| 问题规模 | M = N = K = 1024 |
| 数据类型 | fp16 输入/输出，fp32 累加 |
| 理论计算量 | 2·M·N·K ≈ **2.147 GFLOP** |
| 精度判据 | `fro_err < 1e-3` 且 `cos > 0.9999` |
| TFLOPS 口径 | 2·M·N·K / 单次耗时（不含 H2D/D2H） |
| 计时方式 | `aclrtEvent` 夹住 N 次 launch，去 warmup，取平均 |
| 设备 | Ascend 910B3 |

**阅读方式**：`TFLOPS` 是核心指标，`avg_ms` 是原始耗时，`结果` 是精度校验是否通过。
每行数据由 `bench.sh` 自动追加，日志见 `logs/`。

| 时间 | 版本 | SOC/构建类型 | blockDim | repeat | avg_ms | TFLOPS | fro_err | cos | 结果 |
|------|------|--------------|----------|--------|--------|--------|---------|-----|------|
EOF

echo "[INFO] 已初始化 ${RESULTS}"

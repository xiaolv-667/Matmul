#!/bin/bash
###############################################################################
# bench.sh —— 只负责跑实验并记账，不编译
#
# 依赖: 先执行 ./build.sh 生成 out/bin/ascendc_kernels_bbit
#
# 用法:
#   ./bench.sh              # blockDim=1，默认 100 次重复取平均
#   ./bench.sh 20           # 指定 blockDim=20（多核扫描用）
#   REPEAT=200 ./bench.sh 40
#   RUN_WITH_TOOLCHAIN=1 ./bench.sh 1   # 走 msprof op 做 profiling
#
# 输出:
#   控制台打印 [BENCH] / [VERIFY] 两行机读结果
#   同时把一行数据追加进 results.md 台账
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
cd "${CURRENT_DIR}"

BLOCK_DIM="${1:-${BLOCK_DIM:-1}}"
REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-5}"
RUN_MODE="npu"
BIN="${CURRENT_DIR}/out/bin/ascendc_kernels_bbit"
LIBDIR="${CURRENT_DIR}/out/lib"

# ---- 定位 CANN ----
ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH:-${ASCEND_HOME_PATH}}"
if [ -z "${ASCEND_INSTALL_PATH}" ]; then
    if [ -d "${HOME}/Ascend/ascend-toolkit/latest" ]; then
        ASCEND_INSTALL_PATH="${HOME}/Ascend/ascend-toolkit/latest"
    else
        ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi
export ASCEND_TOOLKIT_HOME="${ASCEND_INSTALL_PATH}"
export ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"

# ---- 前置检查 ----
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] 找不到可执行文件: ${BIN}"
    echo "        请先执行 ./build.sh"
    exit 1
fi
if [ ! -f "${LIBDIR}/libascendc_kernels_${RUN_MODE}.so" ]; then
    echo "[ERROR] 找不到 kernel 库: ${LIBDIR}/libascendc_kernels_${RUN_MODE}.so"
    echo "        请先执行 ./build.sh"
    exit 1
fi

# ---- 数据只生成一次（gen_data.py 用固定 seed=42，可复现）----
if [ ! -f "input/a.bin" ] || [ ! -f "input/b.bin" ] || [ ! -f "output/golden.bin" ]; then
    echo "[INFO] 生成输入数据与 golden"
    mkdir -p input output
    python3 scripts/gen_data.py
fi

# ---- 记录本次实际使用的构建配置（而不是命令行以为的配置）----
build_soc=$(sed -n 's/^SOC_VERSION:STRING=//p' build/CMakeCache.txt 2>/dev/null | head -1)
build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' build/CMakeCache.txt 2>/dev/null | head -1)
build_soc="${build_soc:-unknown}"
build_type="${build_type:-unknown}"

if [ "${build_soc}" != "Ascend910B3" ]; then
    echo "[WARN] 当前构建 SOC_VERSION=${build_soc}，与本机 910B3 不一致，数据不可信！"
fi
if [ "${build_type}" != "Release" ]; then
    echo "[WARN] 当前构建 CMAKE_BUILD_TYPE=${build_type}，性能对比实验建议用 Release！"
fi

export LD_LIBRARY_PATH="${LIBDIR}:${LIBDIR}64:${ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}"
export BENCH_REPEAT="${REPEAT}"
export BENCH_WARMUP="${WARMUP}"

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${CURRENT_DIR}/logs/bench_${STAMP}_blk${BLOCK_DIM}.log"

echo "=============================================="
echo " blockDim : ${BLOCK_DIM}"
echo " repeat   : ${REPEAT}  (warmup ${WARMUP})"
echo " soc/type : ${build_soc} / ${build_type}"
echo " log      : ${LOG}"
echo "=============================================="

# ---- 执行 ----
if [ "${RUN_WITH_TOOLCHAIN:-0}" = "1" ]; then
    msprof op --application="${BIN} ${BLOCK_DIM}" 2>&1 | tee "${LOG}"
else
    "${BIN}" "${BLOCK_DIM}" 2>&1 | tee "${LOG}"
fi

# ---- 校验精度 ----
set +e
python3 scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | tee -a "${LOG}"
set -e

# ---- 解析机读结果 ----
avg_ms=$(sed -n 's/.*avg_ms=\([0-9.]*\).*/\1/p' "${LOG}" | head -1)
tflops=$(sed -n 's/.*tflops=\([0-9.]*\).*/\1/p' "${LOG}" | head -1)
fro=$(sed -n 's/.*fro_err=\([^ ]*\).*/\1/p' "${LOG}" | head -1)
cos=$(sed -n 's/.*cos=\([^ ]*\).*/\1/p' "${LOG}" | head -1)
if grep -q "test pass" "${LOG}"; then
    verdict="PASS"
else
    verdict="**FAIL**"
fi

avg_ms="${avg_ms:-NA}"
tflops="${tflops:-NA}"
fro="${fro:-NA}"
cos="${cos:-NA}"

# ---- 版本标识 ----
if git rev-parse --git-dir >/dev/null 2>&1; then
    version=$(git describe --tags --always --dirty 2>/dev/null || echo "unknown")
else
    version="no-git"
fi

# ---- 追加进台账 ----
RESULTS="${CURRENT_DIR}/results.md"
if [ ! -f "${RESULTS}" ]; then
    "${CURRENT_DIR}/scripts/init_results.sh"
fi

printf '| %s | %s | %s/%s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$(date '+%Y-%m-%d %H:%M')" "${version}" "${build_soc}" "${build_type}" \
    "${BLOCK_DIM}" "${REPEAT}" "${avg_ms}" "${tflops}" "${fro}" "${cos}" "${verdict}" \
    >> "${RESULTS}"

echo ""
echo "[INFO] 已追加一行到 results.md:"
tail -1 "${RESULTS}"

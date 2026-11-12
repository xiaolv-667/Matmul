#!/bin/bash
###############################################################################
# prof.sh —— 用 msprof op 采集单算子性能数据，输出到 prof/ 目录
#
# 用法:
#   ./prof.sh                        # blockDim=40 + PipeUtilization（最常用）
#   ./prof.sh 40 Memory              # 看内存搬运量 / GM-L1-L0-UB 各级流量
#   ./prof.sh 40 Default             # 基础指标（耗时、算力、带宽总览）
#   ./prof.sh 40 L2Cache             # 看 L2 命中（判断重复读是否被 L2 吸收）
#   ./prof.sh 40 Roofline            # 屋顶线图，看瓶颈落在哪
#
# 可用的 --aic-metrics（逗号分隔可同时采多个）:
#   ArithmeticUtilization  MemoryUB  Memory  MemoryL0  L2Cache  PipeUtilization
#   ResourceConflictRatio  BasicInfo  Roofline  Occupancy  TimelineDetail
#   KernelScale  Source  MemoryDetail  Default
#
# 注意:
#   1. msprof 的 --application 只接受"可执行文件路径"，不能带参数。
#      本脚本自动生成一个把 blockDim 固化进去的包装器来绕过这个限制。
#   2. 必须 skip 掉程序的 warmup launch，否则采到的是预热那一发。
#   3. profiling 有额外开销，耗时会比 bench.sh 长，REPEAT 默认降到 5。
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}")/.. && pwd)
cd "${CURRENT_DIR}"

BLOCK_DIM="${1:-${BLOCK_DIM:-40}}"
METRICS="${2:-${AIC_METRICS:-PipeUtilization}}"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-2}"

BIN="${CURRENT_DIR}/out/bin/ascendc_kernels_bbit"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] 找不到可执行文件: ${BIN}，请先执行 ./build.sh"
    exit 1
fi

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
set +e
source "${ASCEND_INSTALL_PATH}/bin/setenv.bash" >/dev/null 2>&1
set -e

# 输出目录按 时间_blockDim_指标 命名，避免互相覆盖，便于横向对比
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${CURRENT_DIR}/prof/${STAMP}_blk${BLOCK_DIM}_${METRICS//,/_}"
mkdir -p "${OUTDIR}"

# ---- 包装器：msprof 不能给 --application 传参数，所以把 blockDim 固化 ----
WRAPPER="${OUTDIR}/run_blk${BLOCK_DIM}.sh"
cat > "${WRAPPER}" <<EOF
#!/bin/bash
# 自动生成：msprof 启动入口，固定 blockDim=${BLOCK_DIM}
cd "${CURRENT_DIR}"
export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${ASCEND_INSTALL_PATH}/lib64:\${LD_LIBRARY_PATH}"
exec "${BIN}" ${BLOCK_DIM}
EOF
chmod +x "${WRAPPER}"

export BENCH_REPEAT="${REPEAT}"
export BENCH_WARMUP="${WARMUP}"

echo "=============================================="
echo " blockDim    : ${BLOCK_DIM}"
echo " aic-metrics : ${METRICS}"
echo " repeat      : ${REPEAT} (warmup ${WARMUP}，采集时 skip 前 ${WARMUP} 发)"
echo " 输出目录    : ${OUTDIR}"
echo "=============================================="

# --launch-count=1        : 只采一发 kernel（够用，且快）
# --launch-skip-...       : 跳过程序的 warmup 发射，采到稳定的那一发
# --replay-mode=kernel    : 以 kernel 为单位重放采集，不受 host 侧抖动影响
msprof op \
    --application="${WRAPPER}" \
    --output="${OUTDIR}" \
    --aic-metrics="${METRICS}" \
    --launch-count=1 \
    --launch-skip-before-match="${WARMUP}" \
    --replay-mode=kernel

echo ""
echo "[INFO] 采集完成，产物结构（OPPROF_* 才是数据目录）："
find "${OUTDIR}" -maxdepth 2 -not -path '*/dump/*' | sed "s|${CURRENT_DIR}/||" | sort

echo ""
echo "[INFO] 该看哪几个文件："
echo "  OpBasicInfo.csv      算子名 / 类型 / Task Duration / 核数 —— 先看这个确认采对了"
echo "  ${METRICS%%,*}.csv        流水线占用明细，每个核一行"
echo "  visualize_data.bin   时间线，可用 MindStudio Insight 打开"
echo "  dump/                原始二进制与反汇编，一般不用碰"
echo ""
echo "[INFO] 一条命令汇总结果："
echo "  ./scripts/prof_summary.sh ${OUTDIR}"
echo ""
echo "[INFO] 注意：profiling 会拖慢程序，程序自己打印的 avg_ms 不可用于性能对比；"
echo "       权威时长看 OpBasicInfo.csv 里的 Task Duration(us)。"

#!/bin/bash
###############################################################################
# run.sh —— 兼容入口，等价于 build.sh + bench.sh
#
# 新流程建议分开用：
#   ./build.sh                 # 改了 kernel 才需要编
#   ./bench.sh 1               # 反复跑实验不重编
#
# 本脚本保留是为了兼容原有习惯：
#   ./run.sh -r npu -v Ascend910B3
###############################################################################
set -e

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)

"${CURRENT_DIR}/build.sh" "$@"
"${CURRENT_DIR}/bench.sh" "${BLOCK_DIM:-1}"

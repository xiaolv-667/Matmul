#!/bin/bash
###############################################################################
# prof_summary.sh —— 汇总一次 msprof 采集的结果，打印关键指标
#
# 用法:
#   ./scripts/prof_summary.sh prof/20261112_153239_blk40_PipeUtilization
#   ./scripts/prof_summary.sh prof/*blk40_PipeUtilization
###############################################################################
set -e

DIR="${1:?用法: ./scripts/prof_summary.sh <prof 数据目录>}"
[ -d "${DIR}" ] || { echo "[ERROR] 目录不存在: ${DIR}"; exit 1; }

BASIC=$(find "${DIR}" -name 'OpBasicInfo.csv' | head -1)
PIPE=$(find "${DIR}" -name 'PipeUtilization.csv' | head -1)

if [ -n "${BASIC}" ]; then
    echo "=============== 算子基本信息 ==============="
    # 第二行是数据行：Op Name, Op Type, Task Duration(us), Block Dim, ...
    awk -F, 'NR==2{printf "%-16s %-10s 时长 %s us   核数 %s\n", $1, $2, $3, $4}' "${BASIC}"
fi

if [ -z "${PIPE}" ]; then
    echo ""
    echo "[INFO] 本次采集没有 PipeUtilization.csv（指标不是 PipeUtilization，属正常）"
    echo "       数据目录内容："
    if [ -n "${BASIC}" ]; then
        ls "$(dirname "${BASIC}")"
    else
        ls "${DIR}"
    fi
    exit 0
fi

echo ""
echo "=============== 流水线占用（40 核汇总）==============="
awk -F, '
NR==1{ for(i=1;i<=NF;i++) h[$i]=i; next }
{
  t  = $h["aiv_time(us)"]+0
  vt = $h["aiv_vec_time(us)"]+0;   vr = $h["aiv_vec_ratio"]+0
  mt = $h["aiv_mte2_time(us)"]+0;  mr = $h["aiv_mte2_ratio"]+0
  bw = $h["aiv_mte2_active_bw(GB/s)"]+0
  m3 = $h["aiv_mte3_time(us)"]+0;  m3r= $h["aiv_mte3_ratio"]+0
  sc = $h["aiv_scalar_ratio"]+0
  n++; st+=t; svt+=vt; svr+=vr; smt+=mt; smr+=mr; sbw+=bw; sm3+=m3; sm3r+=m3r; ssc+=sc
  if(t>maxt){maxt=t; maxb=$1; maxvt=vt; maxvr=vr; maxmt=mt; maxmr=mr; maxbw=bw}
  if(t<mint||n==1){mint=t; minb=$1}
}
END{
  printf "核数            %d\n", n
  printf "aiv_time        min %.0f us (blk %s) / mean %.0f us / max %.0f us (blk %s)\n", mint, minb, st/n, maxt, maxb
  printf "实测负载不均    %.2f%%   <- 最慢核比最快核慢多少\n", (maxt-mint)/mint*100
  printf "\n最慢核 (blk %s) 拆解:\n", maxb
  printf "  Vector   %6.0f us  占比 %5.1f%%\n", maxvt, maxvr*100
  printf "  MTE2     %6.0f us  占比 %5.1f%%   活跃带宽 %.2f GB/s\n", maxmt, maxmr*100, maxbw
  printf "  --------- 两者合计 %.0f us = 总时长的 %.1f%%\n", maxvt+maxmt, (maxvt+maxmt)/maxt*100
  printf "  => 该值接近 100%% 说明 搬运与计算完全串行，没有重叠\n"
  printf "\n40 核均值:\n"
  printf "  Vector 占比 %.1f%%   MTE2 占比 %.1f%%   平均 MTE2 带宽 %.2f GB/s\n", svr/n*100, smr/n*100, sbw/n
  printf "  MTE3 %.0f us (占比 %.3f%%)  <- 输出搬运是否成为瓶颈\n", sm3/n, sm3r/n*100
  printf "  Scalar 占比 %.1f%%\n", ssc/n*100
}' "${PIPE}"

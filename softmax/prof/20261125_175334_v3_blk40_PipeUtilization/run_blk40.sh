#!/bin/bash
# 自动生成：msprof 启动入口，固定 blockDim=40
cd "/mnt/workspace/matmul/softmax"
export LD_LIBRARY_PATH="/mnt/workspace/matmul/softmax/out/lib:/mnt/workspace/matmul/softmax/out/lib64:/home/developer/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH}"
exec "/mnt/workspace/matmul/softmax/out/bin/softmax_bbit" 40

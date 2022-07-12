#!/bin/bash
#
# Script to launch a distributed training using PyTorch
# on each node for a multi-node multi-gpu training.
#
# PRIMARY (the primary node hostname) and PRIMARY_PORT (the TCP port
# used to establish communication with the primary node) are provided as
# environment variables by `init_node_info` called in `launch.sh`.
# 
# (c) 2021, Brian J. Stucky, UF Research Computing
# 2021/09, modified by Huiwen Ju, hju@nvidia.com

PT_LAUNCH_UTILS_PATH=$1
TRAINING_CMD=$2
PYTHON_PATH=$3

if [ -z "$PYTHON_PATH" ]
then
    PYTHON_PATH="python"
fi

LAUNCH_CMD="$PYTHON_PATH \
        -m torch.distributed.launch \
        --nproc_per_node=$SLURM_GPUS_PER_TASK \
               --nnodes=$SLURM_JOB_NUM_NODES \
               --node_rank=$SLURM_NODEID \
               --master_addr=$PRIMARY \
               --master_port=$PRIMARY_PORT \
             $TRAINING_CMD"

source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"
run_with_retry "$LAUNCH_CMD"
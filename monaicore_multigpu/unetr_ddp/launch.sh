#!/bin/bash

# Script to launch a multi-node pytorch.distributed training run on UF HiperGator's AI partition,
# a SLURM cluster using Singularity as container runtime.
# 
# This script uses `pt_multinode_helper_funcs.sh` and `run_on_node.sh`.
#
# If launch with torch.distributed.launch, 
#       set #SBATCH --ntasks=--nodes
#       set #SBATCH --ntasks-per-node=1  
#       set #SBATCH --gpus=total number of processes to run on all nodes
#       set #SBATCH --gpus-per-task=--gpus / --ntasks  
#       modify `LAUNCH_CMD` in `run_on_node.sh` to launch with torch.distributed.launch
      
# (c) 2021, Brian J. Stucky, UF Research Computing
# 2021/09, modified by Huiwen Ju, hju@nvidia.com

# Resource allocation.
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --partition=hpg-ai
#SBATCH --exclude=c0906a-s29,c1101a-s29,c1101a-s23,c1004a-s23,c1103a-s17
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=%x.%j.out

module load singularity

export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
# Training command specification: training_script -args.
TRAINING_SCRIPT="$(realpath "$HOME/monai_uf_tutorials/monaicore_multigpu/unetr_ddp/unetr_btcv_ddp.py")"
TRAINING_CMD="$TRAINING_SCRIPT"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Singularity container based on NGC PyTorch container,
# see `build_container.sh` to build a MONAI Singularity container.
PYTHON_PATH="singularity exec --nv --bind /blue/vendor-nvidia/hju/data/unetr_data:/mnt \
         /blue/vendor-nvidia/hju/monaicore0.8.1 python3" 
          
# Location of the PyTorch launch utilities, i.e. `pt_multinode_helper_funcs.sh` & `run_on_node.sh`.
# PT_LAUNCH_UTILS_PATH=$HOME/pt_dist_launch/UF_tutorial_multinode_MONAI
PT_LAUNCH_UTILS_PATH=$HOME/monai_uf_tutorials/monaicore_multigpu/util_multigpu
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")
echo "Running \"$TRAINING_CMD\" on each node..."

srun --unbuffered "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"    

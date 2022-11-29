#!/bin/bash
#
# Script to launch a multi-gpu distributed training using MONAI Core
# on UF HiperGator's AI partition, a SLURM cluster using Singularity 
# as container runtime.
# 
# This script uses `pt_multinode_helper_funcs.sh`, and 
# either `run_on_node.sh`(for single-node multi-gpu training) 
# or `run_on_multinode.sh` (for multi-node multi-gpu training). All
# the three `.sh` files are in \monaicore_multigpu\util_multigpu.
#
# We use torch.distributed.launch to launch the training, so please 
# set as follows: 
#   set #SBATCH --ntasks=--nodes
#   set #SBATCH --ntasks-per-node=1  
#   set #SBATCH --gpus=total number of processes to run on all nodes
#   set #SBATCH --gpus-per-task=--gpus/--ntasks  
#
#   for multi-node training, replace `run_on_node.sh` in 
#   `PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")`
#   with `run_on_multinode.sh`.
#   
#   Modify paths to your own paths.
#      
# (c) 2021, Brian J. Stucky, UF Research Computing
# 2022, modified by Huiwen Ju, hju@nvidia.com

# Resource allocation.
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --partition=hpg-ai
#SBATCH --exclude=c0906a-s29,c1101a-s29,c1101a-s23,c1004a-s23,c1103a-s17
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=%x.%j.out

export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
# Training command specification: training_script -args.
# TRAINING_SCRIPT="$(realpath "$HOME/monai_uf_tutorials/monaicore_multigpu/unet_ddp/unet_training_ddp.py")"
TRAINING_SCRIPT="$(realpath "$HOME/run_monaicore/dynunet/dynunet_pipeline/train.py")"
TRAINING_CMD="$TRAINING_SCRIPT \
-root_dir /mnt \
-datalist_path /mnt/dynunet/config/ \
-fold 0 \
-train_num_workers 4 \
-interval 1 \
-num_samples 1 \
-learning_rate 1e-1 \
-max_epochs 5 \
-task_id 04 \
-pos_sample_num 2 \
-expr_name baseline \
-tta_val True \
-multi_gpu True"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Core Singularity container,
# see `build_container.sh` to build a MONAI Core Singularity container.
PYTHON_PATH="singularity exec --nv \
--bind /blue/vendor-nvidia/hju/data/MSD:/mnt \
/blue/vendor-nvidia/hju/monaicore1.0.1 python3" 
          
# Location of the PyTorch launch utilities, 
# i.e. `pt_multinode_helper_funcs.sh`, `run_on_node.sh` and `run_on_multinode`.
# PT_LAUNCH_UTILS_PATH=$HOME/monai_uf_tutorials/monaicore_multigpu/util_multigpu
PT_LAUNCH_UTILS_PATH=$HOME/run_monaicore/multigpu/util_multigpu
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
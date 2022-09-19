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
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --partition=hpg-ai
#SBATCH --exclude=c0906a-s29,c1101a-s29,c1101a-s23,c1004a-s23,c1103a-s17
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out

export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
# Training command specification: training_script -args.
# TRAINING_SCRIPT="$(realpath "$HOME/monai_uf_tutorials/monaicore_multigpu/unet_ddp/unet_training_ddp.py")"
TRAINING_SCRIPT="$(realpath "$HOME/monai_uf_tutorials/pretrain_modify/main.py")"

# 1. train from scratch (no --use_checkpoint, use --noamp)
# 8 GPUs, batch_size = 1, 722/8 = 90 iters/epoch. val: 49 images
# TRAINING_CMD="$TRAINING_SCRIPT \
# --distributed \
# --logdir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 \
# --lrdecay --lr=6e-6 --decay=0.1 \
# --batch_size=1 \
# --epochs=3 --num_steps=270 --eval_num=90 \
# --noamp"

# 2. train from scratch (with amp, without checkpointing gradients)
# TRAINING_CMD="$TRAINING_SCRIPT \
# --distributed \
# --logdir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 \
# --lrdecay --lr=6e-6 --decay=0.1 \
# --batch_size=1 \
# --epochs=3 --num_steps=270 --eval_num=90"

# 3. NOT WORKING NOW! train from scratch (cachedataset, --mem Specify the real memory required per node.)
# TRAINING_CMD="$TRAINING_SCRIPT \
# --distributed \
# --logdir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 \
# --lrdecay --lr=6e-6 --decay=0.1 \
# --batch_size=1 \
# --epochs=3 --num_steps=270 --eval_num=90 \
# --noamp \
# --cache_dataset"

# 4. NOT WORKING NOW! train from scratch (smartcachedataset, --mem Specify the real memory required per node.)
# TRAINING_CMD="$TRAINING_SCRIPT \
# --distributed \
# --logdir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 \
# --lrdecay --lr=6e-6 --decay=0.1 \
# --batch_size=1 \
# --epochs=3 --num_steps=270 --eval_num=90 \
# --noamp \
# --smartcache_dataset"

# 5. resume training from downloaded pretrained model
# TRAINING_CMD="$TRAINING_SCRIPT \
# --distributed \
# --logdir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 \
# --lrdecay --lr=6e-6 --decay=0.1 \
# --batch_size=1 \
# --epochs=3 --num_steps=270 --eval_num=90 \
# --noamp \
# --resume=/mnt/pretrained_models/model_swinvit.pt"

# 6. resume training from a checkpoint
# TRAINING_CMD="$TRAINING_SCRIPT \
# --distributed \
# --logdir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 \
# --lrdecay --lr=6e-6 --decay=0.1 \
# --batch_size=1 \
# --epochs=3 --num_steps=270 --eval_num=90 \
# --noamp \
# --resume=/mnt/runs/model_bestValRMSE.pt"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Core Singularity container,
# see `build_container.sh` to build a MONAI Core Singularity container.
# PYTHON_PATH="singularity exec --nv \
#          /blue/vendor-nvidia/hju/monaicore0.8.1 python3" 
PYTHON_PATH="singularity exec --nv --bind /blue/vendor-nvidia/hju/data/swinunetr_pretrain_CT:/mnt \
         /blue/vendor-nvidia/hju/monaicore0.9.1 python3"

# Location of the PyTorch launch utilities, 
# i.e. `pt_multinode_helper_funcs.sh`, `run_on_node.sh` and `run_on_multinode`.
PT_LAUNCH_UTILS_PATH=$HOME/monai_uf_tutorials/monaicore_multigpu/util_multigpu
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")
# PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_multinode.sh")
echo "Running \"$TRAINING_CMD\" on each node..."

srun --unbuffered "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"    
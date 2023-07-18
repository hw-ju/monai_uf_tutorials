#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1               
#SBATCH --ntasks=1              
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8     
#SBATCH --cpus-per-task=4       
#SBATCH --mem=64gb             
#SBATCH --partition=hpg-ai
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

module load singularity

export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
# Training command specification: training_script -args.
TRAINING_SCRIPT="$(realpath "$HOME/monai_uf_tutorials/monaicore_hovernet/training.py")"
TRAINING_CMD="$TRAINING_SCRIPT \
--stage=0 \
--ep=3 \
--bs=16 \
--log-dir=./logs \
--root=/mnt/Prepared"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Core Singularity container,
# see `build_container.sh` to build a MONAI Core Singularity container.
PYTHON_PATH="singularity exec --nv \
--bind /blue/vendor-nvidia/hju/data/hovernet_data/CoNSeP:/mnt \
/blue/vendor-nvidia/hju/monaicore1.1.0"
          
# Location of the PyTorch launch utilities, 
# i.e. `pt_multinode_helper_funcs.sh`, `run_on_node.sh` and `run_on_multinode`.
PT_LAUNCH_UTILS_PATH=$HOME/monai_uf_tutorials/monaicore_hovernet/util_multigpu
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_multinode.sh")
echo "Running \"$TRAINING_CMD\" on each node..."

srun --unbuffered "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"

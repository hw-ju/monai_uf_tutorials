#!/bin/bash

#
# Script to launch a multi-gpu raytune HPO
# on UF HiperGator's AI partition
# 
#   
#   Modify from the NERSC repo https://github.com/NERSC/slurm-ray-cluster
#      
# (c) 2023 Yunchao Yang, UF Research Computing


#SBATCH --job-name=test_RayTune_MONAI_single_node
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --partition=hpg-ai

### This script works for any number of nodes, Ray will find and manage all resources
##SBATCH --nodes=2
##SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1000gb

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-task=4
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
##SBATCH --exclusive


# Load modules or your own conda environment here
#module load pytorch
module load conda
# conda activate ray-torch
#conda activate ray-complete-torch
conda activate ray-monai

##############################################################################################
# HJ: "-s" gives c1000a-s11.ufhpc -> c1000a-s11
head_node=$(hostname -s)
# head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
echo head_node=${head_node}
echo head_node_ip=${head_node_ip}

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi

port=6379

echo "*****************"
echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"

# srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --partition=hpg-ai --gpus-per-node=8 --nodelist=${head_node:0:10} start-head.sh $head_node_ip &
# HJ: -w $head_node
srun \
--nodes=1 \
--ntasks=1 \
--ntasks-per-node=1 \
--gpus-per-task=$SLURM_GPUS_PER_TASK \
--cpus-per-task=$SLURM_CPUS_PER_TASK \
-w $head_node \
start-head.sh &
#start-head.sh $head_node_ip &
sleep 10

##############################################################################################
# worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
# echo worker_num=${worker_num}

#srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $head_node_ip:$port &
# srun --ntasks=$worker_num --nodes=$worker_num --ntasks-per-node=1 --partition=hpg-ai --gpus-per-node=8 --exclude ${head_node:0:10} start-worker.sh $head_node_ip:$port &

# HJ:
# srun \
# --nodes=$worker_num \
# --ntasks=$worker_num \
# --ntasks-per-node=1 \
# --gpus-per-task=$SLURM_GPUS_PER_TASK \
# --cpus-per-task=$SLURM_CPUS_PER_TASK \
# --exclude $head_node \
# start-worker.sh $head_node_ip:$port &
# sleep 5
##############################################################################################

#### call your code below
python examples/mnist_pytorch_trainable.py --cuda

exit

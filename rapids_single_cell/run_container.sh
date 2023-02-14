#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

singularity run --nv \
--bind /blue/vendor-nvidia/hju/single_cell_data:/data \
/blue/vendor-nvidia/hju/single-cell-examples_rapids_cuda11.0

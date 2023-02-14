#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

singularity build --sandbox /blue/vendor-nvidia/hju/single-cell-examples_rapids_cuda11.0 docker://claraparabricks/single-cell-examples_rapids_cuda11.0

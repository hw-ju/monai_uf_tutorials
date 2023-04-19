#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# Go to the directory hosting tutorial jupyter notebooks
cd /home/hju/monai_uf_tutorials/AI4HEALTH_workshop/end2end

# Run the RAPIDS container (on the start of the container, a jupyter lab will run by default)
singularity run --nv /apps/nvidia/containers/rapids/workshop_rapids23.02
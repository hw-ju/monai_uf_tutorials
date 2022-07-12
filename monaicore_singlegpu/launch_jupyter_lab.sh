#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# Go to home directory
cd $HOME

# Modify the path to your singularity container 
singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.8.1 jupyter lab
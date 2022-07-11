#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1

date;hostname;pwd

module load singularity

cd $HOME

# Modify the path to your singularity container 
singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.8.1 jupyter lab
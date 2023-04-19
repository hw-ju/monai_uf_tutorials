#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# List all monailabel commands
echo "****************** monailabel --help"
singularity exec /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel --help 
# List sample apps
echo "****************** monailabel apps"
singularity exec /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel apps 
# List sample datasets
echo "****************** monailabel datasets"
singularity exec /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel datasets

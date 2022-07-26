#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# Show all monailabel commands
singularity exec /apps/nvidia/containers/monai/monailabel/ monailabel --help 
# List sample apps
singularity exec /apps/nvidia/containers/monai/monailabel/ monailabel apps 
# List sample datasets
singularity exec /apps/nvidia/containers/monai/monailabel/ monailabel datasets

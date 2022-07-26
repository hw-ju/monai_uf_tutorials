#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# show all monailabel commands to download sample apps, datasets and run server
singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel/ monailabel --help 
# List sample apps
singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel/ monailabel apps 
# List sample datasets
singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel/ monailabel datasets

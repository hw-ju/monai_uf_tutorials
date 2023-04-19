#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=8:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# download sample apps
singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel/ monailabel apps --download --name deepedit --output /workspace/apps
# download sample datasets
singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel/ monailabel datasets --download --name Task03_Liver --output /workspace/datasets

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=8:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

mkdir -p /blue/vendor-nvidia/hju/monailabel_samples

# download sample datasets
singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel datasets --download --name Task03_Liver --output /workspace/datasets

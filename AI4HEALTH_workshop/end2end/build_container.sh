#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

# build a Singularity sandbox container (container in a writable directory) from NGC RAPIDS docker image
singularity build --sandbox /blue/vendor-nvidia/hju/workshop_rapids23.02 docker://nvcr.io/nvidia/rapidsai/rapidsai-core:23.02-cuda11.8-runtime-ubuntu22.04-py3.8

# install current stable monai  
singularity exec --writable /blue/vendor-nvidia/hju/workshop_rapids23.02 pip install monai==1.1.0


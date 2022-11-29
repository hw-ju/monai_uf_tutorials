#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

# Build a MONAI Core singularity sandbox container (container in a writable directory) from MONAI Core docker image
# Find docker images for all MONAI Core versions here https://hub.docker.com/r/projectmonai/monai/tags 
singularity build --sandbox /blue/vendor-nvidia/hju/monaicore0.9.1 docker://projectmonai/monai:0.9.1

# Install all dependencies required by the MONAI Core tutorial scripts 
singularity exec --writable /blue/vendor-nvidia/hju/monaicore0.9.1 pip3 install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

# build a Singularity sandbox container (container in a writable directory) from MONAI Core docker image
singularity build --sandbox /blue/vendor-nvidia/hju/workshop_monaicore1.0.1 docker://projectmonai/monai:1.0.1

# Install all dependencies required by the MONAI Core tutorial scripts 
singularity exec --writable /blue/vendor-nvidia/hju/workshop_monaicore1.0.1 pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt


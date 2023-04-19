#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

# build a Singularity sandbox container (container in a writable directory) from monailabel docker image.
# see all versions of monailabel docker images here https://hub.docker.com/r/projectmonai/monailabel/tags
singularity build --sandbox /blue/vendor-nvidia/hju/monailabel/ docker://projectmonai/monailabel:latest

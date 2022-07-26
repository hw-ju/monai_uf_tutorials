#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3gb
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity
# build a Singularity sandbox container (container in a writable directory) from monailabel docker image
singularity build --sandbox /blue/vendor-nvidia/hju/monailabel/ docker://projectmonai/monailabel:latest

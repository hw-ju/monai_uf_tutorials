#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity
# build a Singularity sandbox container (container in a writable directory) from monailabel docker image
singularity build --sandbox /blue/vendor-nvidia/hju/workshop_monailabel0.6.0/ docker://projectmonai/monailabel:0.6.0

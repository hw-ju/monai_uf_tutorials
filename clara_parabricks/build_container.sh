#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

singularity pull /blue/vendor-nvidia/hju/clara-parabricks-4.0.1-1 docker://nvcr.io/nvidia/clara/clara-parabricks:4.0.1-1

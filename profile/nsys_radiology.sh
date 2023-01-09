#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 \
nsys profile \
--output ./output_base \
--force-overwrite true \
--trace-fork-before-exec true \
python3 $HOME/tutorials/performance_profiling/radiology/train_base_nvtx.py
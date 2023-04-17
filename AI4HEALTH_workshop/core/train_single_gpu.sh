#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# Go to the directory hosting tutorial scripts
cd /home/hju/monai_uf_tutorials/AI4HEALTH_workshop/core

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
python -c "import torch; print(torch.cuda.is_available())"

# Start jupyter lab within the MONAI Core container 
singularity exec --nv /apps/nvidia/containers/monai/core/1.0.1 python unet_training_single_gpu.py
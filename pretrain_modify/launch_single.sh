#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=4:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.9.1 python3 -c "import torch; print(torch.cuda.is_available())"

# Run a tutorial python script within the container. Modify the path to your container and your script.
#singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.8.1 python3 /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py

singularity exec --nv \
--bind /blue/vendor-nvidia/hju/data/swinunetr_pretrain_CT:/mnt \
/blue/vendor-nvidia/hju/monaicore0.9.1 \
python main.py \
--roi_x=128 --roi_y=128 --roi_z=128 \
--lrdecay --lr=6e-6 \
--batch_size=1 \
--epochs=3 --num_steps=6 --eval_num=2 \
--noamp
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

# Single GPU Pre-Training with Gradient Check-pointing
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python main.py \
# --json_list=/mnt/brats21_folds.json --data_dir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
# --feature_size=48 \
# --val_every=1 --max_epochs=2 \
# --use_checkpoint --noamp --save_checkpoint

singularity exec --nv \
--bind /blue/vendor-nvidia/hju/data/swinunetr_pretrain_CT:/mnt \
/blue/vendor-nvidia/hju/monaicore0.9.1 \
python main.py \
--roi_x=128 --roi_y=128 --roi_z=128 \
--lrdecay --lr=6e-6 \
--batch_size=1 \
--epochs=3 --num_steps=6 --eval_num=2 \
--use_checkpoint \
--noamp

# without --use_checkpoint
# resume --resume=path

# --smartcache_dataset

# --cache_dataset

# multi-gpu
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 main.py
# --batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --lr=6e-6 --decay=0.1
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.9.1 python3 -c "import torch; print(torch.cuda.is_available())"

# Run a tutorial python script within the container. Modify the path to your container and your script.
#singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.8.1 python3 /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py

# 1. train from scratch (without amp)
singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
/blue/vendor-nvidia/hju/monaicore0.9.1 \
python main.py \
--json_list=/mnt/brats21_folds.json --data_dir=/mnt \
--roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
--feature_size=48 \
--val_every=1 --max_epochs=2 \
--use_checkpoint --noamp --save_checkpoint

# 2. train from scratch (with amp)
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python main.py \
# --logdir=/mnt \
# --json_list=/mnt/brats21_folds.json --data_dir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
# --feature_size=48 \
# --val_every=1 --max_epochs=3 \
# --use_checkpoint --save_checkpoint

# 3. train from scratch (without gradient checkpointing)
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python main.py \
# --logdir=/mnt \
# --json_list=/mnt/brats21_folds.json --data_dir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
# --feature_size=48 \
# --val_every=1 --max_epochs=10 \
# --save_checkpoint

# 4. finetune a Swin UNETR model pretrained on fold 1 with gradient check-pointing and without amp
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python main.py \
# --logdir=/mnt \
# --json_list=/mnt/brats21_folds.json --data_dir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
# --feature_size=48 \
# --val_every=1 --max_epochs=2 \
# --use_checkpoint --noamp --save_checkpoint \
# --resume_ckpt \
# --pretrained_model_name=model.pt \
# --pretrained_dir=/mnt/pretrained_models/fold1_f48_ep300_4gpu_dice0_9059 \
# --fold=1

# 5. train from saved checkpoint
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python main.py \
# --logdir=/mnt \
# --json_list=/mnt/brats21_folds.json --data_dir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
# --feature_size=48 \
# --val_every=1 --max_epochs=6 \
# --use_checkpoint --noamp --save_checkpoint \
# --checkpoint=/mnt/runs/model.pt

# 6. evaluation(i.e., inference) using a pretrained model on a single GPU
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BraTS2021:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python test.py \
# --logdir=/mnt \
# --json_list=/mnt/brats21_folds.json --data_dir=/mnt \
# --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 --spatial_dims=3 \
# --feature_size=48 \
# --infer_overlap=0.6 \
# --pretrained_model_name=model.pt \
# --pretrained_dir=/mnt/pretrained_models/fold1_f48_ep300_4gpu_dice0_9059
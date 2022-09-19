#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.9.1 python3 -c "import torch; print(torch.cuda.is_available())"

# Run a tutorial python script within the container. Modify the path to your container and your script.
#singularity exec --nv /blue/vendor-nvidia/hju/monaicore0.8.1 python3 /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py

# 1. train from scratch (without amp)
singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BTCV:/mnt \
/blue/vendor-nvidia/hju/monaicore0.9.1 \
python main.py \
--logdir=/mnt \
--data_dir=/mnt --json_list=dataset_0.json \
--roi_x=96 --roi_y=96 --roi_z=96 --feature_size=48 \
--batch_size=1 \
--val_every=1 --max_epochs=2 \
--save_checkpoint \
--noamp

# 2. evaluation(i.e., inference) using a pretrained model on a single GPU
# singularity exec --nv --bind /blue/vendor-nvidia/hju/data/BTCV:/mnt \
# /blue/vendor-nvidia/hju/monaicore0.9.1 \
# python test.py \
# --logdir=/mnt \
# --data_dir=/mnt --json_list=dataset_0.json \
# --roi_x=96 --roi_y=96 --roi_z=96 --feature_size=48 \
# --infer_overlap=0.6 \
# --pretrained_dir=/mnt/runs \
# --pretrained_model_name=model.pt
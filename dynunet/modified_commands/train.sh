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

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 python3 -c \
# "import torch; print('torch.cuda.is_available'); print(torch.cuda.is_available())"

# Run a tutorial python script within the container. Modify the path to your container and your script.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 python3 \
# /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py

lr=1e-1
fold=0

singularity exec --nv \
--bind /blue/vendor-nvidia/hju/data/MSD:/mnt \
/blue/vendor-nvidia/hju/monaicore1.0.1 \
python3 /home/hju/run_monaicore/dynunet/dynunet_pipeline/train.py \
-root_dir /mnt \
-datalist_path /mnt/dynunet/config/ \
-fold $fold \
-train_num_workers 4 \
-interval 1 \
-num_samples 1 \
-learning_rate $lr \
-max_epochs 5 \
-task_id 04 \
-pos_sample_num 2 \
-expr_name baseline \
-tta_val True \
-determinism_flag True \
-determinism_seed 0
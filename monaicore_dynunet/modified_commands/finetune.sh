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

# train step 2, finetune with small learning rate
# please modify the weight variable
lr=1e-3
fold=0
weight=$HOME/run_monaicore/monaicore_dynunet/modified_commands/runs_04_fold0_baseline/net_key_metric=0.0101.pt

singularity exec --nv \
--bind /blue/vendor-nvidia/hju/data/MSD:/mnt \
/blue/vendor-nvidia/hju/monaicore1.0.1 \
python3 /home/hju/run_monaicore/monaicore_dynunet/dynunet_pipeline/train.py \
-root_dir /mnt \
-datalist_path /mnt/dynunet/config/ \
-fold $fold \
-train_num_workers 4 \
-interval 1 \
-num_samples 1 \
-learning_rate $lr \
-max_epochs 5 \
-task_id 04 \
-pos_sample_num 1 \
-expr_name baseline \
-tta_val True \
-checkpoint $weight \
-determinism_flag True \
-determinism_seed 0
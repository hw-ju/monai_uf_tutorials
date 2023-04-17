#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition=hwgui
#SBATCH --nodelist=c0308a-s9
#SBATCH --gpus=2 
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

srun --unbuffered singularity exec --nv -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel start_server --app /workspace/apps/radiology --studies /workspace/datasets/demo_Liver --conf models deepedit --conf skip_scoring false --conf skip_strategies false --conf epistemic_enabled true

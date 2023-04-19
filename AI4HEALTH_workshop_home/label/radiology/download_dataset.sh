#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

mkdir -p /blue/vendor-nvidia/hju/monailabel_samples

# download sample datasets by `monailabel datasets --download`
# singularity exec -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel datasets --download --name Task03_Liver --output /workspace/datasets

# set up a small dataset by copying from a pre-downloaded dataset on HiperGator
DATA_DIR="/blue/vendor-nvidia/hju/monailabel_samples/datasets/radiology"
mkdir -p ${DATA_DIR}
cp /apps/nvidia/containers/monai/datasets/Task09_Spleen/imagesTr/spleen_2{0,1,2}.nii.gz ${DATA_DIR}
ls ${DATA_DIR}
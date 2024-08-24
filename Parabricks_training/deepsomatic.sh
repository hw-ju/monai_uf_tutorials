#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=3:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load parabricks/4.3.1

# Set file paths
DATA_DIR="/blue/vendor-nvidia/hju/PB_training_Dungan"
INPUT_TUMOR_BAM="${DATA_DIR}/SRR7890824-WGS_FD_T.bam"
INPUT_NORMAL_BAM="${DATA_DIR}/SRR7890827-WGS_FD_N.bam"
REFERENCE="${DATA_DIR}/GRCh38.d1.vd1.fa"
OUT_VARIANTS="${DATA_DIR}/deepsomatic.vcf"
NUM_GPUS=4

# run deepsomatic (BAM ==> VCF)
pbrun deepsomatic \
--ref ${REFERENCE} \
--in-tumor-bam ${INPUT_TUMOR_BAM} \
--in-normal-bam ${INPUT_NORMAL_BAM} \
--out-variants ${OUT_VARIANTS} \
--num-gpus ${NUM_GPUS}

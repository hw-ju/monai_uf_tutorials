#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load parabricks/4.3.1

# Set file paths
DATA_DIR="/blue/vendor-nvidia/hju/PB_training_Dungan"
INPUT_TUMOR_BAM="${DATA_DIR}/SRR7890824-WGS_FD_T.bam"
INPUT_NORMAL_BAM="${DATA_DIR}/SRR7890827-WGS_FD_N.bam"
INPUT_TUMOR_RECAL_FILE="${DATA_DIR}/SRR7890824-WGS_FD_T_BQSR_REPORT.txt"
INPUT_NORMAL_RECAL_FILE="${DATA_DIR}/SRR7890827-WGS_FD_N_BQSR_REPORT.txt"
REFERENCE="${DATA_DIR}/GRCh38.d1.vd1.fa"
OUT_VCF="${DATA_DIR}/SRR7890824-SRR7890827-WGS_FD_mutect2.vcf"
TUMOR_NAME="sm_SRR7890824"
NORMAL_NAME="sm_SRR7890827"
NUM_GPUS=4

# run mutect2 (BAM ==> VCF)
pbrun mutectcaller \
--ref ${REFERENCE} \
--in-tumor-bam ${INPUT_TUMOR_BAM} \
--in-normal-bam ${INPUT_NORMAL_BAM} \
--in-tumor-recal-file ${INPUT_TUMOR_RECAL_FILE} \
--in-normal-recal-file ${INPUT_NORMAL_RECAL_FILE} \
--out-vcf ${OUT_VCF} \
--tumor-name ${TUMOR_NAME} \
--normal-name ${NORMAL_NAME} \
--num-gpus ${NUM_GPUS}

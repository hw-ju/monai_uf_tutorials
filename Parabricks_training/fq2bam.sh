#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load parabricks/4.3.1

# Set file paths 
DATA_DIR="/blue/vendor-nvidia/hju/PB_training_Dungan"
NORMAL_SAMPLE_1="${DATA_DIR}/SRR7890827_1.fastq.gz"
NORMAL_SAMPLE_2="${DATA_DIR}/SRR7890827_2.fastq.gz"
NORMAL_READ_GROUP="@RG\tID:id_SRR7890827_rg1\tLB:lib1\tPL:bar\tSM:sm_SRR7890827\tPU:pu_SRR7890827_rg1"
TUMOR_SAMPLE_1="${DATA_DIR}/SRR7890824_1.fastq.gz"
TUMOR_SAMPLE_2="${DATA_DIR}/SRR7890824_2.fastq.gz"
TUMOR_READ_GROUP="@RG\tID:id_SRR7890824_rg1\tLB:lib1\tPL:bar\tSM:sm_SRR7890824\tPU:pu_SRR7890824_rg1"

NORMAL_OUT_BAM="${DATA_DIR}/SRR7890827-WGS_FD_N.bam"
TUMOR_OUT_BAM="${DATA_DIR}/SRR7890824-WGS_FD_T.bam"

REFERENCE="${DATA_DIR}/GRCh38.d1.vd1.fa"
KNOWNSITES_1="${DATA_DIR}/Mills_and_1000G_gold_standard.indels.b38.primary_assembly.vcf.gz"
KNOWNSITES_2="${DATA_DIR}/GCF_000001405.39.vcf.gz"
KNOWNSITES_3="${DATA_DIR}/ALL.wgs.1000G_phase3.GRCh38.ncbi_remapper.20150424.shapeit2_indels.vcf.gz"

NORMAL_OUT_RECAL_FILE="${DATA_DIR}/SRR7890827-WGS_FD_N_BQSR_REPORT.txt"
NORMAL_OUT_BAM="${DATA_DIR}/SRR7890827-WGS_FD_N.bam"

TUMOR_OUT_RECAL_FILE="${DATA_DIR}/SRR7890824-WGS_FD_T_BQSR_REPORT.txt"
TUMOR_OUT_BAM="${DATA_DIR}/SRR7890824-WGS_FD_T.bam"

NUM_GPUS=4

## Aligning Normal sample FASTQ file
pbrun fq2bam \
--ref ${REFERENCE} \
--in-fq ${NORMAL_SAMPLE_1} ${NORMAL_SAMPLE_2} ${NORMAL_READ_GROUP} \
--knownSites ${KNOWNSITES_1} \
--knownSites ${KNOWNSITES_2} \
--knownSites ${KNOWNSITES_3} \
--out-recal-file ${NORMAL_OUT_RECAL_FILE} \
--bwa-options=-Y \
--out-bam ${NORMAL_OUT_BAM} \
--num-gpus ${NUM_GPUS} 

## Aligning Tumor sample FASTQ file
#pbrun fq2bam \
#--ref ${REFERENCE} \
#--in-fq ${TUMOR_SAMPLE_1} ${TUMOR_SAMPLE_2} ${TUMOR_READ_GROUP} \
#--knownSites ${KNOWNSITES_1} \
#--knownSites ${KNOWNSITES_2} \
#--knownSites ${KNOWNSITES_3} \
#--out-recal-file ${TUMOR_OUT_RECAL_FILE} \
#--bwa-options=-Y \
#--out-bam ${TUMOR_OUT_BAM} \
#--num-gpus ${NUM_GPUS}

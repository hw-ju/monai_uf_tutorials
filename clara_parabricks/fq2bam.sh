#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load singularity

# Set file paths
SINGULARITY_IMAGE="/blue/vendor-nvidia/hju/clara-parabricks-4.0.1-1"
DATA_DIR="/blue/vendor-nvidia/hju/data_parabricks/parabricks_sample"
SAMPLE_1="${DATA_DIR}/Data/sample_1.fq.gz"
SAMPLE_2="${DATA_DIR}/Data/sample_2.fq.gz"
REFERENCE="${DATA_DIR}/Ref/Homo_sapiens_assembly38.fasta"
OUT_BAM="${DATA_DIR}/output.bam"

# run PARABRICKS alignment tool - FQ2BAM (FASTA + FASTQ ==> BAM)
singularity exec \
    --nv \
    --bind ${DATA_DIR}:${DATA_DIR} \
    ${SINGULARITY_IMAGE} \
    pbrun fq2bam \
        --ref ${REFERENCE} \
        --in-fq ${SAMPLE_1} ${SAMPLE_2} \
        --out-bam ${OUT_BAM} \
        --num-gpus 4
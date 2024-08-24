#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load gatk/4.4.0.0
gatk SelectVariants \
-R GRCh38.d1.vd1.fa \
-V SRR7890824-SRR7890827-WGS_FD.FilterMutectCalls.vcf \
--select-type-to-exclude INDEL \
-O SRR7890824-SRR7890827-WGS_FD.SNV-MNV.FilterMutectCalls.vcf

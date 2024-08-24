#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load bedtools/2.30.0

bedtools intersect \
-header \
-a SRR7890824-SRR7890827-WGS_FD.SNV-MNV.FilterMutectCalls.vcf \
-b High-Confidence_Regions_v1.2.bed > SRR7890824-SRR7890827-WGS_FD.SNV-MNV.FilterMutectCalls.hc.vcf

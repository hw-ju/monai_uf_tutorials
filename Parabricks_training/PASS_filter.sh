#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

grep "#" SRR7890824-SRR7890827-WGS_FD.SNV-MNV.FilterMutectCalls.hc.vcf > mutect_header.txt

grep -v "#" SRR7890824-SRR7890827-WGS_FD.SNV-MNV.FilterMutectCalls.hc.vcf | awk '{if ($7 == "PASS")print}' > mutect_body.txt

cat mutect_header.txt mutect_body.txt > SRR7890824-SRR7890827-WGS_FD.SNV-MNV.FilterMutectCalls.hc.PASS.vcf

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

module load sra/3.0.8

cd /blue/vendor-nvidia/hju/PB_training_Dungan

## Convert SRA to FASTQ files
echo "start converting 2"
fasterq-dump --threads 16 --progress --split-files SRR7890824.sra
echo "done"
#fastq-dump --split-files ./SRR7890824.sra --gzip

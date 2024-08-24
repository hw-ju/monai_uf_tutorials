#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

cd /blue/vendor-nvidia/hju/PB_training_Dungan

## Download publicly available SRA files using wget. Both files are
# about 65 GB in size.
# Normal sample
echo "start downloading normal sample"
wget https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR7890827/SRR7890827 --output-document=SRR7890827.sra

# Tumor sample
echo "start downloading tumor sample"
wget https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR7890824/SRR7890824 --output-document=SRR7890824.sra

echo "Done"

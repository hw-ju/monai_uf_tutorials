#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=2:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

# download sample dataset
DATA_DIR="/blue/vendor-nvidia/hju/monailabel_samples/datasets/pathology"

mkdir -p ${DATA_DIR}

wget "https://demo.kitware.com/histomicstk/api/v1/item/5d5c07509114c049342b66f8/download" -O "${DATA_DIR}/JP2K-33003-1.svs"

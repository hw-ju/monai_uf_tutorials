#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

mkdir -p /blue/vendor-nvidia/hju/single_cell_models

wget -P /blue/vendor-nvidia/hju/single_cell_models https://api.ngc.nvidia.com/v2/models/nvidia/atac_bulk_lowcov_5m_50m/versions/0.3/files/models/model.pth.tar
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd

mkdir -p /blue/vendor-nvidia/hju/single_cell_data

# download dataset for example 1 and 3
wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad

# download dataset for example 2
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad

# download dataset for example 4
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_nonzeropeaks.h5ad
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_peaknames_nonzero.npy
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_cell_metadata.csv

# download dataset for example 5
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X.sparse.h5ad
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X_fragments.tsv.gz
# wget -P /blue/vendor-nvidia/hju/single_cell_data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X_fragments.tsv.gz.tbi

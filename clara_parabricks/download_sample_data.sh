#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
date;hostname;pwd


cd /blue/vendor-nvidia/hju/data_parabricks

# get the sample data
wget -O parabricks_sample.tar.gz \
"https://s3.amazonaws.com/parabricks.sample/parabricks_sample.tar.gz"

# extract the data
tar xvf parabricks_sample.tar.gz

ls .

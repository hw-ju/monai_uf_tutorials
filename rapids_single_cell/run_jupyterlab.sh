#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module add jupyter
cd $HOME
launch_jupyter_notebook
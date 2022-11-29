#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load singularity

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 python3 -c \
# "import torch; print('torch.cuda.is_available'); print(torch.cuda.is_available())"

# Run a tutorial python script within the container. Modify the path to your container and your script.
# singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 python3 \
# /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py

echo "pip3 freeze | grep ignite"
singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 \
pip3 freeze | grep ignite

echo "import ignite; print(ignite.__version__)"
singularity exec --nv /blue/vendor-nvidia/hju/monaicore1.0.1 \
python3 -c "import ignite; print(ignite.__version__)"
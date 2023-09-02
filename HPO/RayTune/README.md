# **use RayTune on HiperGator**

## **RayTune**
RayTune doc https://docs.ray.io/en/master/tune/index.html

## **Build conda env**
```
srun --nodes=1 --ntasks=1 --partition=gpu --gpus=a100:1 --cpus-per-task=4 --mem-per-cpu 64gb --time=08:00:00 --pty -u bash -i
```
```
module load conda
conda create --name ray-monai python=3.9
conda activate ray-monai

pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt
pip install monai==1.0.1 
pip install -U "ray[all]" 
```
```
# test 
python -c "import monai; monai.config.print_config()"

python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

x = torch.randn(1, 3, 224, 224, device='cuda')
conv = torch.nn.Conv2d(3, 3, 3).cuda()
out = conv(x)
print(out.shape)

torch.cuda.is_available()
```

## **How to run**
Pull this directory.

Make the scripts within this directory executable by
```
chmod -R +x this_directory
```

```
sbatch submit-ray-cluster-monai-single-node.sbatch
```
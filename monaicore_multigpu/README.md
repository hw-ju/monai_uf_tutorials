# **Run multi-gpu distributed trainings using MONAI Core on HiperGator**
This directory hosts sample scripts to launch a multi-gpu distributed training using `torch.distributed.DistributedDataParallel` and MONAI Core on UF HiperGator's AI partition, a SLURM cluster using Singularity as container runtime.

## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator).
2. In all following commands, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
3. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs.
4. Please read the comments at the beginning of each script to get a better understanding on how to tune the scripts to your own needs. 
5. If you have no experience running distributed training with MONAI Core before, I strongly recommend going through the following examples in order, i.e., from simple to complex, which .

## **Example 1: multi-gpu training of Unet**
The training python script we're using here is adapted from [a script in MONAI Core tutorial repository](https://github.com/Project-MONAI/tutorials/blob/master/acceleration/distributed_training/unet_training_ddp.py). Synthetic data will be generated, so you don't need to download any data or have your own data to use this script.

Go to directory `unet_ddp/`
```
cd ~/monai_uf_tutorials/monaicore_multigpu/unet_ddp/
```

Submit a SLURM job script `launch.sh` to launch a distributed training on a **single node**(see sample script [`launch.sh`](./unet_ddp/launch.sh))
```
sbatch launch.sh
```

Submit a SLURM job script `launch.sh` to launch a distributed training on **multiple nodes**(see sample script [`launch_multinode.sh`](./unet_ddp/launch_multinode.sh))
```
sbatch launch_multinode.sh
```

**Note**
The difference in [`launch.sh`](./unet_ddp/launch.sh) (for training on a **single node**) and [`launch_multinode.sh`](./unet_ddp/launch_multinode.sh) (for training on multiple nodes):
1. `#SBATCH` settings.
2. use `run_on_node.sh` or `run_on_multinode.sh` in line 70 
```py
PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")
```


Check SLURM output file, file name format `launch.sh.job_id.out` or `launch_multinode.sh.job_id.out` (see sample file [`launch.sh.job_id.out`](./unet_ddp/launch.sh.job_id.out)).
```
cat launch.sh.job_id.out
```

## **Example 2: multi-gpu training of Unet**



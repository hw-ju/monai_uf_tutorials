# Run multi-gpu distributed trainings using MONAI Core on HiperGator
This directory hosts sample scripts to launch a multi-gpu distributed training using MONAI Core on UF HiperGator's AI partition, a SLURM cluster using Singularity as container runtime.

## **Note**
1. This tutorial assumes you have downloaded the repository monai_uf_tutorials following [this section](../README.md/#download-this-repository-on-hipergator).
2. In all following commands, change `hju` to your HiperGator username; change the path to files according to your settings on HiperGator. 
3. In all following SLURM job scripts, alter the “#SBATCH” settings for your needs.
4. Please read the comments at the top of each script to get a better understanding on how to tune the scripts to your own needs. 

## **Example: single-node multi-gpu training of Unet by** `torch.distributed.DistributedDataParallel` 
Go to directory `unet_ddp/`
```
cd ~/monai_uf_tutorials/monaicore_multigpu/unet_ddp
```

Submit a SLURM job script `launch.sh` to launch a training (see sample script [`launch.sh`](./unet_ddp/launch.sh))
```
sbatch launch.sh
```

Check SLURM output file, file name format `launch.sh.job_id.out` (see sample file [`launch.sh.job_id.out`](./unet_ddp/launch.sh.job_id.out)).
```
cat launch.sh.job_id.out
```

## More examples will be added soon!


# **use MONAI Core dynunet pipeline (modified nnUNet pipeline) on HiperGator**

This directory hosts scripts for runnning MONAI Core's dynunet pipeline (modified nnUNet pipeline) on HiperGator, adapted from [dynunet pipeline tutorial](https://github.com/Project-MONAI/tutorials/tree/main/modules/dynunet_pipeline). 

For ease of use and in case there's breaking change in the MONAI Core tutorial scripts in the future, the current (2023/01/31) dynunet pipeline tutorial is copied in `./dynunet_pipeline` in this directory. The launch scripts in `./dynunet_pipeline/commands` are not suitable for running on HiperGator, and the sample adapted ones for HiperGator are put in `./modifeid_commands`. The `.py` scripts are not modified.  

## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator). 
2. If you have no experience running MONAI Core using Singularity as container runtime on HiperGator before, I strongly recommend going through tutorial [`monaicore_singlegpu`](../monaicore_singlegpu/) and making sure it's working before moving on to this tutorial. 
3. If you have no experience running distributed training with MONAI Core on HiperGator before, I strongly recommend going through the unet_ddp example in tutorial [`monaicore_multigpu`](../monaicore_multigpu/) and making sure it's working before moving on to this tutorial. 
4. In all following commands, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
5. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs.
6. Please read the comments at the beginning of each script to get a better understanding on how to tune the scripts to your own needs. 

## **How to run**
0. Read the original data description in [dynunet pipeline tutorial](https://github.com/Project-MONAI/tutorials/tree/main/modules/dynunet_pipeline). Download dataset. Put the dataset and its corresponding datalist `.json` file (you can copy from `./dynunet_pipeline/config`) on your HiperGator storage space. Here, I take one task, i.e., training on Task04_Hippocampus, as an example.

1. Go to directory `/monai_uf_tutorials/monaicore_dynunet/`
    ```
    cd ~/monai_uf_tutorials/monaicore_dynunet/modified_commands
    ```

2. To launch training (from scratch) on a single GPU,
    ```
    sbatch train.sh
    ```
    Note, in `train.sh`, I use MONAI Core v1.0.1 container.

    See sample SLURM output [./modified_commands/train.sh.job_id.out](./modified_commands/train.sh.job_id.out).

2. To launch finetuning (from a checkpoint) on a single GPU,
    ```
    sbatch finetune.sh
    ```
    See sample SLURM output [./modified_commands/finetune.sh.job_id.out](./modified_commands/finetune.sh.job_id.out).

4. To launch training on multiple GPUs,
    ```
    sbatch train_multi_gpu.sh
    ```    
    See sample SLURM output [./modified_commands/train_multi_gpu.sh.job_id.out](./modified_commands/train_multi_gpu.sh.job_id.out).  

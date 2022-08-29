# **use MONAI Core for Swin UNETR training and evaluation on HiperGator**

This directory hosts scripts for runnning Swin UNETR on HiperGator, adapted from [Swin UNETR BRATS21 MONAI research contribution](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21).  

## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator).
2. If you have no experience running MONAI Core using Singularity as container runtime on HiperGator before, I strongly recommend going through tutorial [`monaicore_singlegpu`](../monaicore_singlegpu/) and making sure it's working before moving on to this tutorial. 
3. If you have no experience running distributed training with MONAI Core on HiperGator before, I strongly recommend going through tutorial [`monaicore_multigpu`](../monaicore_multigpu/) and making sure it's working before moving on to this tutorial. 
4. In all following commands, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
5. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs.
6. Please read the comments at the beginning of each script to get a better understanding on how to tune the scripts to your own needs. 

## **How to run**
0. Read the original description [Swin UNETR BRATS21 MONAI research contribution](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21). Download and put required data on your HiperGator storage space.

1. Go to directory `/monai_uf_tutorials/monaicore_swinUNETR/`
    ```
    cd ~/monai_uf_tutorials/monaicore_swinUNETR/
    ```

2. To launch training/inference on a single GPU,
    ```
    sbatch launch_single.sh
    ```
   
   To launch training/inference on multiple GPUs,
    ```
    sbatch launch_multi.sh
    ```    

    **Note**
    - Launch scripts [`launch_single.sh`](./launch_single.sh) and [`launch_multi.sh`](./launch_multi.sh) includes many training options, e.g., whether use AMP, whether use gradient checkpointing, train from scratch or train from a pretrained/checkpointed model. Please read comments in the scripts.
    - In training scripts e.g., [`main.py`](./main.py), [`trainer.py`](./trainer.py), some commented out code is from original research-contribution scripts and the replacement code is usually underneath, some is for debugging purpose. Besides, there're other code modification.
    - When using validation in training with gradient checkpointing, you might see warning output `None of the inputs have requires_grad=True. Gradients will be None` which can be ignored.
    

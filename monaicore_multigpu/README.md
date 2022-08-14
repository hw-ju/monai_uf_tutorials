# **Run multi-gpu distributed trainings using MONAI Core on HiperGator**
This directory hosts sample scripts to launch a multi-gpu distributed training using `torch.distributed.DistributedDataParallel` and MONAI Core on UF HiperGator's AI partition, a SLURM cluster using Singularity as container runtime.

## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator).
2. If you have no experience running MONAI Core using Singularity as container runtime on HiperGator before, I strongly recommend going through tutorial [`monaicore_singlegpu`](../monaicore_singlegpu/) and making sure it's working before moving on to this tutorial. 
3. If you have no experience running distributed training with MONAI Core on HiperGator before, I strongly recommend trying out the following examples in order, i.e., from simple to complex, which will make debugging easier.
4. In all following commands, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
5. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs.
6. Please read the comments at the beginning of each script to get a better understanding on how to tune the scripts to your own needs. 

## **Example 1: simple multi-gpu training with Unet**
### **About this example**
1. The [training python script](./unet_ddp/unet_training_ddp.py) we're using here is adapted from [a MONAI Core tutorial script](https://github.com/Project-MONAI/tutorials/blob/master/acceleration/distributed_training/unet_training_ddp.py). 
2. Synthetic data will be generated, so you don't need to download any data or have your own data to use this script.
3. Validation is not implemented within the training loop, see other following examples for validation implementation.

### **How to run this example**
1. Go to directory `unet_ddp/`
    ```
    cd ~/monai_uf_tutorials/monaicore_multigpu/unet_ddp/
    ```

2. Submit a SLURM job script `launch.sh` to launch a distributed training on a **single node**(see sample script [`/unet_ddp/launch.sh`](./unet_ddp/launch.sh))
    ```
    sbatch launch.sh
    ```

    Alternatively, submit a SLURM job script `launch.sh` to launch a distributed training on **multiple nodes**(see sample script [`/unet_ddp/launch_multinode.sh`](./unet_ddp/launch_multinode.sh))
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

3. Check SLURM output file, file name format `launch.sh.job_id.out` or `launch_multinode.sh.job_id.out` (see sample file [`/unet_ddp/launch.sh.job_id.out`](./unet_ddp/launch.sh.job_id.out) and [`/unet_ddp/launch_multinode.sh.job_id.out`](./unet_ddp/launch_multinode.sh.job_id.out)).
    ```
    cat launch.sh.job_id.out
    ```

## **Example 2: multi-gpu training for Brain Tumor segmentation with UNet/SegResNet**
### **About this example**
1. The [training python script](./brats_ddp/brats_training_ddp.py) is adapted from [a MONAI Core tutorial script](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py). 
2. This example is a real-world task based on Decathlon challenge Task01: Brain Tumor segmentation, so it's more complicated than [Exmaple 1](#example-1-simple-multi-gpu-training-with-unet). 
2. Steps to get the required data:
    - go to http://medicaldecathlon.com/, click on `Get Data` and then download `Task01_BrainTumour.tar` to your local computer. 
    - upload it to your storage partition (e.g. blue or red partition) on HiperGator, sample command:
    ```
    scp path_to_Task01_BrainTumour.tar hju@hpg.rc.ufl.edu:path_to_storage_directory 
    ```
    - extract the data to directory `/Task01_BrainTumour`:
    ```
    tar xvf path_to_Task01_BrainTumour.tar
    ```
3. To make the data visible to the MONAI Core Singularity container, we need to bind the data directory into the container, see line 44 in [`/brats_ddp/launch.sh`](./brats_ddp/launch.sh):
    ```
    PYTHON_PATH="singularity exec --nv --bind /blue/vendor-nvidia/hju/data/brats_data:/mnt \
         /blue/vendor-nvidia/hju/monaicore0.8.1 python3" 
    ```
    note the use of `--bind` flag:
    ```
    --bind path_to data_directory_on_hipergator:directory_name_seen_by_container
    ```
    you can name `directory_name_seen_by_container` whatever you like, i.e., it doen't have to be `/mnt`. See [Singularity doc on bind paths](https://docs.sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html?highlight=bind%20mount) for more details.

    note for this example, I'm binding `/brats_data`, the parent directory of `/brats_data/Task01_BrainTumour`, to the container, which is for the sake of point 4 in this section. See [Example 3](#example-3-multi-gpu-training-for-brain-tumor-segmentation-with-unetsegresnet) for binding data directory (not its parent directory) to the container.  
4. For this [training script](./brats_ddp/brats_training_ddp.py), we also need to provide the parent directory of `mnt/Task01_BrainTumour` as an input argument. See line 39 in [`/brats_ddp/launch.sh`](./brats_ddp/launch.sh):
    ```
    TRAINING_CMD="$TRAINING_SCRIPT -d=/mnt --epochs=20"
    ```
5. Multiple fast model training techniques are used: optimizer Novograd, cache intermediate data on GPU memory, ThreadDataLoader, Automated Mixed Precision (AMP). See [Fast Model Training guide](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md) to learn more.  

6. Dataset is splitted and cached on each GPU before training, see the implementation of `BratsCacheDataset`. This can avoid duplicated caching content on each GPU, but will not do global shuffle before every epoch. If you want to do global shuffle while caching on GPUs, you can replace the `BratsCacheDataset` object with a `CacheDataset` object and a `DistributedSampler` object, where each GPU will cache the whole dataset, see [discussion](https://github.com/Project-MONAI/tutorials/discussions/672).

### **How to run this example**
Steps are similar to [Example 1](#how-to-run-this-example), except sample scripts and output files are in directory `brats_ddp/`.


## **Example 3: multi-gpu training 3D Multi-organ Segmentation with UNETR** (in preparation...)
### **About this example**
1. The [training python script](./unetr_ddp/unetr_btcv_ddp.py) is adapted from [a MONAI Core tutorial script](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb).
2. data, bind
3. single gpu
3. fast techs
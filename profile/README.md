## **Performance profiling of MONAI Core training by NVIDIA Nsight Systems on HiperGator (work in progress)**

MONAI Core tutorial repo hosts training pipeline profiling tutorials:
1. In [Fast Model Training Guide](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md), see [section 2. NVIDIA Nsight Systems](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md#2-nvidia-nsight-systems) and [section 3. NVIDIA Tools Extension (NVTX)](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md#3-nvidia-tools-extension-nvtx).
2. Profile a [radiology pipeline for spleen segmentation](https://github.com/Project-MONAI/tutorials/tree/main/performance_profiling/radiology).
3. Profile a [pathology pipeline for metastasis detection](https://github.com/Project-MONAI/tutorials/tree/main/performance_profiling/pathology).
4. ...

To learn more about NVIDIA Nsight Systems and NVTX, refer to:
1. [NVIDIA Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/). Highly recommend to check out previous [Training Seminars](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#other_resources).
2. [NVTX documentation](https://nvtx.readthedocs.io/en/latest/index.html).


This README.md shows how to first profile the above mentioned tutorial radiology pipeline within a MONAI Core Singularity container on HiperGator by Nsight Systems CLI and then visualize the generated report in Nsight Systems GUI installed on your local system. The CLI is already installed in the MONAI Core container, so you don't need to install it manually. To install Nsight Systems GUI on your local system, please refer to the [Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html).


## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator). 
2. If you have no experience running MONAI Core using Singularity as container runtime on HiperGator before, I strongly recommend going through tutorial [`monaicore_singlegpu`](../monaicore_singlegpu/) and making sure it's working before moving on to this tutorial. 
3. If you have no experience running distributed training with MONAI Core on HiperGator before, I strongly recommend going through the unet_ddp example in tutorial [`monaicore_multigpu`](../monaicore_multigpu/) and making sure it's working before moving on to this tutorial. 
4. In all following commands, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
5. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs.
6. Please read the comments in each script to get a better understanding on how to tune the scripts to your own needs. 

## **How to run**
1. Go to directory `/monai_uf_tutorials/profile/`
    ```
    cd ~/monai_uf_tutorials/profile
    ```

2. To profile the tutorial radiology pipeline,
    ```
    sbatch nsys_radiology.sh
    ```
    You should see a `.nsys-rep` report file generated, see sample report file [`output_base.nsys-rep`](./output_base.nsys-rep). Also, see sample SLURM output [`nsys_radiology.sh.job_id.out`](./profile/nsys_radiology.sh.job_id.out). 
   
3. Transfer the `.nsys-rep` report file back to your local system, see [UFRC doc on Trasfer Data](https://help.rc.ufl.edu/doc/Transfer_Data) for all available methods. E.g., we can use `scp`, 
    ```
    scp hju@hpg.rc.ufl.edu:/home/hju/monai_uf_tutorials/profile/output_base.nsys-rep .
    ```    
4. On your local system, launch Nsight Systems GUI, open the `.nsys-rep` report file.
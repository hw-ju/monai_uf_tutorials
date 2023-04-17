# AI4HEALTH Train the Trainer Workshop

## 0. Download this repository to HiperGator
Log in to HiperGator, change `hju` to your HiperGator username

```
ssh hju@hpg.rc.ufl.edu
```

Go to your home directory 

```
cd ~
``` 

Download this repository into a local directory. The default name of the directory is `monai_uf_tutorials`. 

```
git clone https://github.com/hw-ju/monai_uf_tutorials.git
```

Make all files in the directory executable

```
chmod -R +x monai_uf_tutorials/
```

## 1. MONAI Core
In this section "1. MONAI Core", we will use a **read-only** MONAI Core container prebuilt by UF Research Computing on HiperGator. If you're interested in building the same but **writable** container on your own after the workshop, you can use bash script [/core/build_container.sh](./core/build_container.sh) and refer to [use MONAI Core for single-GPU training](./monaicore_singlegpu/) for more details about building container on HiperGator.

### 1.1 Get to know MONAI Core
In this sub-section 1.1, we will run a jupter lab within the MONAI Core container to go through tutorial jupyter notebooks which were "stolen" from [the repo for MONAI Bootcamp 2023 Jan](https://github.com/Project-MONAI/monai-bootcamp/tree/main/MONAICore).

Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/core` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/core
```

Start a jupyter lab within the MONAI Core container
```
sbatch launch_jupyter_lab.sh
```

Check the SLURM output file, file name in format `launch_jupyter_lab.sh.job_id.out`. It might take a while until you see the **hostname** (e.g., c38a-s5) and **http://hostname:8888/?token=** in the [sample output file](launch_jupyter_lab.sh.job_id.out), which we will use in next steps.

Print out the output once
```
cat launch_jupyter_lab.sh.job_id.out
```
or monitor the output along it's spitting out
```
tail -f launch_jupyter_lab.sh.job_id.out  
```

Open another local terminal window, SSH to the jupyter lab. In the command below, you need to alter the **hostname** (e.g., `c38a-s5` between the 2 colons) according to the SLURM output above, alter the remote port number (e.g., `8888` following `c38a-s5:`) according to the SLURM output above (e.g., `8888` in **http://hostname:8888/?token=**), and alter `hju` to your username. 
```
ssh -NL 8888:c38a-s5:8888 hju@hpg.rc.ufl.edu
```

You will be prompted to enter your password. If the password is correct, there will be no console output; if wrong, you will be prompted to enter it again.

In a local web browser, go to `http://localhost:8888/`. If you're prompted to authenticate, copy & paste token from the SLURM output above in the top prompt box. Note: if you do not want to go through copying/pasting the token for every jupyter job, you can set a default password, see [remote jupyter notebook on HiperGator](https://help.rc.ufl.edu/doc/Remote_Jupyter_Notebook).

Open any jupyter notebook in the directory `/core` on the left pane and you're ready to run it.

Please first try out jupyter notebooks [Intro to MONAI.ipynb](./core/Intro%20to%20MONAI.ipynb) and [MONAI End-to-End Workflow - Solution.ipynb](./core/MONAI%20End-to-End%20Workflow%20-%20Solution.ipynb) in order during the workshop. We will keep [MONAI Bundle and MONAI Model Zoo.ipynb](./core/MONAI%20Bundle%20and%20MONAI%20Model%20Zoo.ipynb) as a bonus jupyter notebook if you have extra time or you can explore it after the workshop.

When you're done with a jupyter notebook, we recommend you to kill the corresponding jupyter kernel.

When you're done with all the jupyter notebooks, cancel this SLURM job
```
scancel job_id
```

### 1.2 Train on multi-gpu
In this sub-section 1.2, we'll train a Unet, first on single gpu, then on multiple gpus within a single node (using [PyTorch Distributed Data Parallel(DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and [torchrun](https://pytorch.org/docs/stable/elastic/run.html)).

Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/core` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/core
```

Train a Unet on a single gpu 
```
sbatch train_single_gpu.sh 
```
See sample output [/core/train_single_gpu.sh.job_id.out](./core/train_single_gpu.sh.job_id.out).


Train a Unet on multiple gpus 
```
sbatch train_ddp_torchrun.sh 
```
See sample output [/core/train_ddp_torchrun.sh.job_id.out](./core/train_ddp_torchrun.sh.job_id.out).

## 2. MONAI Label
In this section,


## 3. Build end-to-end AI pipeline on GPU with RAPIDS & CuPy
In this section, we will run a **read-only** RAPIDS container (on the start of the container, a jupyter lab will run by default) prebuilt by UF Research Computing on HiperGator. This container has MONAI Core pre-installed. If you're interested in building the same but **writable** container on your own after the workshop, you can use bash script [/end2end/build_container.sh](./end2end/build_container.sh) and refer to [use MONAI Core for single-GPU training](./monaicore_singlegpu/) for more details about building container on HiperGator.

We'll go through the tutorial jupyter notebook [interop_blog_adapted.ipynb](./end2end/interop_blog_adapted.ipynb) which was "stolen" from [here](https://gist.github.com/gravitino/0fd27d841c37cc25fe2032eafdc8feb2).

Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/end2end` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/end2end
```

Run the RAPIDS container (on the start of the container, a jupyter lab will run by default)
```
sbatch run_container.sh
```

SSH to the jupyter lab. Please see [section 1.1 Get to know MONAI Core](#11-get-to-know-monai-core) for how to set it up.



## Contents
1. [use MONAI Core for single-GPU training](./monaicore_singlegpu/)
2. [use MONAI Core for multi-GPU training](./monaicore_multigpu/)
3. [use MONAI Core for Swin UNETR training & evaluation for BRATS21](./monaicore_swinUNETR/)
4. [use MONAI Core for Swin UNETR self-supervised pretraining for BTCV](./pretrain_modify/)
5. [use MONAI Core for Swin UNETR training & evaluation for BTCV](./btcv_modify/)
6. [use MONAI Core for dynunet(nnUnet) training](./monaicore_dynunet/)
7. [use Datasets accelerated by caching in MONAI Core](./caching/)
8. [performance profiling of MONAI Core training by NVIDIA Nsight Systems](./profile/)
9. [Use Clara Parabricks on HiperGator](./clara_parabricks/)
10. [Run GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS on HiperGator](./rapids-single-cell/)
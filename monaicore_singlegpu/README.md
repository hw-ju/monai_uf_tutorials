# Step-by-step Tutorial: use MONAI Core by running a singularity container on a single GPU at HiperGator
## **Why use container?**
1. clean developing environment: easy to maintain dependencies of specific version, easy to debug.
2. portable(reproducible).
3. scalable.
4. MONAI Core containers are based on [NGC (NVIDIA GPU CLOUD) PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch): optimized performance on NVIDIA GPUs, keep updating.

## **Note**
1. This tutorial assumes you have downloaded the repository monai_uf_tutorials following [this section](../README.md/#download-this-repository-on-hipergator).
2. In all following commands, change `hju` to your HiperGator username; change the path to files according to your settings on HiperGator. 
3. In all following SLURM job scripts, alter the “#SBATCH” settings for your needs.


## **Major steps**
1. [Download MONAI Core tutorial repository.](#step-1-download-monai-core-tutorial-repository)
2. [Build a MONAI Core singularity container.](#step-2-build-a-monai-core-singularity-container)
3. [Run tutorial scripts within the container.](#step-3-run-tutorial-scripts-within-the-container)

## **step 1. Download MONAI Core tutorial repository**
Log in to HiperGator, change `hju` to your username

```
ssh hju@hpg.rc.ufl.edu
```

Go to your home directory 

```
cd ~
```

Download MONAI Core tutorial repository into a local directory. The default name of the directory is `tutorials`. 

```
git clone https://github.com/Project-MONAI/tutorials.git
```

Make all files in the directory executable

```
chmod -R +x tutorials/
```

## **step 2. Build a MONAI Core singularity container**
Go to directory `monaicore_singlegpu/`.

```
cd ~/monai_uf_tutorials/monaicore_singlegpu/
```

Submit a SLURM job script (see sample script [`build_container.sh`](build_container.sh)) to build a [singularity sandbox container](https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html?highlight=sandbox#creating-writable-sandbox-directories) (a writable directory) in a directory (e.g., a directory in your blue storage, don't store the container in your home directory, see [HiperGator Storage](https://help.rc.ufl.edu/doc/Storage)). See [useful SLURM Commands on HiperGator](https://help.rc.ufl.edu/doc/SLURM_Commands) to learn more about commands like `sbatch`.

```
sbatch build_container.sh
```

Sample console output

```shell
Submitted batch job 41779619  # 41779619 is the job id
```

**This job might take a long time.** Check if the job is still running (`watch` command will update the job status every 2 sec.)

```
watch squeue -u $USER
```

Sample console output
```shell
Every 2.0s: squeue -u hju                                                                       Mon Jul 11 20:23:29 2022

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          41784715       gpu build_co      hju  R       0:05      1 c38a-s5
```
(check [useful SLURM Commands on HiperGator](https://help.rc.ufl.edu/doc/SLURM_Commands) to learn more about commands like `squeue`.)

Once the job is done, check the SLURM output file, file name in format `build_container.sh.job_id.out`. See [sample output file](build_container.sh.job_id.out)). If there is no error, then the container was built successfully.

```
cat build_container.sh.job_id.out 
```


## **step 3. Run tutorial scripts within the container** 
Tutorial scripts are in 2 formats, python scripts (.py) and jupyter notebooks (.ipynb). How to run each script format is shown separately below. 

Go to directory `monaicore_singlegpu/`.

```
cd ~/monai_uf_tutorials/monaicore_singlegpu
```

### **1) Run python script .py in interactive mode**
Request resources for an interactive session and start a bash shell (learn more about [interactive session on HiperGator](https://help.rc.ufl.edu/doc/Development_and_Testing), learn more about [GPU access on HiperGator](https://help.rc.ufl.edu/doc/GPU_Access)). 

To request an A100 GPU:

```
srun --nodes=1 --ntasks=1 --partition=gpu --gpus=a100:1 --cpus-per-task=4 --mem-per-cpu 64gb --time=03:00:00 --pty -u bash -i
```

To request a GeForce GPU:

```
srun --nodes=1 --ntasks=1 --partition=gpu --gpus=geforce:1 --cpus-per-task=4 --mem-per-cpu 64gb --time=03:00:00 --pty -u bash -i
```

When you see the prompt changes from something like ```[hju@login5 monaicore_singlegpu]$``` to something like ```[hju@c39a-s39 monaicore_singlegpu]$```, you have successfully hopped from the login node to a compute node. Sample console output:

```
srun: job 41777769 queued and waiting for resources
srun: job 41777769 has been allocated resources
[hju@c39a-s39 monaicore_singlegpu]$
```

Load singularity.

```
module load singularity
```

Spawn a new shell within your container and interact with it. Alter the path to your container. 

```
singularity shell --nv /blue/vendor-nvidia/hju/monaicore0.9.1
```

It might hang for a while before any console output. The change in prompt (from hju@... to Singularity>) indicates that you have entered the container. Sample console output:

```
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (484) bind mounts  # can be ignored
"uname": executable file not found in $PATH  # can be ignored
Singularity>
```

Run a python script.

```
python3 /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py
```

The first time you run scripts within a newly spun-up container, it might hang for quite a while before you see any console output, and the following runs will be much faster to see output. See [sample console output](interactive_python_console.out).

Exit the container.

```
Ctrl + D
```

Exit the compute node and go back to the login node.

```
Ctrl + D
```

### **2) Run a python script .py in batch mode**
Submit a SLURM job script. See sample job script [`run_container.sh`](run_container.sh).

```
sbatch run_container.sh
```

Check the SLURM output file, file name in the format `run_container.sh.job_id.out`. See [sample output file](run_container.sh.job_id.out).

```
cat run_container.sh.job_id.out
```

or you can use `tail` command to follow the SLURM output while it's spitting out. 
```
tail -f run_container.sh.job_id.out
```

### **3) Run jupyter notebooks .ipynb by SSH tunneling**
Submit a SLURM job script to launch a jupyter lab server (see sample script [`launch_jupyter_lab.sh`](launch_jupyter_lab.sh)). Note in the SLURM job script, do not set cpu memory too small, otherwise some cells (e.g., cells for training) in the jupyter notebooks cannot execute. 

```
sbatch launch_jupyter_lab.sh
```

Check the SLURM output file, file name in format `launch_jupyter_lab.sh.job_id.out`. It might take a while until you see the **hostname** (e.g., c38a-s5) and **http://hostname:8888/?token=** in the [sample output file](launch_jupyter_labe.sh.job_id.out), which we will use in next steps.

```
cat launch_jupyter_lab.sh.job_id.out
```

Open another local terminal window, SSH to the server host, alter the **hostname** (e.g., `c1000a-s17` between the 2 colons) according to the SLURM output above, alter `hju` to your username. 

```
ssh -NL 8888:c38a-s5:8888 hju@hpg.rc.ufl.edu
```

You will be prompted to enter your password. If the password is correct, there will be no console output; if wrong, you will be prompted to enter it again.

In a web browser, go to `http://localhost:8888/`. You might be prompted to authenticate as shown in the screenshot below. Copy & paste token from the SLURM output above in the prompt box. Note: if you do not want to go through copying/pasting the token for every jupyter job, you can set a default password, see [remote jupyter notebook on HiperGator](https://help.rc.ufl.edu/doc/Remote_Jupyter_Notebook).

Open any jupyter notebooks in the directory `tutorials/` on the left pane (e.g. ~/tutorials/2d_classification/mednist_tutorial.ipynb) and you're ready to run.

















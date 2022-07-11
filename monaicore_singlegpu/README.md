# Step-by-step Tutorial: use MONAI Core by running a singularity container on a single GPU at HiperGator
## **Why use container?**
1. clean developing environment: easy to maintain dependencies of specific version, easy to debug.
2. portable(reproducible).
3. scalable.
4. MONAI Core containers are based on [NGC (NVIDIA GPU CLOUD) PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch): optimized performance on NVIDIA GPUs, keep updating.

## **Note**
1. This tutorial assumes you have downloaded the repository monai_uf_tutorials following [this section](../README.md/#download-this-repository-on-hipergator).
2. In all following commands, change “hju” to your HiperGator username; change the path to files according to your settings on HiperGator. 
3. In all following SLURM job scripts, alter the “#SBATCH” settings for your needs.


## **Major steps**
1. [Download MONAI Core tutorial repository.](#step-1-download-monai-core-tutorial-repository)
2. [Build a MONAI Core singularity container.](#step-2-build-a-monai-core-singularity-container)
3. [Run tutorial scripts within the container.](#step-3-run-tutorial-scripts-within-the-container)

## **step 1. Download MONAI Core tutorial repository**
Log in to HiperGator, change `hju` to your username

```bash
ssh hju@hpg.rc.ufl.edu
```

Go to your home directory 

```bash
cd ~
```

Download MONAI Core tutorial repository into a local directory. The default name of the directory is `tutorials`. 

```bash
git clone https://github.com/Project-MONAI/tutorials.git
```

Make all files in the directory executable

```bash
chmod -R +x tutorials
```

## **step 2. Build a MONAI Core singularity container**
Submit a SLURM job script (see sample script [`build_container.sh`](build_container.sh)) to build a [singularity sandbox container](https://docs.sylabs.io/guides/3.7/user-guide/build_a_container.html?highlight=sandbox#creating-writable-sandbox-directories) (a writable directory) in a directory (e.g., a directory in your blue storage, don't store the container in your home directory, see [HiperGator Storage](https://help.rc.ufl.edu/doc/Storage)). This job might take a long time. 

```bash
sbatch build_container.sh
```

Check the SLURM output file (see sample file [`build_container.sh.41212325.out`](build_container.sh.41212325.out)). If there is no error, then the container was built successfully.

```bash
cat SLURM_output_file 
```


## **step 3. Run tutorial scripts within the container** 
Tutorial scripts are in 2 formats, python scripts (.py) and jupyter notebooks (.ipynb). How to run each script format is shown separately below. 

### **1) Run python script .py in interactive mode**
Request resources for an interactive session and start a bash shell (learn more about [interactive session on HiperGator](https://help.rc.ufl.edu/doc/Development_and_Testing), learn more about [GPU access on HiperGator](https://help.rc.ufl.edu/doc/GPU_Access)). 

To request an A100 GPU:

```bash
srun --nodes=1 --ntasks=1 --partition=gpu --gpus=a100:1 --cpus-per-task=2 --mem-per-cpu 64gb --time=03:00:00 --pty -u bash -i
```

To request a GeForce GPU:

```bash
srun --nodes=1 --ntasks=1 --partition=gpu --gpus=geforce:1 --cpus-per-task=2 --mem-per-cpu 64gb --time=03:00:00 --pty -u bash -i
```

You will see output similar to the screenshot below. 


Load singularity
```bash
module load singularity
```

Spawn a new shell within your container and interact with it. Alter the path to your container. It might hang for a while before any output. The change in prompt (from hju@... to Singularity>) indicates that you have entered the container, as shown in the screenshot below.

```bash
singularity shell --nv /blue/vendor-nvidia/hju/monaicore0.8.1
```

Run a python script. See screenshot below for sample output of this script. 

```bash
python3 /home/hju/tutorials/2d_segmentation/torch/unet_training_array.py
```

### **2) Run a python script .py in batch mode**
Submit a SLURM job script. See sample job script [`run_container.sh`](run_container.sh).

```bash
sbatch run_container.sh
```

Check the SLURM output file (see sample file [`run_container.sh.41299884.out`](run_container.sh.41299884.out)).

```bash
cat run_container_4934380.log
```

### **3) Run jupyter notebooks .ipynb by SSH tunneling**
Submit a SLURM job script to launch a jupyter lab server (see sample script [`launch_jupyter_lab.sh`](launch_jupyter_lab.sh)). Note in the SLURM job script, do not set cpu memory too small, otherwise some cells (e.g., cells for training) in the jupyter notebooks cannot execute. 

```bash
sbatch launch_jupyter_lab.sh
```

Check the SLURM output file (see screenshot below). We will need the hostname (e.g., c1000a-s17) and token from the output file in next steps.

```bash
cat jupyter_lab_4913076.log
```


Open another local terminal, SSH to the server host, alter the hostname (between the 2 colons) according to the SLURM output above, alter `hju` to your username. Then you will be prompted to enter your password. If the password is correct, there will be no output, as shown in the sample screenshot below; if wrong, you will be prompted to enter it again.

```bash
ssh -NL 8888:c1000a-s17:8888 hju@hpg.rc.ufl.edu
```

In a web browser, go to `http://localhost:8888/`. You might be prompted to authenticate as shown in the screenshot below. Copy & paste token from the SLURM output above in the prompt box. Note: if you do not want to go through copying/pasting the token for every jupyter job, you can set a default password, see [remote jupyter notebook on HiperGator](https://help.rc.ufl.edu/doc/Remote_Jupyter_Notebook).

Open any jupyter notebooks (.ipynb) in the directory “tutorials” on the left pane and you're ready to run. A sample screenshot is shown below.

















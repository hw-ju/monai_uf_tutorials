# AI4HEALTH Train the Trainer Workshop
**1. HiPerGator access for AI4Health workshop:**
- For new users: 
    - Please review the [New User Guide](https://help.rc.ufl.edu/doc/New_user_training) and get yourself familiar with the environment. 
    - Your user group is “ai-workshop”. 
    - Please do not run jobs from your home directory under /home/YOUR-USER-NAME. Please use the space under /blue/ai-workshop/YOUR-USER-NAME to run jobs. 

- For existing users:
    - You have been added to group “ai-workshop” as your secondary group. 
    - Please see [instructions](https://help.rc.ufl.edu/doc/Checking_and_Using_Secondary_Resources) on using secondary group’s resources to run jobs. 
    - If you would like to use ai-workshop group’s storage space for the workshop, please create your own sub-folders using the following command:
    ```
    $ cd /blue/ai-workshop
    $ mkdir YOUR-USER-NAME
    ```
- *All commands and scripts in this directory assumes that you donwload this monai_uf_tutorials repo to your home directory, which will take 3.2G in total to run all the hands-ons in this workshop.*  

**2. Computer resource reservation for AI4Health workshop:** 
- We have reserved 16 A100 nodes. The reservation name is "ai4health".
- We have reserved 2 HWGUI nodes. The reservation name is "ai4health-hwgui".
- *All sbatch and srun commands in this `README.md` assume using the above reservations, i.e., sbatch and srun commands have flags `--account=ai-workshop --qos=ai-workshop --reservation=ai4health`.*
- *In the form for scheduling an OOD session, we put `ai-workshop` in both `SLURM Account` box and `QoS` box, and put `ai4health` in `reservation` box to use the reservation.*
    ```
    sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health --account=ai-workshop --qos=ai-workshop --reservation=ai4health my_script.sh
    ```
    ```
    srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui my_script.sh
    ```


**3. NOTE:** `pip install` any package directly on a supercomputer like HiperGator (**including within Jupyter Notebook cells**) is bad practice, see the HiperGator doc on this https://help.rc.ufl.edu/doc/Managing_Python_environments_and_Jupyter_kernels. If a package is both `pip install`ed directly in your local environment and is available within a container, you might end up using the `pip install`ed one when you run the container. Please remove any previous directly `pip install`ed MONAI Core before proceeding to use the containers prebuilt for this workshop.


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
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health launch_jupyter_lab.sh
```

Check the SLURM output file, file name in format `launch_jupyter_lab.sh.job_id.out`. It might take a while until you see the **hostname** (e.g., c38a-s5) and **http://hostname:8888/?token=** in the sample output file [/core/launch_jupyter_lab.sh.job_id.out](./core/launch_jupyter_lab.sh.job_id.out), which we will use in next steps.

Print out the output once
```
cat launch_jupyter_lab.sh.job_id.out
```
or monitor the output along it's spitting out
```
tail -f launch_jupyter_lab.sh.job_id.out  
```

Open a new local terminal window, SSH to the jupyter lab. In the command below, you need to alter the **hostname** (e.g., `c38a-s5` between the 2 colons) according to the SLURM output above, alter the remote port number (e.g., `8888` following `c38a-s5:`) according to the SLURM output above (e.g., `8888` in **http://hostname:8888/?token=**), and alter `hju` to your username. 
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
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health train_single_gpu.sh 
```
See sample output [/core/train_single_gpu.sh.job_id.out](./core/train_single_gpu.sh.job_id.out).


Train a Unet on multiple gpus 
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health train_ddp_torchrun.sh 
```
See sample output [/core/train_ddp_torchrun.sh.job_id.out](./core/train_ddp_torchrun.sh.job_id.out).

## 2. MONAI Label
In this section, we will run a **read-only** MONAI Label container prebuilt by UF Research Computing on HiperGator as the server, and we will run medical image viewer 3DSlicer (for radiology applications) and QuPath (for pathology applications) as the client, respectively.

If you're interested in building the same but **writable** MONAI Label container on your own after the workshop, you can use bash script [/label/build_container.sh](./label/build_container.sh) and refer to [use MONAI Core for single-GPU training](./monaicore_singlegpu/) for more details about building container on HiperGator.

### 2.1 List MONAI Label commands
In this sub-section, let's see all available MONAI Label commands and sample applications/datasets that can be easily downloaded by MONAI Label commands.

Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/label` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/label
```

List
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health list.sh
```
See sample output [/label/list.sh.job_id.out](./label/list.sh.job_id.out).

### 2.2 Radiology applications
In this sub-section, we will do two applications showing how to use MONAI Label server + 3DSlicer  for annotation at stage 1 (cold start) and stage 2 (start from a decent pretrained model).

#### Set up server
Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/label/radiology` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/label/radiology
```

Download the sample radiology applications
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health download_app.sh
```
See sample output [/label/radiology/download_app.sh.job_id.out](./label/radiology/download_app.sh.job_id.out).

Get the sample radiology dataset. To save time, instead of using command `monailabel datasets --download` to download a large dataset as commented out in [/label/radiology/download_dataset.sh](./label/radiology/download_dataset.sh), we will copy some images from a pre-downloaded dataset on HiperGator to set up a smaller dataset.
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health download_dataset.sh
```
See sample output [/label/radiology/download_dataset.job_id.out](./label/radiology/download_dataset.job_id.out).


As running 3DSlicer on `partition=hwgui` on HiperGator gives us GPU-accelerated visualization, we'll run both MONAI Label server and 3DSlicer on the same node in `partition=hwgui` to minimize server-client communication latency. 

**NOTE**: For a hwgui node, sum of the compute resources for server and client can’t exceed:
32 CPU cores, 8 gpus, 186 mem.


Find any idle hwgui node
```
sinfo -p hwgui
```
Copy an idle node name somewhere as we will use it later.

Schedule an interactive session on HiperGator. 
    Set `--partition=hwgui` in the `srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui` command below.
    Use the idle node name from above to set `--nodelist` in the `srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui` command below.
```
srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui --ntasks=1 --nodes=1 --cpus-per-task=4 --mem=64gb --partition=hwgui --nodelist=c0308a-s9 --gpus=1 --time=01:00:00 --pty -u bash -i
```
You should see output similar to below
```
srun: job 62051871 queued and waiting for resources
srun: job 62051871 has been allocated resources
```
and the prompt is changed, e.g., from `hju@login1` to `hju@c0801a-s35`, which means that you left a login node and jumped on a compute node.

Load Singularity module
```
module load singularity
```

**Start a server**
Edit the configuration file of your chosen application to e.g., set label names, whether use pre-trained model, choose active learning strategies. The configuration file is in directory `/apps` where you downloaded the sample app.

1. Stage 1 - cold start (no labels/pretrained models available), manual from-scratch annotation.

Modify `apps/radiology/lib/configs/deepedit.py`:
    1. set single-label instead of multi-label
    2. not use pretrained model 
```
vi /blue/vendor-nvidia/hju/monailabel_samples/apps/radiology/lib/configs/deepedit.py
```

```
singularity exec --nv -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel start_server --app /workspace/apps/radiology --studies /workspace/datasets/radiology --conf models deepedit --conf skip_scoring false --conf skip_strategies false --conf epistemic_enabled true
```

2. Stage 2 - decent pretrained model, faster annotation
To remove the history from the Stage 1 applicaiton above and start this application from a fresh setting, you can do the following three steps before start a MONAI Label server.
```
rm -r /blue/vendor-nvidia/hju/monailabel_samples/apps/radiology
```

rm dataset labels
```
cd /blue/vendor-nvidia/hju/monailabel_samples/datasets/radiology
rm datastore_v2.json
rm -r labels
```

```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health download_app.sh
```

```
singularity exec --nv -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel start_server --app /workspace/apps/radiology --studies /workspace/datasets/radiology --conf models deepedit
```

Now, the server will keep outputing, which is easy to see what's going on on the server and to debug. We don't recommend to start the server using `sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health` as it might give you long latency to see the server output.

#### Set up client
Go to https://ood.rc.ufl.edu, launch a OOD session for application `Console` with `--partition=hwgui`, `--gpus=1`, and `--nodelist=the_node_server_runs_on`(e.g. `--nodelist=c0308a-s9`) in `Additional SLURM Options.

Load 3DSlicer module
```
module load slicer/5.2.1
```

Start 3DSlicer
```
Slicer
```

In 3DSlicer, switch to MONAI Label module, and connect with the server by putting the correct node name in field `MONAI Label server:` and then clicking the right button.

### 2.3 Pathology applicaitons
In this sub-section, we will follow [the MONAI Label with QuPath tutorial](https://github.com/Project-MONAI/tutorials/blob/main/monailabel/monailabel_pathology_nuclei_segmentation_QuPath.ipynb) with some HiperGator-specific modifications.

#### Set up server
Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/label/pathology` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/label/pathology
```

Download the sample pathology applications
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health download_app.sh
```
See sample output [/label/pathology/download_app.job_id.out](./label/pathology/download_app.job_id.out).

Download the sample pathology dataset
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health download_dataset.sh
```
See sample output [/label/pathology/download_dataset.job_id.out](./label/pathology/download_dataset.job_id.out).

Schedule an interactive session on HiperGator. 
If your client will run on `partition=hwgui` on HiperGator, before running the `srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui` command below, run command `sinfo -p hwgui` to find any idle hwgui node and use it to set `--nodelist` in the `srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui` command. Also, set `--partition=hwgui` in the `srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui` command.
If your client will run locally, the above setting is not needed.
```
srun --account=ai-workshop --qos=ai-workshop --reservation=ai4health-hwgui --ntasks=1 --nodes=1 --cpus-per-task=4 --mem=64gb --partition=gpu --gpus=a100:1 --time=01:00:00 --pty -u bash -i
```
You should see output similar to below
```
srun: job 62051871 queued and waiting for resources
srun: job 62051871 has been allocated resources
```
and the prompt is changed, e.g., from `hju@login1` to `hju@c0801a-s35`, which means that you left a login node and jumped on a compute node. Copy the hostname (e.g., `c0801a-s35`) somewhere, we'll use it for SSH tunneling later.

Load Singularity module
```
module load singularity
```

Start a server running the sample pathology applications
```
singularity exec --nv -B /blue/vendor-nvidia/hju/monailabel_samples:/workspace /apps/nvidia/containers/monai/monailabel.0.6.0/0.6.0 monailabel start_server --app /workspace/apps/pathology --studies /workspace/datasets/pathology
```
Now, the server will keep outputing, which is easy to see what's going on on the server and to debug.

#### Set up client

Since the installation of QuPath and the latest MONAI Label plugin for QuPath on HiperGator is still in progress, we will run QuPath on a local machine, and then SSH tunnel to the server running on HiperGator. 

To install QuPath and the MONAI Label plugin on your local machine, refer to section "2. Install QuPath and MONAI Label Plugin" in [the MONAI Label with QuPath tutorial](https://github.com/Project-MONAI/tutorials/blob/main/monailabel/monailabel_pathology_nuclei_segmentation_QuPath.ipynb).

Open a new local terminal window, SSH tunnel to the server. In the command below, you need to alter the **hostname** (e.g., `c38a-s5` between the 2 colons) accordingly (you can find it in the prompt, e.g., `hju@c0801a-s35`), and alter `hju` to your username. 
```
ssh -NL 8000:c0308a-s9:8000 hju@hpg.rc.ufl.edu
```

In QuPath, make sure MONAILabel Server URL (Host+Port) is correct through `Preferences`:
Edit -> Preferences -> MONAI Label  http://127.0.0.1:8000

**Use MONAI Label with QuPath**
Follow [the MONAI Label with QuPath tutorial](https://github.com/Project-MONAI/tutorials/blob/main/monailabel/monailabel_pathology_nuclei_segmentation_QuPath.ipynb) from section "3. Nuclei Auto Segmentation with QuPath" to the bottom.

## 3. Build end-to-end AI pipeline on GPU with RAPIDS, CuPy & Numba
In this section, we will run a **read-only** RAPIDS container (on the start of the container, a jupyter lab will run by default) prebuilt by UF Research Computing on HiperGator. This container has MONAI Core pre-installed. If you're interested in building the same but **writable** container on your own after the workshop, you can use bash script [/end2end/build_container.sh](./end2end/build_container.sh) and refer to [use MONAI Core for single-GPU training](./monaicore_singlegpu/) for more details about building container on HiperGator.

We'll go through the tutorial jupyter notebook [interop_blog_adapted.ipynb](./end2end/interop_blog_adapted.ipynb) which was "stolen" from [here](https://gist.github.com/gravitino/0fd27d841c37cc25fe2032eafdc8feb2) with some modifications to run with newer version of RAPIDS. There'a [variant version] of this jupyter notebook (https://github.com/nvahmadi/NVIDIA_IKIM_Workshop/blob/main/exercise4_zerocopy/interoperability_zerocopy.ipynb) used at another workshop (including code for plotting heartbeats sampled from latent space, link to a **cheat sheet for data converting between frameworks**, etc.).

Go to directory `/monai_uf_tutorials/AI4HEALTH_workshop/end2end` 
```
cd ~/monai_uf_tutorials/AI4HEALTH_workshop/end2end
```

Run the RAPIDS container (on the start of the container, a jupyter lab will run by default)
```
sbatch --account=ai-workshop --qos=ai-workshop --reservation=ai4health run_container.sh
```
See sample SLURM output [/end2end/run_container.sh.job_id.out](./end2end/run_container.sh.job_id.out).

SSH to the jupyter lab. Please see [section 1.1 Get to know MONAI Core](#11-get-to-know-monai-core) for how to set it up.
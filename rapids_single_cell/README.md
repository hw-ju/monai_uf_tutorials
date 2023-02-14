# **Run GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS on HiperGator**

This directory shows how to run the tutorial for **GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS** hosted in [https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples](https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) on HiperGator.  

## **Note**
1. This tutorial assumes you have downloaded the repository `monai_uf_tutorials` following [this section](../README.md/#download-this-repository-on-hipergator).
2. In all following commands and scripts, replace `hju` with your HiperGator username; change the path to files according to your own settings on HiperGator. 
3. In all following SLURM job scripts, alter the `#SBATCH` settings for your own needs.
4. Please read the comments in each script to get a better understanding on how to tune the scripts to your own needs.
5. **In all tutorial jupyter notebooks, modify the paths to the dataset and models.**

## **How to run**
0. Read the original tutorial [README](https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) to get an overview of each tutorial jupyter notebook and the datasets/pretrained-models required by each one. 

1. Download datasets by [`download_datasets.sh`](./download_datasets.sh)    
    ```
    sbatch download_datasets.sh
    ```

    For example 5, to run inference on a pretrained (AtacWorks)[https://github.com/NVIDIA-Genomics-Research/AtacWorks] model, download the pretrained model by using [`download_models.sh`](./download_models.sh) 
    ```
    sbatch download_models.sh

2. You can setup the environment required to run this tutorial in **two** ways on HiperGator: 
- **Pull/build a Singularity container** based on an all-in-one container image with all dependencies, notebooks and source code. I.e., you don't need to install any dependencies or download the tutorial notebooks yourself, because everything is in the container out-of-box.<br/><br/>**Note**, the current container image has two issues below:
    - The tutorial jupyter notebooks within the container are not the [lastest version on the repo](https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples). The latest version leverages the lastest features in RAPIDS and gives you better performance.
    - In example 5, we can't run inference with the pretrained (AtacWorks)[https://github.com/NVIDIA-Genomics-Research/AtacWorks] model on A100 on HiperGator.

    The above two issues will be fixed by next container image release.
    
    See section [**Run in all-in-one tutorial container**](#run-in-all-in-one-tutorial-container) for building a container and run within the container on HiperGator.

- **Create a conda environment** with all dependencies. I.e., you need to install dependencies and download the tutorial jupyter notebooks yourself.
    
    Jupyter notebooks for example 3 and 5 cannot be run properly now, and I'm working on that.

    See section [**Run in conda environment**](#run-in-conda-environment) for creating a conda environement and run the jupyter kernel based on the environment on HiperGator.


### **Run in conda environment**
1. Go to your home directory 

    ```
    cd ~
    ``` 
    Download the [rapids-single-cell-examples repository](https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) into a local directory. The default name of the directory is `rapids-single-cell-examples`. 

    ```
    git clone https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples.git
    ```

    Make all files in the directory executable

    ```
    chmod -R +x rapids-single-cell-examples/
    ```

1. Go to directory `/monai_uf_tutorials/rapids_single_cell/` (which hosts useful scripts for setting up conda environment and run tutorials on HiperGator)
    ```
    cd ~/monai_uf_tutorials/rapids_single_cell/
    ```

2. Create a conda environment and build a jupyter kernel based on it by submitting [`create_conda_env.sh`](./create_conda_env.sh). 
    
    Use [`create_conda_env.sh`](./create_conda_env.sh) for exmaple 1, 2, 4 and 5, and use [`create_conda_env_example3.sh`](./create_conda_env_example3.sh) for exmaple 3.
    ```
    sbatch create_conda_env.sh
    ```

3. Launch a jupyter lab on HiperGator, open up jupyter notebooks in `~/rapids-single-cell-examples`, choose the correct kernel (e.g., Python (rapidgenomics-viz) for example 3) and you can run it now.

    You can run jupyter lab in multiple ways on HiperGator, refer to [UFRC doc on this](https://help.rc.ufl.edu/doc/Jupyter_Notebooks). Below is using the SSH-tunneling way.

    Launch a jupyter lab by submitting [`run_jupyterlab.sh`](./run_jupyterlab.sh). 

        ```
        sbatch run_jupyterlab.sh
        ```

    Open the SLURM job output. You might need to wait a while for the jupyter lab to launch to see the output like the sample SLURM job output [`run_jupyterlab.sh.job_id.out`](./run_jupyterlab.sh.job_id.out) 
    ```
    cat run_jupyterlab.sh.job_id.out
    ``` 

    Open another local terminal window, copy and paste the `ssh -NL` line from the above SLURM job output file. The `ssh -NL` line should look similar to 
    ```
    ssh -NL 20515:c1004a-s17.ufhpc:20515 hju@hpg.rc.ufl.edu
    ```
    You will be prompted to enter your password. If the password is correct, there will be no console output; if wrong, you will be prompted to enter it again.

    In the bottom of this SLURM job output file, copy and paste either of the line under "Or copy and paste one of these URLs:" from the above SLURM job output file into your local web browser to see the jupyter lab. You should look for something like 
    ```
    Or copy and paste one of these URLs:
        http://c1100a-s17.ufhpc:28389/?token=17e964c5a99580cd9cef391b72c35a8f43d225105d277db2
     or http://127.0.0.1:28389/?token=17e964c5a99580cd9cef391b72c35a8f43d225105d277db2
    ```
    On the left pane, you should see all tutorial jupyter notebooks in the directory `rapids-single-cell-examples` and you're ready to run.  

4. For example 3, to use the interactive dashboard in your local browser, open another terminal window and use 
    ```
    ssh -NL 5000:hostname:5000 hju@hpg.rc.ufl.edu
    ``` 
    Then go to `http://localhost:5000/` in your local browser to use the interactive dashboard.


### **Run in all-in-one tutorial container**
1. Go to directory `/monai_uf_tutorials/rapids_single_cell/`
    ```
    cd ~/monai_uf_tutorials/rapids_single_cell/
    ```

2. Build an all-in-one container by submitting a SLURM job script [`build_container.sh`](./build_container.sh)
    ```
    sbatch build_container.sh
    ```
    See sample output [`build_container.sh.job_id.out`](./build_container.sh.job_id.out).

3. Run the container (where a jupyter lab will be launched automatically) by using [`run_container.sh`](./run_container.sh).
    ```
    sbatch run_container.sh
    ```

    It will take a while to set up a jupyter lab. Monitor the SLURM job output file until you see something like the sample output file [`run_container.sh.job_id.out`](./run_container.sh.job_id.out) by

    ```
    tail -f run_container.sh.job_id.out
    ```
    You should wait to see a line similar to 
    ```
    [I 2023-02-10 15:18:30.130 ServerApp] http://c0900a-s23.ufhpc:8888/lab
    ```
    We will use the **hostname** (e.g., `c0900a-s23`) and the **port** (e.g., `8888`) in this line next step.
    
4. Open another local terminal window, and then `ssh tunneling` to the server host by executing the command below. **Note**, alter the **hostname** (e.g., `c0900a-s23` between the 2 colons) and the **port** (e.g., `8888`) according to your SLURM output above, alter `hju` to your username 

    ```
    ssh -NL 8888:c0900a-s23:8888 hju@hpg.rc.ufl.edu
    ```

    You will be prompted to enter your password. If the password is correct, there will be no console output; if wrong, you will be prompted to enter it again.

5. In a local web browser, go to `http://localhost:port/` (e.g., http://localhost:8888). If you are prompted to authenticate, copy & paste the token from the SLURM output above into the prompt box. Note: if you do not want to go through copying/pasting the token for every jupyter job, you can set a default password, see [remote jupyter notebook on HiperGator](https://help.rc.ufl.edu/doc/Remote_Jupyter_Notebook).

    Choose `Python 3` as the kernel.

    On the left pane, you should see all tutorial jupyter notebooks in the directory `rapids-single-cell-examples` and you're ready to run. 

6. For example 3, to use the interactive dashboard in your local browser, refer to **step 4** in [**`Run in conda environment`**](#run-in-conda-environment)

7. For example 5, to run inference on a pretrained atacworks model, use [`run_container_example5.sh`](./run_container_example5.sh) to run the container as you need to bind the path of the pretrained model to the container.
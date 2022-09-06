#  Run a zero-copy end-to-end pipeline using CuPy, RAPIDS, Numba & MONAI Core.
Demo at UFRC BoF `Accelerated general data science in medicine with RAPIDS and more`.

## Script
[Jupyter Noteboook](./) adapted from [here](https://gist.github.com/gravitino/0fd27d841c37cc25fe2032eafdc8feb2) was shown at demo. 
There'a [variant version](https://github.com/nvahmadi/NVIDIA_IKIM_Workshop/blob/main/exercise4_zerocopy/interoperability_zerocopy.ipynb) used at another workshop  (including code for plotting heartbeats sampled from latent space, link to a cheat sheet for data converting between frameworks, etc.). 

## Set up environment
For running on both HiperGator and local workstation, a conda environment was created to install RAPIDS and MONAI Core. Note you need to refer to [RAPIDS release selector](https://rapids.ai/start.html#get-rapids) to get the installation command suitable for your systems. Below are the commands used to set up the conda environment:

```
conda update conda
conda create -n rapids_monaicore
conda activate rapids_monaicore
conda install -c conda-forge monai
conda install -c rapidsai -c nvidia -c conda-forge rapids=22.08 python=3.9 cudatoolkit=11.5 jupyterlab
pip install jupyterlab_nvdashboard
# check if the extension nvdashboard is enabled
jupyter server extension list
jupyter labextension list
# if run on local workstation
jupyter lab
```

For running on HiperGator, refer to [this tutorial](https://help.rc.ufl.edu/doc/Managing_Python_environments_and_Jupyter_kernels) about building a customized jupyter kernel from a conda environment.
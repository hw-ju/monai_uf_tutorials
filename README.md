# monai_uf_tutorials
This repository hosts tutorials for using [MONAI](https://monai.io/) on UF HiperGator. 

## Download this repository on HiperGator
Log in to HiperGator, change `hju` to your HiperGator username

```bash
ssh hju@hpg.rc.ufl.edu
```

Go to your home directory 

```bash
cd ~
``` 

Download this repository into a local directory. The default name of the directory is `monai_uf_tutorials`. 

```bash
git clone https://github.com/hw-ju/monai_uf_tutorials.git
```

Make all files in the directory executable

```bash
chmod -R +x monai_uf_tutorials
```

## Contents
1. [use MONAI Core for single-GPU training](./monaicore_singlegpu/)
2. [use MONAI Core for multi-GPU training](./monaicore_multigpu/)
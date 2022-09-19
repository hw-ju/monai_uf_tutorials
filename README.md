# monai_uf_tutorials
This repository hosts tutorials for using [MONAI](https://monai.io/) on UF HiperGator. 

## Download this repository on HiperGator
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

## Contents
1. [use MONAI Core for single-GPU training](./monaicore_singlegpu/)
2. [use MONAI Core for multi-GPU training](./monaicore_multigpu/)
3. [use MONAI Core for Swin UNETR training & evaluation for BRATS21](./monaicore_swinUNETR/)
4. [use MONAI Core for Swin UNETR self-supervised pretraining for BTCV](./pretrain_modify/)
4. [use MONAI Core for Swin UNETR training & evaluation for BTCV](./btcv_modify/)
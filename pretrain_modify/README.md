# **use MONAI Core for Swin UNETR self-supervised pretraining for BTCV on HiperGator**

This directory hosts scripts adapted from [here](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain).  

## **Note**
For testing purpose, only dataset TCIA Covid 19 is used.

## **How to run**
To launch training/inference on a single GPU,
```
sbatch launch_single.sh
```

To launch training on multiple GPUs,
```
sbatch launch_multi.sh
```    
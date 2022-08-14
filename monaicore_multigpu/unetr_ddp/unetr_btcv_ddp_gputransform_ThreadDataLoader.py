# This script is adpated from a MONAI Core tutorial script 
# https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb
# so that it can be run in a multi-gpu distributed-training mode on UF 
# HiperGator's AI partition.  
#
# torch packages used for a distributed training:
# - `torch.distributed.launch` is used to help launch a distributed 
# training. In scripts `\util_multigpu\run_on_node.sh` and 
# `\util_multigpu\run_on_multinode.sh`, `torch.distributed.launch` is 
# called to spawn processes on every node.
# - `torch.distributed.DistributedDataParallel` is used in this script. 
#
# How to run this script:
# - See sample SLURM batch script `\unetr_ddp\launch_gputransform_ThreadDataLoader.sh` 
# (also the called helper scripts `\util_multigpu\run_on_node.sh`, 
# `\util_multigpu\run_on_multinode.sh` & 
# `\util_multigpu\pt_multinode_helper_funcs.sh`), which can launch a 
# PyTorch/MONAI script like this one using `torch.distributed.launch` on 
# a SLURM cluster like HiperGator using Singularity as container runtime. 
#
# Steps to use `torch.distributed.DistributedDataParallel` in this script:
# - Call `init_process_group` to initialize a process group. In this 
#   script, each process runs on one GPU. Here we use `NVIDIA NCCL` as the 
#   backend for optimized multi-GPU training performance and 
#   `init_method="env://"`to initialize a process group by environment 
#   variables.
# - Create a `DistributedSampler` and pass it to DataLoader. Disable 
#   `shuffle` in DataLoader; instead, shuffle data by turning on `shuffle` 
#   in `DistributedSampler` and calling `set_epoch` at the beginning of 
#   each epoch before creating the DataLoader iterator.
# - Wrap the model with `DistributedDataParallel` after moving to expected 
#   GPU.
# - Call `destroy_process_group` after training finishes.
# 
# This script caches imtermediate preprocessed data in GPU memory and then
# performs GPU-based transforms, by using `EnsureTyped``, `ToDeviced` and 
# `CacheDataset`. It also uses `ThreadDataLoader` instead of `DataLoader` for 
# faster I/O. To learn more, see Fast Model Training guide  
# https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md.
#
# 
# References:
# torch.distributed: 
# - https://pytorch.org/tutorials/beginner/dist_overview.html#
# torch.distributed.launch: 
# - https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md 
# - https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
# torch.distributed.DistributedDataParallel:
# - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
#
# Huiwen Ju, hju@nvidia.com
# Aug 2022

import argparse
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from monai.data import DistributedSampler
from datetime import timedelta

from monai.transforms import (
    EnsureTyped,    
    ToDeviced,
)

from monai.data import ThreadDataLoader
# print_config()


def train(args):
    # disable logging for processes except local_rank=0 on every node
    # if args.local_rank != 0:
    #     f = open(os.devnull, "w")
    #     sys.stdout = sys.stderr = f

    # initialize a process group, every GPU runs in a process
    # (all processes connects to the master, obtain information about the other processes,
    # and finally handshake with them)

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=10))
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"],),  #data_type="tensor"
            ToDeviced(keys=["image", "label"], device=f"cuda:{args.local_rank}"),            
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]), # data_type="tensor"
            ToDeviced(keys=["image", "label"], device=f"cuda:{args.local_rank}"),            
        ]
    )

    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)    

    split_JSON = "dataset_1.json"
    datasets = args.data_dir + split_JSON
    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)    
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0, # 8        
    #     pin_memory=False, # True
    #     sampler=train_sampler,
    # )    
    # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
    train_loader = ThreadDataLoader(
        train_ds, 
        num_workers=0, 
        batch_size=1, 
        shuffle=False,
        sampler=train_sampler,
    )

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_sampler = DistributedSampler(dataset=val_ds, even_divisible=False, shuffle=False)
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0, # 2
    #     pin_memory=False, #True,
    #     sampler=val_sampler,
    # )
    # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
    val_loader = ThreadDataLoader(
        val_ds, 
        num_workers=0, 
        batch_size=1, 
        shuffle=False,
        sampler=val_sampler,        
    )

    # create UNETR, DiceLoss and Adam optimizer.
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    # device = torch.device("cuda")  # single GPU
    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(args.num_epoch):
        print(f"[{dist.get_rank()}] " + "-" * 10 + f" epoch {epoch + 1}/{args.num_epoch}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"[{dist.get_rank()}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"[{dist.get_rank()}] " + f"epoch {epoch + 1}, average loss: {epoch_loss:.4f}")
        # validation
        if (epoch + 1) % args.val_interval == 0:
            print(f"[{dist.get_rank()}] " + f"validation at epoch {epoch + 1}/{args.num_epoch}")
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_images, (96, 96, 96), 4, model)
                    val_labels_list = decollate_batch(val_labels)
                    val_labels_convert = [
                        post_label(val_label_tensor) for val_label_tensor in val_labels_list
                    ]
                    val_outputs_list = decollate_batch(val_outputs)
                    val_output_convert = [
                        post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                    ]
                    dice_metric(y_pred=val_output_convert, y=val_labels_convert) # compute metric for current iteration
                metric = dice_metric.aggregate().item() # aggregate the final mean dice result
                dice_metric.reset() # reset the status for next validation round
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    # torch.save(model.state_dict(), "best_metric_model_unetr.pth")
                    print(f"[{dist.get_rank()}] " + "saved new best metric model")
                print(
                    f"[{dist.get_rank()}] " + "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
    print(f"[{dist.get_rank()}] " + f"train completed, epoch losses: {epoch_loss_values}")

    # if dist.get_rank() == 0:
        # saving it in one process is sufficient, because all processes start from the same random parameters
        # and are synchronized
        # torch.save(model.state_dict(), "final_model.pth")
    dist.destroy_process_group()


def main():
    # our CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/", help="directory the dataset is in")
    parser.add_argument("--val_interval", type=int, default=2, help="minimum image intensity")
    parser.add_argument("--num_epoch", type=int, default=100, help="minimum image intensity")

    # parse the command-line argument --local_rank, provided by torch.distributed.launch
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    # for debugging purpose
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    t_start = time.time()
    train(args=args)
    t_end = time.time()
    print(f"time - {args.num_epoch} epoch: {t_end - t_start} sec")


if __name__ == "__main__":
    main()

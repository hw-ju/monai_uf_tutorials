# This script is adpated from a MONAI Core tutorial script 
# https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py
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
# - See sample SLURM batch script `\brats_ddp\launch.sh` (also the called 
# helper scripts `\util_multigpu\run_on_node.sh`, 
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
# - Partition dataset before training, so every rank process will only 
#   handle its own data partition.  
# - Call `destroy_process_group` after training finishes.
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
# ------------------------------------------------------------------------
# Below are some (not all) comments taken from the original script that 
# also applies to running on UF HiperGator. 
#
# This example is a real-world task based on Decathlon challenge Task01: Brain Tumor segmentation.
# So it's more complicated than other distributed training demo examples.
#
# Under default settings, each single GPU needs to use ~12GB memory for network training. In addition, in order to
# cache the whole dataset, ~100GB GPU memory are necessary. Therefore, at least 5 NVIDIA TESLA V100 (32G) are needed.
# If you do not have enough GPU memory, you can try to decrease the input parameter `cache_rate`.
#
# Some codes are taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import numpy as np
import os
import sys
import time
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.apps import DecathlonDataset
from monai.data import ThreadDataLoader, partition_dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet
from monai.optimizers import Novograd
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToDeviced,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class BratsCacheDataset(DecathlonDataset):
    """
    Enhance the DecathlonDataset to support distributed data parallel.

    """
    def __init__(
        self,
        root_dir,
        section,
        transform=LoadImaged(["image", "label"]),
        cache_rate=1.0,
        num_workers=0,
        shuffle=False,
    ) -> None:

        if not os.path.isdir(root_dir):
            raise ValueError("root directory root_dir must be a directory.")
        self.section = section
        self.shuffle = shuffle
        self.val_frac = 0.2
        self.set_random_state(seed=0)
        dataset_dir = os.path.join(root_dir, "Task01_BrainTumour")
        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"cannot find dataset directory: {dataset_dir}, please download it from Decathlon challenge."
            )
        data = self._generate_data_list(dataset_dir)
        super(DecathlonDataset, self).__init__(data, transform, cache_rate=cache_rate, num_workers=num_workers)

    def _generate_data_list(self, dataset_dir):
        data = super()._generate_data_list(dataset_dir)
        # partition dataset based on current rank number, every rank trains with its own data
        # it can avoid duplicated caching content in each rank, but will not do global shuffle before every epoch
        return partition_dataset(
            data=data,
            num_partitions=dist.get_world_size(),
            shuffle=self.shuffle,
            seed=0,
            drop_last=False,
            even_divisible=self.shuffle,
        )[dist.get_rank()]


def main_worker(args):
    # disable logging for processes except 0 on every node
    # if args.local_rank != 0:
    #     f = open(os.devnull, "w")
    #     sys.stdout = sys.stderr = f
    if not os.path.exists(args.dir):
        raise FileNotFoundError(f"missing directory {args.dir}")

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    total_start = time.time()
    train_transforms = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"]),
            ToDeviced(keys=["image", "label"], device=device),
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
    )

    # create a training data loader
    train_ds = BratsCacheDataset(
        root_dir=args.dir,
        transform=train_transforms,
        section="training",
        num_workers=4,
        cache_rate=args.cache_rate,
        shuffle=True,
    )
    # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.batch_size, shuffle=True)

    # validation transforms and dataset
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
            ToDeviced(keys=["image", "label"], device=device),
        ]
    )
    val_ds = BratsCacheDataset(
        root_dir=args.dir,
        transform=val_transforms,
        section="validation",
        num_workers=4,
        cache_rate=args.cache_rate,
        shuffle=False,
    )
    # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=args.batch_size, shuffle=False)

    # create network, loss function and optimizer
    if args.network == "SegResNet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.0,
        ).to(device)
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

    loss_function = DiceFocalLoss(
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
        batch=True,
    )
    optimizer = Novograd(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device])

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    print(f"[{dist.get_rank()}] time elapsed before training: {time.time() - total_start}")
    train_start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # print("-" * 10)
        print(f"[{dist.get_rank()}] epoch {epoch + 1}/{args.epochs}")
        epoch_loss = train(train_loader, model, loss_function, optimizer, lr_scheduler, scaler)
        print(f"[{dist.get_rank()}] epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            metric, metric_tc, metric_wt, metric_et = evaluate(
                model, val_loader, dice_metric, dice_metric_batch, post_trans
            )

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), "best_metric_model.pth")
            print(
                f"[{dist.get_rank()}] current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"[{dist.get_rank()}] tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\n[{dist.get_rank()}] best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )

        print(f"[{dist.get_rank()}] time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    print(
        f"[{dist.get_rank()}] train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch},"
        f"[{dist.get_rank()}] total train time: {(time.time() - train_start):.4f}"
    )
    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler):
    model.train()
    step = 0
    epoch_len = len(train_loader)
    epoch_loss = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(batch_data["image"])
            loss = criterion(outputs, batch_data["label"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(f"[{dist.get_rank()}] {step}/{epoch_len}, train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}")
        step_start = time.time()
    lr_scheduler.step()
    epoch_loss /= step

    return epoch_loss


def evaluate(model, val_loader, dice_metric, dice_metric_batch, post_trans):
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    inputs=val_data["image"], roi_size=(240, 240, 160), sw_batch_size=4, predictor=model, overlap=0.6
                )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_data["label"])
            dice_metric_batch(y_pred=val_outputs, y=val_data["label"])

        metric = dice_metric.aggregate().item()
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()
        dice_metric.reset()
        dice_metric_batch.reset()

    return metric, metric_tc, metric_wt, metric_et


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory of Brain Tumor dataset")
    # must parse the command-line argument: ``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by DDP
    parser.add_argument("--local_rank", type=int, help="node rank for distributed training")
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="mini-batch size of every GPU")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="larger cache rate relies on enough GPU memory.")
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--network", type=str, default="SegResNet", choices=["UNet", "SegResNet"])
    args = parser.parse_args()

    if args.seed is not None:
        set_determinism(seed=args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main_worker(args=args)
    

if __name__ == "__main__":
    main()


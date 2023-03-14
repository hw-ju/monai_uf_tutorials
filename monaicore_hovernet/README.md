Work in progress...

To launch a single-node multi-gpu training,
```
sbatch train_torchrun.sh
```
See sample output `train_torchrun.sh.59565834.out`

**Note:**
1. For multi-gpu pytorch ddp training, here we use `torchrun` instead of `torch.distributed.launch` to launch the training. Thus, in `train_torchrun.sh`, we use `/monai_uf_tutorials/monaicore_hovernet/util_multigpu` instead of `/monai_uf_tutorials/monaicore_multigpu/util_multigpu`.
2. The `training.py` is the same one on MONAI Core's tutorial repo.
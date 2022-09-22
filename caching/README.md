# **use Datasets accelerated by caching in MONAI Core on HiperGator**

This directory hosts a [tutorial script](./dataset_type_performance_modified.ipynb) adapted from [here](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/dataset_type_performance.ipynb).  

## **Note**
Notes about using the caching-accelerated Datasets on HiperGator (more details see [MONCAI Core doc](https://docs.monai.io/en/stable/)):

### 1. PersistentDataset
- Great for larger-than-memory intermediate (i.e., non-randomly transformed) data.
- Intermediate (i.e., non-randomly transformed) data are computed when **first used**, and stored in the cache_dir for rapid retrieval on subsequent uses. Thus, the cost of first time processing of data is distributed across each first use, which is different from CacheDatset(see below).
- PersistentDataset has a similar memory footprint to the simple Dataset, with performance characteristics close to the CacheDataset at the expense of disk storage.

### 2. CacheDataset
- CacheDataset executes non-random transforms and prepares **all cache content** in the main process before the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process during training. It may take a long time to prepare cache content according to the size of expected cache data. So to **debug or verify** the program before real training, users can set cache_rate=0.0 or cache_num=0 to temporarily skip caching. 
- Allocate **enough** CPU memory in #SBATCH settings, e.g., by using #SBATCH --mem-per-cpu=large_value.

### 3. SmartCacheDataset
- In each epoch, **only the items in the cache are used for training**, and simultaneously, another thread is preparing replacement items by applying the transforms to items that are not in the cache. Once one epoch is completed, SmartCache replaces the same number of items with replacement items.
- The usage of SmartCacheDataset contains 4 steps (see [tutorial script](./dataset_type_performance_modified.ipynb)), thus you need to modify your code using other Dataset types in order to use SmartCacheDataset. 
- Allocate **enough** CPU memory in #SBATCH settings, e.g., by using #SBATCH --mem-per-cpu=large_value. 






## **How to run**
The tutorial script is a jupyter notebook. To learn how to run a jupyter notebook within a MONAI Core container on HiperGator, refer [here](../monaicore_singlegpu/README.md).
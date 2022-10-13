from re import S
from scipy.spatial import KDTree
from src.tools import load_file, save_file, downsample
import numpy as np
import torch
import cupy
import pandas

mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()
def clean_up():
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
##---------------------------------------------------------------------------------------------------------

def SOR(df,k,nsd):
print("Returns indices of inliers")
arr = df[['x', 'y', 'z']].values
available_mem = (torch.cuda.get_device_properties(0).total_memory/1024.0**3)/1.10
required_mem = np.ceil(arr[0].nbytes*k*arr.shape[0]/1024.0**3)*2
block_size = available_mem/required_mem
block_size = int(arr.shape[0]*block_size)
blocks = np.array_split(np.arange(arr.shape[0]), np.ceil(arr.shape[0] / block_size))
#1. compute neighbours using fast kd tree
nbrs = KDTree(arr, compact_nodes=True)
dist, indices = nbrs.query(arr, k=k, workers=-1)
results_blocks = []
    for b, _ in enumerate(blocks):
        #2. Load data into memory
        if len(blocks)==1:
            array_gpu = cupy.asarray(dist)
            indices_gpu = cupy.asarray(indices).flatten()
        else:
            array_gpu = cupy.asarray(dist[blocks[b]]) 
            indices_gpu = cupy.asarray(indices[blocks[b]]).flatten()
#Create rule 
avg = cupy.average(array_gpu, axis=1)
sd = cupy.std(avg)
thres = sd*nsd+cupy.average(avg)


tmp = cupy.expand_dims(cupy.add(cupy.multiply(nsd,sd),avg),1)
thres, _ = cupy.broadcast_arrays(tmp, array_gpu)
del tmp, sd, avg
clean_up()
#apply rule 
bl = (array_gpu < thres).flatten()
idx = cupy.unique(indices_gpu[bl])
results_blocks.append(cupy.asnumpy(idx))
#clear up
del thres,bl,idx,array_gpu,indices_gpu
clean_up()
            #
    return results_blocks

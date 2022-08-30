from re import S
from scipy.spatial import KDTree
from src.tools import load_file, save_file, downsample, denoise
import numpy as np
import torch
import pandas 

# import cupy
# mempool = cupy.get_default_memory_pool()
# pinned_mempool = cupy.get_default_pinned_memory_pool()

df = load_file('/home/harryowen/Desktop/wood.ply')
df = df.iloc[denoise(df, 100, 1.0)].reset_index()
arr = df[['x','y','z']].values

#Create KD tree structure 
nbrs = KDTree(arr, compact_nodes=True)
dist, indices = nbrs.query(arr, k=16, workers=-1)
#
min_dist = 0.02
min_cluster_size = 10
max_cluster_size = np.inf

processed = np.zeros(len(indices), dtype=bool)
cluster_id = 0
clusters = []

for i, nn in enumerate(indices):
    #Skip this point if it has been assigned already to a cluster
    if processed[i] == True:
        continue
    sq_idx = 0
    processed[i] = True
    seed_queue  = []#np.zeros(len(indices), dtype=int)
    seed_queue.append(i)
    while sq_idx < len(seed_queue):
        #print(sq_idx, "   ", len(seed_queue))
        within_dist = dist[seed_queue[sq_idx]] <= min_dist
        nn_idx = indices[seed_queue[sq_idx]][within_dist]
        #remove any neighbours within distance but that have already been processed
        nn_idx = nn_idx[~processed[nn_idx]]
        #add unprocessed but within distance neighbours to Queue
        seed_queue = np.append(seed_queue, nn_idx)
        #ensure clusters don't get bigger than max cluster size
        if len(seed_queue) < max_cluster_size: 
            processed[nn_idx] = True
            sq_idx += 1
        else:
            diff = len(seed_queue)-max_cluster_size
            seed_queue = seed_queue[:-diff]
            processed[nn_idx[:-diff]] = True
            sq_idx = np.inf
    #
    if (len(seed_queue) > min_cluster_size):
        clusters.append(np.column_stack((arr[seed_queue],np.full((len(seed_queue)),cluster_id))))
        cluster_id += 1
    #print(cluster_id)

out = pandas.DataFrame(np.concatenate(clusters),columns = ['x','y','z','id'])
save_file('/home/harryowen/Desktop/clusters.ply',out,additional_fields=['id'])







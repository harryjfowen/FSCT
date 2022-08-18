from pykdtree.kdtree import KDTree
from src.tools import load_file, save_file, downsample, denoise
import numpy as np
import torch
import cupy
import pandas 

mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

#df = load_file('/home/harryowen/Desktop/tropical-large.ply')

##---------------------------------------------------------------------------------------------------------

def calc_features_gpu(e):
    # See Wan et al., 2021 [https://doi.org/10.1111/2041-210X.13715]
    l = (e[:, 0] - e[:, 1]) / e[:, 0]
    p = (e[:, 1] - e[:, 2]) / e[:, 0]
    s = e[:, 2] / e[:, 0]
    lsod = l + ((1-l) * (l - cupy.maximum(p,s)))
    s = 1-s
    sv = e[:, 2] / cupy.sum(e, axis=1)
    sv = 1-(sv / cupy.amax(sv))
    t = (e[:, 0] - e[:, 2]) / e[:, 0]
    return cupy.vstack(([l, p, s, sv, t,lsod]))

def calc_linearity_gpu(e):
    # See Wan et al., 2021 [https://doi.org/10.1111/2041-210X.13715]
    l = (e[:, 0] - e[:, 1]) / e[:, 0]
    p = (e[:, 1] - e[:, 2]) / e[:, 0]
    s = e[:, 2] / e[:, 0]
    lsod = l + ((1-l) * (l - cupy.maximum(p,s)))
    return cupy.vstack(([l, lsod]))

##---------------------------------------------------------------------------------------------------------

def add_features_gpu(df):
    knn = [20,100,200]
    'Calculating Geometric Features using CUDA.' if torch.cuda.is_available() else 'No GPU found'
    arr = df[['x', 'y', 'z']].values
    results_knn = np.zeros([arr.shape[0], 6], dtype=float)
    it = 0
    for i, k in enumerate(knn):
        available_mem = (torch.cuda.get_device_properties(0).total_memory/1024.0**3)/1.05
        required_mem = np.ceil(arr[0].nbytes*k*arr.shape[0]/1024.0**3)*2
        block_size = available_mem/required_mem
        block_size = int(arr.shape[0]*block_size)
        blocks = np.array_split(np.arange(arr.shape[0]), np.ceil(arr.shape[0] / block_size))
        #1. compute neighbours using fast kd tree
        nbrs = KDTree(arr)
        dist, indices = nbrs.query(arr, k=k, distance_upper_bound = 1000)
        indices=indices[~np.isinf(dist).any(1)].astype(int)
        results_blocks = np.zeros([arr.shape[0], 2], dtype=float)
        for b, _ in enumerate(blocks):
            #2. Load data into memory
            if len(blocks)==1:
                array_gpu = cupy.asarray(arr[indices])
            else: 
                array_gpu = cupy.asarray(arr[indices[blocks[b]]])
            #calc other features using svd
            diffs = array_gpu - cupy.mean(array_gpu, axis=1, keepdims = True)
            del array_gpu; mempool.free_all_blocks() 
            cov = cupy.einsum('ijk,ijl->ikl', diffs, diffs)/k
            del diffs; mempool.free_all_blocks() 
            evals = cupy.linalg.svd(cov, compute_uv=False)
            del cov; mempool.free_all_blocks() 
            features = calc_linearity_gpu((evals.T / cupy.sum(evals, axis=1)).T).T
            del evals; mempool.free_all_blocks() 
            features[cupy.isnan(features)] = 0
            # Move result from gpu to cpu
            results_blocks[blocks[b]] = cupy.asnumpy(features)
            del features; mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        ####
        ### 
        results_knn[:, it:it+2] = results_blocks
        it+=2
        print('Finished calculating features using ', k, ' neighbours...')
    #
    cols = ['l20','sod20','l100','sod100','l200','sod200']
    features = pandas.DataFrame(results_knn, columns = cols)
    del results_knn, results_blocks
    #
    return features

#y = add_features_gpu(df)
#save_file('/home/harryowen/Desktop/xyz-features.ply',pandas.concat([df,y],axis=1), additional_fields=cols)

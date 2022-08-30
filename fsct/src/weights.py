from re import S
from scipy.spatial import KDTree
from src.tools import load_file, save_file, downsample, denoise
import numpy as np
import torch
import cupy
import pandas 

mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

#df = load_file('/home/harryowen/Desktop/tile2.ply')
#df = df.iloc[denoise(df, 100, 1.0)].reser_index()
##---------------------------------------------------------------------------------------------------------

def calc_linearity_gpu(e):
    # See Wan et al., 2021 [https://doi.org/10.1111/2041-210X.13715]
    l = (e[:, 0] - e[:, 1]) / e[:, 0]
    return l

def calc_sphericity_gpu(e):
    # See Wan et al., 2021 [https://doi.org/10.1111/2041-210X.13715]
    s = (e[:, 0] - e[:, 2]) / e[:, 0]
    return s

def calc_max_feature(e):
    # See Wan et al., 2021 [https://doi.org/10.1111/2041-210X.13715]
    l = (e[:, 0] - e[:, 1]) / e[:, 0]
    a = (e[:, 0] - e[:, 2]) / e[:, 0]
    return cupy.amax(cupy.vstack([l,a]),axis=0)

##---------------------------------------------------------------------------------------------------------

def add_features_gpu(df):
    knn = [50,100,200,400]
    'Calculating Geometric Features using CUDA.' if torch.cuda.is_available() else 'No GPU found'
    arr = df[['x', 'y', 'z']].values
    results_knn = np.zeros([arr.shape[0], len(knn)], dtype=float)
    it = 0
    for i, k in enumerate(knn):
        available_mem = (torch.cuda.get_device_properties(0).total_memory/1024.0**3)/1.10
        required_mem = np.ceil(arr[0].nbytes*k*arr.shape[0]/1024.0**3)*2
        block_size = available_mem/required_mem
        block_size = int(arr.shape[0]*block_size)
        blocks = np.array_split(np.arange(arr.shape[0]), np.ceil(arr.shape[0] / block_size))
        #1. compute neighbours using fast kd tree
        nbrs = KDTree(arr, compact_nodes=True)
        dist, indices = nbrs.query(arr, k=k, workers=-1)
        results_blocks = np.zeros([arr.shape[0]], dtype=float)
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
            features = calc_max_feature((evals.T / cupy.sum(evals, axis=1)).T).T
            del evals; mempool.free_all_blocks() 
            features[cupy.isnan(features)] = 0
            # Move result from gpu to cpu
            results_blocks[blocks[b]] = cupy.asnumpy(features)
            del features; mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        ####
        ### 
        results_knn[:, it] = results_blocks
        it+=1
        print('Finished calculating features using ', k, ' neighbours...')
    #
    cols = [str(s) + 'NN' for s in knn]
    features = pandas.DataFrame(results_knn, columns = cols)
    del results_knn, results_blocks
    #
    return features

#y = add_features_gpu(df)
#cols = ['f20','f100','f200']
#cols = ['fmax']
#save_file('/home/harryowen/Desktop/xyz-features.ply',pandas.concat([df,x],axis=1), additional_fields=cols + ['refl'])

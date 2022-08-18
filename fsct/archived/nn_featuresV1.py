from pykdtree.kdtree import KDTree
from src.tools import load_file, save_file, downsample
import numpy as np
import torch
import threading
from tqdm import tqdm
import cupy
mempool = cupy.get_default_memory_pool()

df = load_file('/home/harryowen/Desktop/tile.ply')
import pandas 
##---------------------------------------------------------------------------------------------------------
#       CuPy GPU Version
##---------------------------------------------------------------------------------------------------------
def calc_features_gpu(e):
    # Calculating salient features.
    l = (e[:, 0] - e[:, 1]) / e[:, 0]
    p = (e[:, 1] - e[:, 2]) / e[:, 0]
    s = e[:, 2] / e[:, 0]
    lsod = l + ((1-l) * (l - cupy.maximum(p,s)))
    s = 1-s
    sv = e[:, 2] / cupy.sum(e, axis=1)
    sv = 1-(sv / cupy.amax(sv))
    t = (e[:, 0] - e[:, 2]) / e[:, 0]
    return cupy.vstack(([l, p, s, sv, t,lsod]))

def calc_verticality_gpu(xyz):
    ids = np.arange(len(xyz))
    V = cupy.amax(xyz[:, :, -1],axis=1)-cupy.amin(xyz[:, :, -1],axis=1)
    V = V / cupy.percentile(V,95)
    V=cupy.where(V > 1, 1, V)
    return V

def add_features_gpu(df,knn):
#knn=20 good for our data
'Calculating Geometric Features using CUDA.' if torch.cuda.is_available() else 'No GPU found'
arr = df[['x', 'y', 'z']].values
#1. compute neighbours using fast kd tree
nbrs = KDTree(arr)
dist, indices = nbrs.query(arr, k=knn, distance_upper_bound = 1000)
#remove any values which are infinite (beyond distance threshold)
# currently suboptimcal as deleting entire neighbourhoods rather than erroneous nbrs within!
#indices=indices[~np.isinf(dist).any(1)].astype(int)
# if(array_gpu.nbytes >= torch.cuda.get_device_properties(0).total_memory):
#     print('File too big to fit into gpu memory')    
array_gpu = cupy.asarray(arr[indices])
arr_shape = array_gpu.shape[1]
print('Allocating ', round(array_gpu.nbytes/1024.0**3,2), ' GB to the gpu')
# Calculating the covariance of the stack of arrays
# [see https://stackoverflow.com/questions/35756952/quickly-compute-eigenvectors-for-each-element-of-an-array-in-python]
#calc verticality
v = calc_verticality_gpu(array_gpu)
#calc other features using svd
diffs = array_gpu - cupy.mean(array_gpu, axis=1, keepdims = True)
diffs2 = cupy.copy(diffs)
del array_gpu; mempool.free_all_blocks() 
cov = cupy.einsum('ijk,ijl->ikl', diffs, diffs2)/arr_shape
del diffs, diffs2; mempool.free_all_blocks() 
# Calculating the eigenvalues using Singular Value Decomposition (svd).
evals = cupy.linalg.svd(cov, compute_uv=False)
del cov; mempool.free_all_blocks() 
#3. calc ratios of the eigen vectors [sum to 1]
#features = cupy.column_stack((calc_features_gpu((evals.T / cupy.sum(evals, axis=1)).T),curvature))
features = calc_features_gpu((evals.T / cupy.sum(evals, axis=1)).T).T
features = cupy.column_stack((features,v))
features[cupy.isnan(features)] = 0
#del evals, v; mempool.free_all_blocks() 
#create pandas data frame 
output = pandas.DataFrame(cupy.asnumpy(features), columns = ['l','p','s','sv','t','lsod','v'])
output['v'] = output['v']*output['s']
del features; mempool.free_all_blocks() 
##alternative output
#weights = np.amax(features, axis=1)
#output = pandas.DataFrame(cupy.asnumpy(weights), columns = ['weights'])
#save_file('/home/harryowen/Desktop/xyz-features.ply',pandas.concat([pandas.DataFrame(arr, columns=['x','y','z']),output],axis=1), additional_fields=['weights'])

#write out for visual inspection
save_file('/home/harryowen/Desktop/xyz-features.ply',pandas.concat([pandas.DataFrame(arr, columns=['x','y','z']),output],axis=1), additional_fields=['l','p','s','sv','t','lsod','v'])
    return output


from sklearn.mixture import GaussianMixture as GMM
def classify(variables, n_classes):
    gmm = GMM(n_components=n_classes)
    gmm.fit(variables)
    return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)

pred = classify(output, 2)
#df = pandas.concat([output, pandas.DataFrame(pred[2], columns=["prob1", "prob2", "prob3"])], axis=1)
df = pandas.concat([output, pandas.DataFrame(pred[0], columns=["label"])], axis=1)
save_file('/home/harryowen/Desktop/xyz-features.ply',pandas.concat([pandas.DataFrame(arr, columns=['x','y','z']),df],axis=1), additional_fields=['l','p','s','sv','t','v','label'])


##---------------------------------------------------------------------------------------------------------
#       Numpy CPU Version
##---------------------------------------------------------------------------------------------------------


def calc_features_cpu(e):
    # Calculating salient features.
    e1 = e[:, 2]
    e2 = e[:, 0] - e[:, 1]
    e3 = e[:, 1] - e[:, 2]
    # Calculating tensor features.
    t1 = (e[:, 1] - e[:, 2]) / e[:, 0]
    t2 = ((e[:, 0] * np.log(e[:, 0])) + (e[:, 1] * np.log(e[:, 1])) +
          (e[:, 2] * np.log(e[:, 2])))
    t3 = (e[:, 0] - e[:, 1]) / e[:, 0]
    return np.vstack(([e1, e2, e3, t1, t2, t3])).T

def calc_ratios(arr,nbrs_idx,I):
    diffs = arr[nbrs_idx[I]] - arr[nbrs_idx[I]].mean(1, keepdims = True)
    cov = np.einsum('ijk,ijl->ikl', diffs, diffs)/arr[nbrs_idx[I]].shape[1]
    tmp = np.linalg.svd(cov, compute_uv=False)
    ratio = (tmp.T / np.sum(tmp, axis=1)).T
    features[I] = calc_features_cpu(ratio)



def add_features_numpy(arr,knn):
    
    block_size = 100000
    if block_size > arr.shape[0]:
        block_size = arr.shape[0]

    #compute neighbours using fast kd tree
    nbrs = KDTree(arr)
    dist, nbrs_idx = nbrs.query(arr, k=knn, distance_upper_bound = 0.25)

    # Creating block of ids.
    ids = np.arange(arr.shape[0])
    ids = np.array_split(ids, int(arr.shape[0] / block_size))

    # Making sure nbr_idx has the correct data type.
    nbrs_idx = nbrs_idx.astype(int)

    features = np.zeros([arr.shape[0], 6], dtype=float)
    threads = []
    for i in ids:
        threads.append(threading.Thread(target=calc_ratios, args=(arr, nbrs_idx, i)))

    for x in tqdm(threads, desc='Calculating features', disable=False):
        x.start()

    for x in threads:
        x.join()

    features[np.isnan(features)] = 0

    #out = pandas.concat([df,pandas.DataFrame(features, columns = ['e1','e2','e3','t1','t2','t3'])],axis=1)
    #save_file('/home/harryowen/Desktop/features.ply', out, additional_fields = ['e1','e2','e3','t1','t2','t3'])

    return pandas.DataFrame(features, columns = ['e1','e2','e3','t1','t2','t3'])


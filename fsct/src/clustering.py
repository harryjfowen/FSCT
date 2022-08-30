from re import S
from scipy.spatial import KDTree
from src.tools import load_file, save_file, downsample, denoise
import numpy as np
import torch
import pandas 
from cuml.cluster import HDBSCAN

res = 0.02

df = load_file('/home/harryowen/Desktop/wood.ply')
df = downsample(df, res).reset_index(drop=True)

labels = HDBSCAN(min_samples=10, min_cluster_size = 3, max_cluster_size = 100000000, cluster_selection_method = 'leaf',cluster_selection_epsilon=res*2).fit_predict(df)

out = pandas.concat([df, pandas.DataFrame(labels, columns=["label"])], axis=1)
save_file('/home/harryowen/Desktop/tropical-clusters.ply',out, additional_fields=['label'])

# xyzl = np.column_stack((df.values,labels))
# indices = np.argsort(labels)
# stack = np.array_split(xyzl[indices, :-1], np.where(np.diff(xyzl[indices][:,3])!=0)[0]+1)
# stack = sorted(stack, key=len, reverse=True)

arr = df.values
def svd_evals(arr):
    centroid = np.average(arr, axis=0)
    _, evals, evecs = np.linalg.svd(arr - centroid, full_matrices=False)
    return evals

def cluster_filter(arr, labels, eval_threshold):
    final_evals = np.zeros([labels.shape[0], 3])
    for L in np.unique(labels):
        ids = np.where(labels == L)[0]
        if (L != -1) and len(ids) >= 20:
            if len(ids) <= 10000:
                e = svd_evals(arr[ids])
                final_evals[ids] = e
            else:
                final_evals[ids] = np.array([1,0,0])
    ratio = np.asarray([i / np.sum(i) for i in final_evals])
    return ratio[:, 0] >= eval_threshold

x = cluster_filter(arr,labels,0.66)
out=df[x]
save_file('/home/harryowen/Desktop/tropical-filtered.ply',out, additional_fields=[])





# import threading
# from tqdm import tqdm

# features = np.zeros([arr.shape[0], 6], dtype=float)
# threads = []
#     for i in labels:
#         threads.append(threading.Thread(target=cluster_filter, args=(arr, i)))

#     for x in tqdm(threads, desc='Filtering clusters', disable=False):
#         x.start()

#     for x in threads:
#         x.join()
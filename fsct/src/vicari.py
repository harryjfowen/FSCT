from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
from pandas import DataFrame
import pandas as pd
import numpy as np

def get_diff(arr1, arr2):
    assert arr1.shape[1] == arr2.shape[1]
    arr3 = np.vstack((arr1, arr2))
    df = pd.DataFrame(arr3)
    diff = df.drop_duplicates(keep=False)
    return np.asarray(diff)

def remove_duplicates(arr, return_ids=False):
    df = pd.DataFrame({'x': arr[:, 0],
                       'y': arr[:, 1], 'z': arr[:, 2]})
    if return_ids:
        return np.where(df.duplicated((['x', 'y', 'z'])))[0]
    else:
        unique = df.drop_duplicates(['x', 'y', 'z'])
        return np.asarray(unique).astype(float)


def curvature(arr):
    print("Finding neighbours...")
    nbrs_idx = set_nbrs_knn(arr, arr, 9, return_dist=False, block_size=100000)
    evals = np.zeros([arr.shape[0], 3], dtype=float)
    print("Calculating curvature...")
    for i, nids in enumerate(nbrs_idx.astype(int)):
        if arr[nids].shape[0] > 3:
            evals[i] = svd_evals(arr[nids])
    c = evals[:, 2] / np.sum(evals, axis=1)
    return c

def knn_features(arr, nbr_idx, block_size=200000):

    if block_size > arr.shape[0]:
        block_size = arr.shape[0]

    ids = np.arange(arr.shape[0])
    ids = np.array_split(ids, int(arr.shape[0] / block_size))
    nbr_idx = nbr_idx.astype(int)
    s = np.zeros([arr.shape[0], 3], dtype=float)

    for i in ids:
        s[i] = knn_evals(arr[nbr_idx[i]])

    ratio = (s.T / np.sum(s, axis=1)).T
    features = calc_features(ratio)
    features[np.isnan(features)] = 0

    return features


def knn_evals(arr_stack):
    cov = vectorized_app(arr_stack)
    evals = np.linalg.svd(cov, compute_uv=False)
    return evals


def calc_features(e):

    # Calculating salient features.
    e1 = e[:, 2]
    e2 = e[:, 0] - e[:, 1]
    e3 = e[:, 1] - e[:, 2]

    # Calculating tensor features.
    t1 = (e[:, 1] - e[:, 2]) / e[:, 0]
    t2 = ((e[:, 0] * np.log(e[:, 0])) + (e[:, 1] * np.log(e[:, 1])) +
          (e[:, 2] * np.log(e[:, 2])))
    t3 = (e[:, 0] - e[:, 1]) / e[:, 0]

    # Calculating Curvature
    #c = e[:, 2] / np.sum(e, axis=1)

    return np.vstack(([e1, e2, e3, t1, t2, t3])).T


def vectorized_app(arr_stack):
    diffs = arr_stack - arr_stack.mean(1, keepdims=True)
    return np.einsum('ijk,ijl->ikl', diffs, diffs)/arr_stack.shape[1]


def svd_evals(arr):
    centroid = np.average(arr, axis=0)
    _, evals, evecs = np.linalg.svd(arr - centroid, full_matrices=False)
    return evals

def set_nbrs_knn(arr, pts, knn, return_dist=True, block_size=100000):
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)
    if block_size > pts.shape[0]:
        block_size = pts.shape[0]
    ids = np.arange(pts.shape[0])
    ids = np.array_split(ids, int(pts.shape[0] / block_size))
    if return_dist is True:
        distance = np.zeros([pts.shape[0], knn])
    indices = np.zeros([pts.shape[0], knn])
    if return_dist is True: 
        for i in ids:
            nbrs_dist, nbrs_ids = nbrs.kneighbors(pts[i])
            distance[i] = nbrs_dist
            indices[i] = nbrs_ids
        return distance, indices
    elif return_dist is False:
        for i in ids:
            nbrs_ids = nbrs.kneighbors(pts[i], return_distance=False)
            indices[i] = nbrs_ids
        return indices

def classify(variables, n_classes):

    gmm = GMM(n_components=n_classes)
    gmm.fit(variables)

    return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)

def class_select_abs(classes, cm, nbrs_idx, feature=1, threshold=0.5):
    if np.max(np.sum(cm, axis=1)) >= threshold:
        class_id = np.argmax(cm[:, feature])
        mask = classes == class_id
    else:
        mask = []
    return mask

def wlseparate_abs(arr, knn, n_classes=3):

    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    gd_1 = knn_features(arr, idx_1)
    classes_1, cm_1, proba_1 = classify(gd_1, n_classes)

    #here problem where reflectance values messsing this up [returning an empty list]
    mask_1 = class_select_abs(classes_1, cm_1, idx_1)
    arr_ids = np.arange(0, arr.shape[0], 1, dtype=int)

    class_indices = {}
    class_probability = {}
    try:
        class_indices['wood'] = arr_ids[mask_1]
        class_probability['wood'] = np.max(proba_1, axis=1)[mask_1]
        
    except:
        class_indices['wood'] = []
        class_probability['wood'] = []
    try:
        class_indices['leaf'] = arr_ids[~mask_1]
        class_probability['leaf'] = np.max(proba_1, axis=1)[~mask_1]
    except:
        class_indices['leaf'] = []
        class_probability['leaf'] = []

    return class_indices, class_probability
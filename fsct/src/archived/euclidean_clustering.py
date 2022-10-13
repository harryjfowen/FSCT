from re import S
from scipy.spatial import KDTree
import numpy as np
import pandas 

def euclidean_clustering(df,knn,min_cluster_size,max_cluster_size):
    df = df.sort_values(['x', 'y', 'z'])#.reset_index(drop=True)
    arr = df[['x','y','z']].values
    #Create KD tree structure 
    nbrs = KDTree(arr, compact_nodes=True)
    dist, indices = nbrs.query(arr, k=knn, workers=-1)
    #
    min_dist = np.percentile(np.mean(dist,axis=1), 10)#0.02
    print('clustering using ', min_dist, ' as minimum distance...')
    #max_cluster_size = np.inf
    processed = np.zeros(len(indices), dtype=bool)
    cluster_id = 1
    clusters = []
    for i, nn in enumerate(indices):
        #Skip this point if it has been assigned already to a cluster
        if processed[i] == True:
            continue
        processed[i] = True
        sq_idx = 0
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
                #diff = len(seed_queue)-max_cluster_size
                #seed_queue = seed_queue[:-diff]
                #processed[nn_idx[:-diff]] = True
                processed[nn_idx] = True
                sq_idx = np.inf
            #
        if (len(seed_queue) >= min_cluster_size):
            clusters.append(np.column_stack((arr[seed_queue],np.full((len(seed_queue)),cluster_id))))
            cluster_id += 1
    #out = pandas.DataFrame(np.concatenate(clusters),columns = ['x','y','z','id'])
    #save_file('/home/harryowen/Desktop/clusters.ply',out,additional_fields=['id'])
    return pandas.DataFrame(np.concatenate(clusters),columns = ['x','y','z','id'])


def svd_evals(arr):
    centroid = np.average(arr, axis=0)
    _, evals, evecs = np.linalg.svd(arr - centroid, full_matrices=False)
    return evals

def cluster_filter(arr, labels, eval_threshold, direction):
    final_evals = np.zeros([labels.shape[0], 3])
    for L in np.unique(labels):
        ids = np.where(labels == L)[0]
        if direction=='positive':
                final_evals[ids] = np.array([1,0,0])
        if direction=='negative':
                final_evals[ids] = np.array([0,0,0])
        if len(ids) >= 3:
            if len(ids) <= 1000:
                e = svd_evals(arr[ids])
                final_evals[ids] = e
    if direction == 'positive':
        ratio = np.asarray([i / np.sum(i) for i in final_evals])
        return ratio[:, 0] >= eval_threshold
    if direction == 'negative':
        ratio = np.asarray([i / np.sum(i) for i in final_evals])
        ratio = np.nan_to_num(ratio)
        return ratio[:, 0] < eval_threshold

    
## upscale
def upscale_labels(labelled_df, df, knn, max_dist):
    nbrs = scipy.spatial.KDTree(labelled_df[['x','y','z']].values,compact_nodes=True)
    dist, indices = nbrs.query(df[['x','y','z']].values, k=knn, workers=-1)
    dist_bool = dist < max_dist
    new_label = np.zeros(df.shape[0],dtype=int)
    labels = labelled_df[['label']].values
    label_ = labels[indices]
    row_del = []
    for i in range(len(indices)):
        unique, count = np.unique(label_[i, dist_bool[i]], return_counts=True)
        if len(count) == 0:
            #new_label[i] = 1
            row_del.append(i)
            continue
        new_label[i] = unique[np.argmax(count)]
        #new_label[i] = np.min(label_[i, dist_bool[i]])
    df.loc[:, 'label'] = new_label
    return df.drop(row_del)


def upscale(df1, df2, knn, max_dist):
    nbrs = scipy.spatial.KDTree(df1[['x','y','z']].values)
    dist, indices = nbrs.query(df2[['x','y','z']].values, k=knn, workers=-1)
    dist_bool = dist < max_dist
    row_del = []
    idx = []
    for i in range(len(indices)):
        nbr_idx = indices[i, dist_bool[i]]
        if len(nbr_idx) == 0:
            idx.append(df1.index[i])
            continue
        idx.append(nbr_idx)
    return np.unique(np.hstack(idx))


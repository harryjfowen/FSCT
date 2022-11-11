import open3d as o3d
import numpy as np
import pandas as pd
from src.tools import save_file, load_file
import math
import os
from tqdm import tqdm
from scipy.spatial import cKDTree
import CSF #NOTE add automated installation to requirements

idx_list = []
def traverse_octree(node, node_info):

    max_vx_size = 1
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0

            for child in node.children:
                if child is not None:
                    n += 1

            early_stop = len(node.indices) < 1000

            if len(node.indices) > 1000 and node_info.size <= max_vx_size:
                idx_list.append(np.asarray(node.indices))

    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        #extract indices for all final leaf nodes
        if len(node.indices) >= 1000:
            idx_list.append(np.asarray(node.indices))

    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


def verticality(arr, knn, r):

    ids = np.arange(arr.shape[0])
    nbrs = cKDTree(arr)
    _, nbrs_idx = nbrs.query(arr, k=knn, distance_upper_bound = r)
    V = np.zeros([arr.shape[0], 1], dtype=float)

    for i in ids:
        vec = arr[nbrs_idx[i]][:, 2]
        vec = vec[~np.isnan(vec)]
        V[i] = (np.max(vec)-np.min(vec))

    V = V/np.percentile(V,99)
    V[np.isnan(V)] = 0

    return V


def classify_ground(args):
    
    print("Classifying ground points...")
    csf = CSF.CSF()

    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.33
    csf.params.class_threshold = 0.33

    xyz=args.pc[['x', 'y', 'z']].to_numpy().astype('double')

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    tmpIDX = args.pc.loc[ground].index.to_numpy()

    # Filter remaining stumps    
    V = verticality(args.pc.loc[tmpIDX][['x','y','z']].to_numpy(), 50, 1000)
    groundIDX = tmpIDX[np.where(V.flatten() < 0.75)]

    bool = args.pc.index.isin(groundIDX)
    del tmpIDX, csf, V, bool

    return args.pc.index[bool]#improve this, seems overkill to use isin and not index directly


def pcd2voxels(args):

    args.grdidx = classify_ground(args.pc)
    canopy = args.point_cloud.drop(args.grdidx)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(args.point_cloud[['x', 'y', 'z']].to_numpy().astype('double'))

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=1)
    args.point_cloud = args.point_cloud.loc[np.array(ind)].reset_index(drop=True)
    args.point_cloud.rename(columns = {'scalar_refl':'refl'}, inplace = True)

    #SORT OUT LABEL COLUMNS
    args.point_cloud.set_axis(labels=['label' if c.endswith('label') else c for c in args.point_cloud], axis=1, inplace=True)

    if 'label' in args.point_cloud.columns:
        args.point_cloud = args.point_cloud.astype({"label": int})

    #Rough way to determine the depth of octree to alleviate senstivity to canopy dimensions with fixed depth param 
    dim_lengths = args.point_cloud.max()-args.point_cloud.min()
    maxSize = pd.Series.max(dim_lengths[:3])
    D = math.ceil(math.log(maxSize,2))+1

    octree = o3d.geometry.Octree(max_depth=D)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    octree.traverse(traverse_octree)

    meta = []
    for i, idx in enumerate(tqdm(idx_list)):

        if len(idx) > 20000:
            vx = args.point_cloud.loc[idx].sample(n=20000)
        else:
            vx = args.point_cloud.loc[idx]
        np.save(os.path.join("/home/harryjfowen/Desktop/tropical-vxs/", f'{i:07}'), vx)
        ##write out label bias information
        u, c  = np.unique(vx['label'], return_counts = True)
        class_divergence = abs(np.diff(c/sum(c)))# higher more balanced sample
    
    del canopy
    return args








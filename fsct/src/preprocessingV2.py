import open3d as o3d
import numpy as np
import pandas as pd
from src.tools import save_file, load_file
import math
import os
from tqdm import tqdm

#params.point_cloud = load_file("/home/harryjfowen/Desktop/small_train_set/lw.ply")

def pcd2voxels(params):
    #pandas to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(params.point_cloud[['x', 'y', 'z']].to_numpy().astype('double'))
    #denoise
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=1)
    params.point_cloud = params.point_cloud.loc[np.array(ind)].reset_index(drop=True)
    params.point_cloud.rename(columns = {'scalar_refl':'refl'}, inplace = True)
    #SORT OUT LABEL COLUMNS
    params.point_cloud.set_axis(labels=['label' if c.endswith('label') else c for c in params.point_cloud], axis=1, inplace=True)
    if 'label' in params.point_cloud.columns:
        params.point_cloud = params.point_cloud.astype({"label": int})

    idx_list = []
    def f_traverse(node, node_info):
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

    dim_lengths = params.point_cloud.max()-params.point_cloud.min()
    maxSize = pd.Series.max(dim_lengths[:3])
    D = math.ceil(math.log(maxSize,2))

    octree = o3d.geometry.Octree(max_depth=D+1)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    octree.traverse(f_traverse)

    meta = []
    for i, idx in enumerate(tqdm(idx_list)):
        if len(idx) > 20000:
            vx = params.point_cloud.loc[idx].sample(n=20000)
        else:
            vx = params.point_cloud.loc[idx]
        np.save(os.path.join("/home/harryjfowen/Desktop/tropical-vxs/", f'{i:07}'), vx)
        ##write out label bias information
        u, c  = np.unique(vx['label'], return_counts = True)
        class_divergence = abs(np.diff(c/sum(c)))# higher more balanced sample








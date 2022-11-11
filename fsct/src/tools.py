import numpy as np
import glob
from multiprocessing import Pool, get_context
import pandas as pd
import os
import shutil
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from scipy.interpolate import griddata
from copy import deepcopy
from multiprocessing import get_context
from scipy.spatial import cKDTree
import string
import struct
from scipy import ndimage
from src import ply_io, pcd_io
import CSF
import open3d as o3d

def string_match(string1, string2):
    return all(any(x in y for y in string2) for x in string1)

class dict2class:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def get_fsct_path(location_in_fsct=""):
    current_wdir = os.getcwd()
    output_path = current_wdir[: current_wdir.index("fsct") + 4]
    if len(location_in_fsct) > 0:
        output_path = os.path.join(output_path, location_in_fsct)
    return output_path.replace("\\", "/")

def make_folder_structure(params):

    if params.mode == 'train':
    
        fsct_dir = params.wdir
        dir_list = [os.path.join(fsct_dir, "data"),
                    os.path.join(fsct_dir, "data", "train"),
                    os.path.join(fsct_dir, "data", "validation"),
                    os.path.join(fsct_dir, "data", "train", "sample_dir"),
                    os.path.join(fsct_dir, "data", "validation", "sample_dir")]

        i=1
        for directory in dir_list:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                if i==1: print("Created directories.")

            elif "sample_dir" in directory and params.clean_wdir:
                shutil.rmtree(directory, ignore_errors=True)
                os.makedirs(directory)

            elif i==1:
                print("Directories found.")
            i+=1

    if params.mode == 'predict':
        
        params.directory, params.filename = os.path.split(params.point_cloud)
        params.basename = os.path.splitext(params.filename)[0]

        if params.odir == '.':
            params.odir = os.path.join(params.directory, params.basename + '_LW')
        
        params.wdir = os.path.join(params.odir, params.basename + '.tmp')
        if not os.path.isdir(params.odir):
            os.makedirs(params.odir)

        if not os.path.isdir(params.wdir):
            os.makedirs(params.wdir)
        else:
            shutil.rmtree(params.wdir, ignore_errors=True)
            os.makedirs(params.wdir)
            
        if params.verbose:
            print('output directory:', params.odir)
            print('scratch directory:', params.wdir)
        
    return params
    
def voxelise(tmp, length, method='random', z=True):

    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    if z: tmp.loc[:, 'zz'] = tmp.z // length * length

    if method == 'random':
            
        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
            
        xD = {x:code() for x in tmp.xx.unique()}
        yD = {y:code() for y in tmp.yy.unique()}
        if z: zD = {z:code() for z in tmp.zz.unique()}
            
        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD) 
        if z: tmp.VX += tmp.zz.map(zD)
   
    elif method == 'bytes':
        
        code = lambda row: np.array([row.xx, row.yy] + [row.zz] if z else []).tobytes()
        tmp.loc[:, 'VX'] = self.pc.apply(code, axis=1)
        
    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')
 
    return tmp 


def downsample(pc, vlength, 
               accurate=False,
               keep_columns=[], 
               keep_points=False,
               voxel_method='random', 
               return_VX=False,
               verbose=False):

    """
    Downsamples a point cloud so that there is one point per voxel.
    Points are selected as the point closest to the median xyz value
    
    Parameters
    ----------
    
    pc: pd.DataFrame with x, y, z columns
    vlength: float
    
    
    Returns
    -------
    
    pd.DataFrame with boolean downsample column
    
    """

    pc = pc.drop(columns=[c for c in ['downsample', 'VX'] if c in pc.columns])   
    columns = pc.columns.to_list() + keep_columns # required for tidy up later
    if return_VX: columns += ['VX']
    pc = voxelise(pc, vlength, method=voxel_method)

    if accurate:
        # groubpy to find central (closest to median) point
        groupby = pc.groupby('VX')
        pc.loc[:, 'mx'] = groupby.x.transform(np.median)
        pc.loc[:, 'my'] = groupby.y.transform(np.median)
        pc.loc[:, 'mz'] = groupby.z.transform(np.median)
        pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].to_numpy(dtype=np.float32) - 
                                           pc[['mx', 'my', 'mz']].to_numpy(dtype=np.float32), axis=1)
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True

    else:
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.VX.duplicated(), 'downsample'] = True
        
    if keep_points:
        return pc[columns + ['downsample']]
    else:
        return pc.loc[pc.downsample][columns]
    
def compute_plot_centre(pc):
    """calculate plot centre"""
    plot_min, plot_max = pc[['x', 'y']].min(), pc[['x', 'y']].max()
    return (plot_min + ((plot_max - plot_min) / 2)).values

def compute_bbox(pc):
    bbox_min = pc.min().to_dict()
    bbox_min = {k + 'min':v for k, v in bbox_min.items()}
    bbox_max = pc.max().to_dict()
    bbox_max = {k + 'max':v for k, v in bbox_max.items()}
    return dict2class({**bbox_min, **bbox_max})
    
def load_file(filename, additional_headers=False, verbose=False):
    
    file_extension = os.path.splitext(filename)[1]
    headers = ['x', 'y', 'z']

    if file_extension == '.las' or file_extension == '.laz':

        import laspy

        inFile = laspy.read(filename)
        pc = np.vstack((inFile.x, inFile.y, inFile.z))
        pc = pd.DataFrame(data=pc.T, columns=['x', 'y', 'z'])

    elif file_extension == '.ply':
        pc = ply_io.read_ply(filename)
        
    elif file_extension == '.pcd':
        pc = pcd_io.read_pcd(filename)
        
    else:
        raise Exception('point cloud format not recognised' + filename)

    original_num_points = len(pc)
    
    if verbose: print(f'read in {filename} with {len(pc)} points')
   
    if additional_headers:
        return pc, [c for c in pc.columns if c not in ['x', 'y', 'z']]
    else: return pc


def save_file(filename, pointcloud, additional_fields=[], verbose=False):
    if verbose:
        print('Saving file:', filename)
        
    cols = ['x', 'y', 'z'] + additional_fields

    if filename.endswith('.las'):
        las = laspy.create(file_version="1.4", point_format=7)
        las.header.offsets = np.min(pointcloud[:, :3], axis=0)
        las.header.scales = [0.001, 0.001, 0.001]

        las.x = pointcloud[:, 0]
        las.y = pointcloud[:, 1]
        las.z = pointcloud[:, 2]

        if len(additional_fields) != 0:
            additional_fields = additional_fields[3:]

            #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
            col_idxs = list(range(3, pointcloud.shape[1]))
            additional_fields.reverse()

            col_idxs.reverse()
            for header, i in zip(additional_fields, col_idxs):
                column = pointcloud[:, i]
                if header in ['red', 'green', 'blue']:
                    setattr(las, header, column)
                else:
                    las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                    setattr(las, header, column)
        las.write(filename)
        if not verbose:
            print("Saved.")

    elif filename.endswith('.csv'):
        pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
        print("Saved to:", filename)

    elif filename.endswith('.ply'):

        if not isinstance(pointcloud, pd.DataFrame):
            cols = list(set(cols))
            pointcloud = pd.DataFrame(pointcloud, columns=cols)
        
        ply_io.write_ply(filename, pointcloud[cols])
        print("Saved to:", filename)

def low_resolution_hack_mode(point_cloud, num_iterations, min_spacing, num_procs):
    print('Using low resolution point cloud hack mode...')
    print('Original point cloud shape:', point_cloud.shape)
    point_cloud_original = deepcopy(point_cloud)
    for i in range(num_iterations):
        duplicated = deepcopy(point_cloud_original)

        duplicated[:, :3] = duplicated[:, :3] + np.hstack(
                (np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1)),
                 np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1)),
                 np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1))))
        point_cloud = np.vstack((point_cloud, duplicated))
        point_cloud = subsample_point_cloud(point_cloud, min_spacing, num_procs)
    print('Hacked point cloud shape:', point_cloud.shape)
    return point_cloud

#need to find out how to remove inf values from nbrs query!!
def verticality(arr, knn, r):
    ids = np.arange(arr.shape[0])
    nbrs = cKDTree(arr)
    dist, nbrs_idx = nbrs.query(arr, k=knn, distance_upper_bound = r)
    V = np.zeros([arr.shape[0], 1], dtype=float)
    for i in ids:
        vec = arr[nbrs_idx[i]][:, 2]
        vec = vec[~np.isnan(vec)]
        V[i] = (np.max(vec)-np.min(vec))
    V = V/np.percentile(V,99)
    V[np.isnan(V)] = 0
    return V

def classify_ground(params):

    print("Classifying ground points...")
    csf = CSF.CSF()

    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.33
    csf.params.class_threshold = 0.33

    xyz=params.pc[['x', 'y', 'z']].to_numpy().astype('double')

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()

    csf.do_filtering(ground, non_ground)

    tmpIDX = params.pc.loc[ground].index.to_numpy()

    # Filter remaining stumps    
    V = verticality(params.pc.loc[tmpIDX][['x','y','z']].to_numpy(), 50, 1000)
    groundIDX = tmpIDX[np.where(V.flatten() < 0.75)]

    return params.pc.index.isin(groundIDX)


def make_dtm(params):
    
    """ 
    This function will generate a Digital Terrain Model (dtm) based on the terrain labelled points.
    """

    if params.verbose: print("Making dtm...")

    params.grid_resolution = .5

    ### voxelise, identify lowest points and create DTM
    params.pc = voxelise(params.pc, params.grid_resolution, z=False)
    VX_map = params.pc.loc[~params.pc.VX.duplicated()][['xx', 'yy', 'VX']]
    ground = params.pc.loc[params.pc.label == params.terrain_class]#locate labels that equal the terrain class number
    ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.median)
    ground = ground.loc[ground.z == ground.zmin]
    ground = ground.loc[~ground.VX.duplicated()]

    X, Y = np.meshgrid(np.arange(params.pc.xx.min(), params.pc.xx.max() + params.grid_resolution, params.grid_resolution),
                       np.arange(params.pc.yy.min(), params.pc.yy.max() + params.grid_resolution, params.grid_resolution))

    ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy']) 
    ground_arr = pd.merge(ground_arr, VX_map, on=['xx', 'yy'], how='outer') # map VX to ground_arr
    ground_arr = pd.merge(ground[['z', 'VX']], ground_arr, how='right', on=['VX']) # map z to ground_arr
    ground_arr.sort_values(['xx', 'yy'], inplace=True)
    
    # loop over incresing size of window until no cell are nan
    ground_arr.loc[:, 'ZZ'] = np.nan
    size = 3 
    while np.any(np.isnan(ground_arr.ZZ)):
        ground_arr.loc[:, 'ZZ'] = ndimage.generic_filter(ground_arr.z.values.reshape(*X.shape), # create raster, 
                                                         lambda z: np.nanmedian(z), size=size).flatten()
        size += 2

    ground_arr[['xx', 'yy', 'ZZ']].to_csv(os.path.join(params.odir, f'{params.basename}.dem.csv'), index=False)
    
    # apply to all points   
    MAP = ground_arr.set_index('VX').ZZ.to_dict()
    params.pc.loc[:, 'n_z'] = params.pc.z - params.pc.VX.map(MAP)  
    
    return params

def denoise(cloud, knn, std):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(cloud[['x', 'y', 'z']].to_numpy().astype('double'))

    _, ind = pcd.remove_statistical_outlier(nb_neighbors=knn,std_ratio=std)

    denoise_idx = np.array(ind)
    
    return(denoise_idx)


def smooth_classifcation(classified_arr, raw_cloud, knn):

    nbrs = KDTree(cloud[['x', 'y', 'z']].values)
    _, indices = nbrs.query(cloud[['x', 'y', 'z']], k=knn)

    classes = classified_arr["label"].values.T
    new_class = np.zeros(raw_cloud.shape[0])
    class_ = classes[indices]

    for i in range(len(indices)):
        unique, count = np.unique(class_[i, :], return_counts=True)
        new_class[i] = unique[np.argmax(count)]

    cloud = raw_cloud.loc[raw_cloud.index, 'label'] = new_class
    return cloud

def nn_dist(cloud,knn):

    kd_tree = KDTree(cloud[['x', 'y', 'z']].values)

    dist, _ = kd_tree.query(cloud[['x', 'y', 'z']].values, k=knn)

    return np.mean(dist[:, 1:]) + (np.std(dist[:, 1:]))
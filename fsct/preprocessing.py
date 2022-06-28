import os
import time
import threading
import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from tqdm import tqdm
from jakteristics import compute_features

from tools import *

def save_pts_v2(params, I, bx, by, bz):

    pc = params.pc.loc[(params.pc.x.between(bx, bx + params.box_dims[0])) &
                       (params.pc.y.between(by, by + params.box_dims[0])) &
                       (params.pc.z.between(bz, bz + params.box_dims[0]))]    
    """
    COMPUTE GEOMETRIC FEATURES OF POINTS
    """ 
    features = compute_features(pc.astype('double')[['x','y','z']], search_radius=0.10, feature_names=["linearity","surface_variation"], num_threads=1)
    
    if len(pc) > params.min_points_per_box:

            if len(pc) > params.max_points_per_box:
                features[:, 1]=1-features[:, 1]
                auxilary = np.column_stack((pc['scalar_refl'],features))
                auxilary = preprocessing.minmax_scale(auxilary, feature_range=(0,1))
                weights = np.amax(auxilary,axis=1)
                pc = pc.sample(n=params.max_points_per_box,weights=weights)

            np.save(os.path.join(params.working_dir, f'{I:>09}'), pc[['x', 'y', 'z']].values)
            #np.savetxt(os.path.join(params.working_dir, f'{I:07}.txt'), pc[['x', 'y', 'z']].values)

def save_pts(params, I, bx, by, bz):

    pc = params.pc.loc[(params.pc.x.between(bx, bx + params.box_dims[0])) &
                       (params.pc.y.between(by, by + params.box_dims[0])) &
                       (params.pc.z.between(bz, bz + params.box_dims[0]))]

    if len(pc) > params.min_points_per_box:

        if len(pc) > params.max_points_per_box:
            pc = pc.sample(n=params.max_points_per_box)

        #np.save(os.path.join(params.working_dir, f'{I:07}'), pc[['x', 'y', 'z']].values)
        np.savetxt(os.path.join(params.working_dir, f'{I:07}.txt'), pc[['x', 'y', 'z']].values)


def Preprocessing(params):
    
    
    # classify ground returns using cloth simulation
    params.ground_idx = classify_ground(params)
    #veg_idx = params.pc[~np.in1d(np.arange(len(params.pc)), ground_idx].index.values

    # remove ground points 
    params.pc = params.pc.drop(params.ground_idx)

    # compute plot centre, global shift and bounding box
    params.plot_centre = compute_plot_centre(params.pc)
    params.global_shift = params.pc[['x', 'y', 'z']].mean()-params.pc[['x', 'y', 'z']].mean()
    params.bbox = compute_bbox(params.pc[['x', 'y', 'z']])

    if params.subsample: # subsample if specified
        if params.verbose: print('downsampling to: %s m' % params.subsampling_min_spacing)
        params.pc = downsample(params.pc, params.subsampling_min_spacing, 
                             accurate=False, keep_points=False)

    # apply global shift
    if params.verbose: print('global shift:', params.global_shift.values)
    params.pc[['x', 'y', 'z']] = params.pc[['x', 'y', 'z']] - params.global_shift
	
    params.pc.reset_index(inplace=True)
    params.pc.loc[:, 'pid'] = params.pc.index

    # generate bounding boxes
    xmin, xmax = np.floor(params.pc.x.min()), np.ceil(params.pc.x.max())
    ymin, ymax = np.floor(params.pc.y.min()), np.ceil(params.pc.y.max())
    zmin, zmax = np.floor(params.pc.z.min()), np.ceil(params.pc.z.max())

    box_overlap = params.box_dims[0] * params.box_overlap[0]

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)

    # multithread segmenting points into boxes and save
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts, args=(params, i, bx, by, bz)))

    for x in tqdm(threads, 
                  desc='generating data blocks',
                  disable=False if params.verbose else True):
        x.start()

    for x in threads:
        x.join()

    if params.verbose: print("Preprocessing done in {} seconds\n".format(time.time() - start_time))
    
    return params

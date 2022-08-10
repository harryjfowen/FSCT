import os
import time
import threading
import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from tqdm import tqdm
from src.tools import *

def save_pts(params, I, bx, by, bz):

    pc = params.pc.loc[(params.pc.x.between(bx, bx + params.box_dims[0])) &
                       (params.pc.y.between(by, by + params.box_dims[1])) &
                       (params.pc.z.between(bz, bz + params.box_dims[2]))]

    if len(pc) > params.min_pts:
        if len(pc) > params.max_pts:
            pc = pc.sample(n=params.max_pts,weights=params.weights[pc.index])

        if params.mode == 'train':
            np.save(os.path.join(params.odir, f'{I:07}'), pc[['x', 'y', 'z', 'label']].values)

        else:
            #np.savetxt(os.path.join(params.wdir, f'{I:07}.txt'), pc[['x', 'y', 'z']].values)
            np.save(os.path.join(params.wdir, f'{I:07}'), pc[['x', 'y', 'z']].values)

def Preprocessing(params):
    
    if params.verbose: print('\n----- preprocessing started -----')
    start_time = time.time()

    make_folder_structure(params)

    if params.mode == 'train':
        if ([any(k for k in params.point_cloud if 'train' in k)]):
            print("Preprocessing training point clouds...")
        
        else:
            raise Exception("No training data found!")
        
        if ([any(k for k in params.point_cloud if 'validation' in k)]):
            print("Preprocessing validation point clouds...")
    else: params.point_cloud=[params.point_cloud]
    
    start_idx=0
    for pc_file in params.point_cloud:
        if params.verbose: print('\n...')

        #changing output directory if preprocesing for training purposes
        if params.mode == 'train':
            params.odir = os.path.dirname(pc_file) + '/sample_dir'

        #read in ply files
        params.pc, params.headers = load_file(filename=pc_file,
                                       additional_headers=True,
                                       verbose=params.verbose)

        if params.point_spacing != 0: 
            if params.verbose: print('downsampling to: %s m' % params.point_spacing)
            params.pc = downsample(params.pc, params.point_spacing, 
                                   accurate=False, keep_points=False)
        
        # Denoise the point cloud usign a statistical filter
        if params.mode == 'predict':
            print("Denoising using statistical outlier filter...")
            params.pc = params.pc.iloc[denoise(params.pc, 30, 3.0)]

        # calculate random sampling weights
        try:
            params.weights = pow(preprocessing.minmax_scale(params.pc['refl'].values, feature_range=(1,100)),2)
        except:
            params.weights = np.ones(len(params.pc))

        # Reconfigure plot
        params.plot_centre = compute_plot_centre(params.pc)
        params.global_shift = params.pc[['x', 'y', 'z']].mean()#-params.pc[['x', 'y', 'z']].mean()
        params.bbox = compute_bbox(params.pc[['x', 'y', 'z']])

        if params.verbose: print('global shift:', params.global_shift.values)
        params.pc[['x', 'y', 'z']] = params.pc[['x', 'y', 'z']] - params.global_shift
	
        params.pc.reset_index(inplace=True)
        params.pc.loc[:, 'pid'] = params.pc.index

        #Classifying ground returns using cloth simulation
        params.grdidx = classify_ground(params)
        if params.mode == 'predict':
            params.grd = params.pc.loc[params.pc.index[params.grdidx]]
        else:
            params.pc = params.pc.drop(params.pc.index[params.grdidx])

        # generate bounding boxes
        xmin, xmax = np.floor(params.pc.x.min()), np.ceil(params.pc.x.max())
        ymin, ymax = np.floor(params.pc.y.min()), np.ceil(params.pc.y.max())
        zmin, zmax = np.floor(params.pc.z.min()), np.ceil(params.pc.z.max())

        box_overlap = params.box_dims * params.box_overlap

        x_cnr = np.arange(xmin - box_overlap[0], xmax + box_overlap[0], box_overlap[0])
        y_cnr = np.arange(ymin - box_overlap[1], ymax + box_overlap[1], box_overlap[1])
        z_cnr = np.arange(zmin - box_overlap[2], zmax + box_overlap[2], box_overlap[2])

        # multithread segmenting points into boxes and save
        threads = []
        for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr), start_idx):
            threads.append(threading.Thread(target=save_pts, args=(params, i, bx, by, bz)))

        for x in tqdm(threads, 
                    desc='generating data blocks',
                    disable=False if params.verbose else True):
            x.start()

        for x in threads:
            x.join()

        start_idx+=i
        print('')

    if params.verbose: print("---------- Preprocessing done in {} seconds ----------\n".format(time.time() - start_time))
    
    return params

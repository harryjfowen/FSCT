import sys
import os
import argparse
import pickle
import numpy as np
os.chdir('/home/harryowen/FSCT/fsct')

from src.tools import *
import CSF
import scipy

filepath = sys.argv[1]

pcRAW, additional_headers = load_file(filename=filepath,additional_headers=True,verbose=True)

# classify ground 
csf = CSF.CSF()
csf.params.bSloopSmooth = True
csf.params.cloth_resolution = 0.10
csf.params.class_threshold = 0.30 
csf.setPointCloud(pcRAW[['x', 'y', 'z']].to_numpy().astype('double'))
ground = CSF.VecInt()
non_ground = CSF.VecInt()

csf.do_filtering(ground, non_ground)
pcRAW = pcRAW.loc[pcRAW.index.values[non_ground]].reset_index(drop=True)

pc = downsample(pcRAW, 0.01)
idx = pc.index.values

# filter based on reflectance 
#cutoff = scipy.stats.mode(pc['refl'].values, axis=0)
boolRfl = (pc['scalar_refl'] <= -6)

#	Extract features on canopy points with reflectance below a threshold (i.e. leaf points)
featuresFine = compute_features(pc[['x', 'y', 'z']][boolRfl].to_numpy().astype('double'),
								search_radius=0.05, feature_names=["linearity","verticality"],
								num_threads=30)

boolFine = (featuresFine[:, 0] < 0.75) & (featuresFine[:, 1] < 0.75)


# Filter values of leaf points that still resemble wood points
leafIDX = idx[boolRfl][boolFine]
featuresCoarse = compute_features(pc[['x', 'y', 'z']].loc[leafIDX].to_numpy().astype('double'),
								  search_radius=0.20, feature_names=["linearity","verticality"], 
								  num_threads=30)

boolCoarse = (featuresCoarse[:, 0] < 0.80) & (featuresCoarse[:, 1] < 0.80)

woodIDX = idx[~boolRfl]
woodIDX = woodIDX[denoise(pc.loc[woodIDX],10,1.0)]
opt_nn = nn_dist(pc.loc[woodIDX],16)
woodIDX = woodIDX[cluster_filter(pc.loc[woodIDX], opt_nn, 10, 0.66, 'wood')]

leafIDX = leafIDX[boolCoarse]
opt_nn = nn_dist(pc.loc[leafIDX],16)
leafIDX = leafIDX[cluster_filter(pc.loc[leafIDX], opt_nn, 10, 0.66, 'leaf')]

# add label column
pc.loc[leafIDX, 'label'] = int(2)
pc.loc[woodIDX, 'label'] = int(1)
pc = pc.dropna()

# upscale and average labels at neighbourhood 
#labels = smooth_classifcation(pc, pcRAW, 5)
#pcRAW['label'] = labels

# write out cloud with labels 
pc = pc[["x","y","z","scalar_refl","label"]]
pc.columns = ['x','y','z','refl','label']
save_file(os.path.join(os.path.dirname(filepath), os.path.basename(os.path.splitext(filepath)[0]) + '_LW.ply'),
					   pc[["x","y","z","refl","label"]], ['refl','label'], verbose = True)

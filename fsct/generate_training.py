import sys
import os
import argparse
import pickle
import numpy as np
os.chdir('/home/harryjfowen/FSCT/fsct')

from tools import *
import CSF

"""
	READ IN DATA
"""
pcRAW, additional_headers = load_file(filename='/home/harryjfowen/Desktop/SPA19_011.ply',additional_headers=True,verbose=True)

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
boolRfl = (pc['refl'] <= -5)

#	Extract features on canopy points with reflectance below a threshold (i.e. leaf points)
featuresFine = compute_features(pc[['x', 'y', 'z']][boolRfl].to_numpy().astype('double'), search_radius=0.05, feature_names=["linearity","verticality"], num_threads=8)
boolFine = (featuresFine[:, 0] < 0.75) & (featuresFine[:, 1] < 0.75)

# clean up wood file 
woodIDX = idx[~boolRfl]

woodIDX = woodIDX[denoise(pc.loc[woodIDX])]
woodIDX = woodIDX[cluster_filter(pc.loc[woodIDX], 0.05, 100, 0.66)]

# Filter values of leaf points that still resemble wood points
leafIDX = idx[boolRfl][boolFine]

##2nd Pass but at a coarser scale. Now only leaf points should remain
featuresCoarse = compute_features(pc[['x', 'y', 'z']].loc[leafIDX].to_numpy().astype('double'), search_radius=0.25, feature_names=["linearity","verticality"], num_threads=8)
boolCoarse = (featuresCoarse[:, 0] < 0.80) & (featuresCoarse[:, 1] < 0.80)

leafIDX = leafIDX[boolCoarse]
leafClust = cluster_filter(pc.loc[leafIDX], 0.05, 100, 0.80)
#woodIDX = np.hstack((woodIDX, leafIDX[leafClust]))
leafIDX = leafIDX[~leafClust]

save_file('/home/harryjfowen/Desktop/spa19_leaf.ply', pcRAW.loc[leafIDX], additional_headers, verbose = True)
save_file('/home/harryjfowen/Desktop/spa19_wood.ply', pcRAW.loc[woodIDX], additional_headers, verbose = True)

# add label column
pc.loc[leafIDX, 'label'] = int(2)
pc.loc[woodIDX, 'label'] = int(1)
pc = pc.dropna()

# upscale and average labels at neighbourhood 
out = smooth_classifcation(pc, pcRAW, 5)

# write out cloud with labels 
save_file('/home/harryjfowen/Desktop/spa06_segtest_KNN.ply', out, additional_headers + ['label'], verbose = True)

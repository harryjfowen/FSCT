import sys
import os
import argparse
import pickle
import numpy as np
os.chdir('/home/harryjfowen/FSCT/fsct')

from tools import *
import CSF
import scipy
"""
	READ IN DATA
"""
pc, additional_headers = load_file(filename='/home/harryjfowen/Desktop/spa06_training/SPA06_000.downsample.ply',additional_headers=True,verbose=True)
pc = downsample(pc, 0.025)

# classify ground 
csf = CSF.CSF()

csf.params.bSloopSmooth = True
csf.params.cloth_resolution = 0.10
csf.params.class_threshold = 0.30 

#statistical dennoising
# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pc[['x', 'y', 'z']].to_numpy().astype('double'))
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
# xyz = np.asarray(cl.points)

xyz = pc[['x', 'y', 'z', 'refl']].to_numpy()
csf.setPointCloud(xyz)
ground = CSF.VecInt()
non_ground = CSF.VecInt()

csf.do_filtering(ground, non_ground)
#pc = pc.drop(ground).reset_index(drop=True)
xyz = xyz[ground]

##
featuresFine = compute_features(xyz[:,:3].astype('double'), search_radius=0.06, feature_names=["surface_variation","linearity"], num_threads=8)
featuresCoarse = compute_features(xyz[:,:3].astype('double'), search_radius=0.30, feature_names=["linearity"], num_threads=8)

refl_cutoff=scipy.stats.scoreatpercentile(xyz[:, 3],75)
idx = np.where((xyz[:, 3] > -5) | (featuresFine[:, 0] < 0.1) | (featuresFine[:, 1] > 75) | (featuresCoarse[:, 0] > 75))

pc = pd.DataFrame(xyz, columns = ['x','y','z','refl'])
pc.loc[idx[0], 'label'] = int(1)
pc['label'] = pc['label'].fillna(2)

save_file('/home/harryjfowen/Desktop/spa06_segtest_lw.ply', pc, ['label'], verbose = True)





#test=pc[['x', 'y', 'z']].to_numpy().astype('double')
#cluster_filter(xyz[:,:3], 0.05, 0.1)





import sys
import os
import argparse
import pickle
import numpy as np
os.chdir('/home/harryowen/FSCT/fsct')
import scipy
from src.tools import *
import CSF
from sklearn import preprocessing
from scipy.spatial import KDTree

#filepath = sys.argv[1]
#pcRAW, additional_headers = load_file(filename=filepath,additional_headers=True,verbose=True)

pcRAW = load_file('/home/harryowen/Desktop/tile2.ply')
pc = pc.iloc[denoise(pcRAW, 25, 0.5)]
pc = downsample(pcRAW, 0.01).reset_index(drop=True)
idx = pc.index.values
pc.rename(columns = {'scalar_refl':'refl'}, inplace = True)

##classify ground 
csf = CSF.CSF()
csf.params.bSloopSmooth = True
csf.params.cloth_resolution = 0.10
csf.params.class_threshold = 0.30 
csf.setPointCloud(pc[['x', 'y', 'z']].to_numpy().astype('double'))
ground = CSF.VecInt()
non_ground = CSF.VecInt()

csf.do_filtering(ground, non_ground)
pc = pc.loc[pcRAW.index.values[non_ground]].reset_index(drop=True)

##  Calculate features on gpu
features = add_features_gpu(pc)
extra_cols = features.columns.tolist()
features = pd.DataFrame(preprocessing.minmax_scale(pd.concat([features,pc['refl']],axis=1), feature_range=(0,1),axis=0), columns=extra_cols+['refl'])
features = pd.concat([pc[['x','y','z']],features],axis=1)
save_file('/home/harryowen/Desktop/xyz-features.ply',features, additional_fields=extra_cols+['refl'])

#basic thresholding of features
is_wood=np.any(features[extra_cols+['refl']] > 0.85, axis=1)
pc.loc[is_wood, 'label'] = 0
pc.loc[~is_wood, 'label'] = 1

##smooth out gmm
def smooth_classifcation(df, knn):
    nbrs = scipy.spatial.KDTree(df[['x','y','z']].values,compact_nodes=True)
    _, indices = nbrs.query(df[['x','y','z']].values, k=knn, workers=-1)
    new_label = np.zeros(df.shape[0],dtype=int)
    label_ = df['label'].values[indices]
    for i in range(len(indices)):
        unique, count = np.unique(label_[i, :], return_counts=True)
        new_label[i] = unique[np.argmax(count)]
    df.loc[:, 'label'] = new_label
    return df

smoothed = smooth_classifcation(pc, 16)
save_file('/home/harryowen/Desktop/label-smoove.ply',smoothed, additional_fields=['label'])

##filter clusters of shapes that do not match each class type















# #Run gmm
# from sklearn.mixture import GaussianMixture as GMM
# def classify(variables, n_classes):
#     gmm = GMM(n_components=n_classes)
#     gmm.fit(variables)
#     return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)

# labels = classify(test, 3)[0]
# out = pandas.concat([pc, pandas.DataFrame(labels, columns=["label"])], axis=1)
# save_file('/home/harryowen/Desktop/xyz-gmm.ply',out, additional_fields=['label'])

# ##smooth out gmm
# def smooth_classifcation(df, labels, knn):
#     nbrs = scipy.spatial.KDTree(df[['x','y','z']].values,compact_nodes=True)
#     _, indices = nbrs.query(df[['x','y','z']].values, k=knn, workers=-1)
#     new_label = np.zeros(df.shape[0],dtype=int)
#     label_ = labels[indices]
#     for i in range(len(indices)):
#         unique, count = np.unique(label_[i, :], return_counts=True)
#         new_label[i] = unique[np.argmax(count)]
#     df.loc[:, 'label'] = new_label
#     return df

# smoothed = smooth_classifcation(pc, labels, 50)
# save_file('/home/harryowen/Desktop/xyz-gmm-smoove.ply',smoothed, additional_fields=['label'])

# ##upscale
# def upscale(wood, leaf, knn):
#     nbrs = scipy.spatial.KDTree(wood[['x','y','z']].values,compact_nodes=True)
#     _, indices = nbrs.query(leaf[['x','y','z']].values, k=knn, workers=-1)
#     new_label = np.zeros(df.shape[0],dtype=int)
#     label_ = labels[indices]
#     for i in range(len(indices)):
#         unique, count = np.unique(label_[i, :], return_counts=True)
#         new_label[i] = unique[np.argmax(count)]
#     df.loc[:, 'label'] = new_label
#     return df





















# unique, count = np.unique(class_[i, :], return_counts=True)


# x=[np.argmax(np.unique(idx, return_counts=True)[1] for idx in indices)]

# labels = np.zeros(len(indices),dtype=int)
# for idx in indices:
#     labels[idx] = np.argmax(np.unique(idx, return_counts=True)[1])



# rflidx = np.where(pc['surf_refl'] > -5)
# nbrs = KDTree(pc[['x','y','z']].values)
# dist, indices = nbrs.query(pc[['x','y','z']].values, k=100, distance_upper_bound = np.inf)
# nn_max = np.max(pc['scalar_refl'].values[indices],axis=1)
# out=pandas.concat([pc, pandas.DataFrame(nn_max,columns=['refl'])], axis=1)
# save_file('/home/harryowen/Desktop/fin04_meanrefl.ply',out, additional_fields=['scalar_refl','refl'])

# nbrs = scipy.spatial.KDTree(pc[['x','y','z']].values,compact_nodes=True)
# dist, indices = nbrs.query(pc[['x','y','z']].values, k=100, workers=-1)





# from sklearn.mixture import GaussianMixture as GMM
# def classify(variables, n_classes):
#     gmm = GMM(n_components=n_classes)
#     gmm.fit(variables)
#     return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)

# pred = classify(features, 3)
# out = pandas.concat([pc, pandas.DataFrame(pred[0], columns=["label"])], axis=1)
# save_file('/home/harryowen/Desktop/xyz-gmm.ply',out, additional_fields=['label'])














# weights = np.max(features,axis=1)
# out = pandas.concat([pc[['x','y','z','refl']], weights], axis=1)
# out.columns = ['x','y','z','refl','weights']
# save_file('/home/harryowen/Desktop/xyz-features.ply',out, additional_fields=['refl','weights'])






# # Counting the number of occurrences of each value in the ith instance
#         # of class_.
#         unique, count = np.unique(class_[i, :], return_counts=True)
#         # Appending the majority class into the output variable.
#         c_maj[i] = unique[np.argmax(count)]








# out = pandas.concat([pc, features], axis=1)
# save_file('/home/harryowen/Desktop/xyz-features.ply',out, additional_fields=['l20','s20','l100','s100','l200','s200'])

# nbrs = KDTree(pc[['x','y','z']].values)
# dist, indices = nbrs.query(pc[['x','y','z']].values, k=16, distance_upper_bound = 1000)

# f_arr = features.to_numpy()
# nn_max = np.max(f_arr[indices],axis=2)
# f_max = np.max(nn_max,axis=1)

# out = pandas.concat([pc, pandas.DataFrame(f_max, columns=["fmax"])], axis=1)
# save_file('/home/harryowen/Desktop/xyz-features.ply',out, additional_fields=['fmax'])

# # add label column
# pc.loc[leafIDX, 'label'] = int(2)
# pc.loc[woodIDX, 'label'] = int(1)
# pc = pc.dropna()

# # upscale and average labels at neighbourhood 
# #labels = smooth_classifcation(pc, pcRAW, 5)
# #pcRAW['label'] = labels

# # write out cloud with labels 
# pc = pc[["x","y","z","scalar_refl","label"]]
# pc.columns = ['x','y','z','refl','label']
# save_file(os.path.join(os.path.dirname(filepath), os.path.basename(os.path.splitext(filepath)[0]) + '_LW.ply'),
# 					   pc[["x","y","z","refl","label"]], ['refl','label'], verbose = True)

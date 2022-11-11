import os
import time
import shutil
import sys
import glob

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from tqdm.auto import tqdm

from abc import ABC
import torch
import torch_geometric
from torch_geometric.data import Dataset, DataLoader, Data
from src.model import Net
import torch.nn.functional as F

from src.tools import save_file, make_dtm, get_fsct_path

sys.setrecursionlimit(10 ** 8) # Can be necessary for dealing with large point clouds.

class TestingDataset(Dataset, ABC):
    
    '''
    Create testing dataset function for torch data loader. 
    '''
    def __init__(self, root_dir, points_per_box, device):
        self.filenames = glob.glob(os.path.join(root_dir, '*.npy'))
        self.device = device
        self.points_per_box = points_per_box

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        pos = point_cloud[:, :3]
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)

        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos = pos - local_shift
        data = Data(pos=pos, x=None, local_shift=local_shift)
        return data


def SemanticSegmentation(gpu,args):

    '''
    Script to predict probability of leaf and wood from point cloud data. 
    '''

    if args.verbose: print('----- semantic segmentation started -----')
    args.sem_seg_start_time = time.time()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose: print('using:', args.device)

    # generates pytorch dataset iterable
    test_dataset = TestingDataset(root_dir=args.wdir,
                                  points_per_box=args.max_pts,
                                  device=args.device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0)

    '''
    Initialise model
    '''
    
    model = Net(num_classes=2).to(args.device)

    args.model_filepath = os.path.join(get_fsct_path(),'model',args.model)
    model.load_state_dict(torch.load(args.model_filepath, map_location=args.device), strict=False)
    model.eval()

    with torch.no_grad():
        output_list = []
        for data in tqdm(test_loader, disable=False if args.verbose else True):
            data = data.to(args.device)
            out = model(data)
            print(out)
            batches = np.unique(data.batch.cpu())
            out = torch.sigmoid(out.cpu().detach())#, axis=1)
            print(out)
            pos = data.pos.cpu()
            output = np.hstack((pos, out))

            for batch in batches:
                outputb = np.asarray(output[data.batch.cpu() == batch])
                outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch:3 + (3 * batch)]
                output_list.append(outputb)
#             break

        classified_pc = np.vstack(output_list)

    del outputb, out, batches, pos, output  

    if args.verbose: print("Choosing most confident labels...")
    print(classified_pc[:, :])
    #Build KD tree neighbourhoods to both project results onto original cloud (before downsampling) and smooth out classifcation results
    neighbours = NearestNeighbors(n_neighbors=16, 
                                  algorithm='kd_tree', 
                                  metric='euclidean', 
                                  radius=0.05).fit(classified_pc[:, :3])
    _, indices = neighbours.kneighbors(args.pc[['x', 'y', 'z']].values)

    args.pc = args.pc.drop(columns=[c for c in args.pc.columns if c in ['label', 'pWood', 'pLeaf']])

    #Calculate summary labels and probabilities within KD tree neighbourhoods
    labels = np.zeros((args.pc.shape[0], 2))
    labels[:, :2] = np.median(classified_pc[indices][:, :, -2:], axis=1)
    args.pc.loc[args.pc.index, 'label'] = np.argmax(labels[:, :2], axis=1)

    #Collect probabilites from classification both classes
    probs = pd.DataFrame(index=args.pc.index, data=labels[:, :2], columns=['pWood','pLeaf'])
    args.pc = args.pc.join(probs)

    # attribute points as wood if any points have
    # a wood probability > args.is_wood (Morel et al. 2020)
    #is_wood = np.any(classified_pc[indices][:, :, -2] > args.is_wood, axis=1)
    #args.pc.loc[is_wood, 'label'] = args.wood_class

    ##Add ground points and labeling from prior CSF classifcation
    # args.grd.loc[:, 'label'] = args.terrain_class
    # args.grd.loc[:, ['pWood','pLeaf']] = 0
    # args.pc = args.pc.append(args.grd, ignore_index=True)

    # #calculate dtm
    # args.= make_dtm(args.
    # args.pc.loc[args.pc.n_z <= args.ground_height_threshold, 'label'] = args.terrain_class

    save_file(os.path.join(args.odir, args.basename + '.segmented.' + args.output_fmt), 
              args.pc, additional_fields=['label', 'pWood', 'pLeaf'] + args.headers)

    args.sem_seg_total_time = time.time() - args.sem_seg_start_time
    if not args.keep_npy: [os.unlink(f) for f in test_dataset.filenames]
    print("semantic segmentation done in", args.sem_seg_total_time, 's\n')
    
    return params

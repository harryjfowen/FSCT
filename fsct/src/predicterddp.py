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
import torch.nn as nn

from torch_geometric.data import Dataset, DataLoader, Data
from src.model import Net
import torch.nn.functional as F

from src.tools import save_file, make_dtm, get_fsct_path
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

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

def load_checkpoint(path, rank, model):

    torch.distributed.barrier()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def collect_predictions(classification, original, args):

    original = original.drop(columns=[c for c in original.columns if c in ['label', 'pWood', 'pLeaf']])

    neighbours = NearestNeighbors(n_neighbors=9, algorithm="kd_tree", metric="euclidean", radius=0.025).fit(classification[:, :3])
    _, indices = neighbours.kneighbors(original[:, :3])

    # Bool to classify point as wood if any prediction above a certain threshold
    is_wood = np.any(classification[indices][:, :, -2] > args.is_wood, axis=1)

    #NOTE some of this needs to change to reflect shift from softmax to sigmoid
    labels = np.zeros((original.shape[0], 3))
    labels[:, :2] = np.median(classification[indices][:, :, -2:], axis=1)
    labels[:, 2] = np.argmax(labels[:, :2], axis=1)

    original.loc[original.index, ['pWood','pLeaf', 'label']] = labels[:, :]
    original.loc[is_wood, 'label'] = 1

    return original

#########################################################################################################
#                                       SEMANTIC INFERENCE FUNCTION                                     #
#                                       ==========================                                      #

def SemanticSegmentation(gpu,args):

    '''
    Setup Multi GPU processing. 
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.verbose: print('Using:', args.gpus, "GPUs with", device)
    
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(backend='nccl',init_method='env://',world_size=args.world_size,rank=rank)  

    '''
    Setup model. 
    '''

    model = Net(num_classes=2).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if os.path.isfile(os.path.join(args.wdir,'model',args.model)):
        load_checkpoint(os.path.join(args.wdir,'model',args.model), rank, model)
    else:
        raise Exception(f'No model loaded at {os.path.join(args.wdir,"model",args.model)}')

    #####################################################################################################
    
    '''
    Setup data loader. 
    '''

    test_dataset = TestingDataset(root_dir="FILL THIS IN",
                                    device = rank, min_pts=args.min_pts,
                                    max_pts=args.max_pts, augmentation=False)

    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=rank)
                                    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=0,
                              pin_memory=False, sampler=test_sampler)

    #####################################################################################################

    '''
    Initialise model
    '''

    model.eval()

    with torch.no_grad():
        output_list = []

        for data in tqdm(test_loader, disable=False if args.verbose else True, colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            
            data = data.to(rank)
            
            with torch.cuda.amp.autocast():
                outputs = model(data)
            
            outputs = torch.sigmoid(outputs).cpu().detach()

            batches = np.unique(data.batch.cpu())
            pos = data.pos.cpu()
            output = np.hstack((pos, out))

            for batch in batches:
                outputb = np.asarray(output[data.batch.cpu() == batch])
                outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch:3 + (3 * batch)]
                output_list.append(outputb)

        classified_pc = np.vstack(output_list)

    del outputb, out, batches, pos, output

    #####################################################################################################
    
    '''
    Choosing most confident labels using nearest neighbour search. 
    '''  

    if args.verbose: print("Collecting prediction probabilites and labels...")

    args.pc = collect_predictions(classified_pc, args.pc, args)

    ##Add ground points and labeling from prior CSF classifcation
    # args.grd.loc[:, 'label'] = args.terrain_class
    # args.grd.loc[:, ['pWood','pLeaf']] = 0
    # args.pc = args.pc.append(args.grd, ignore_index=True)

    # #calculate dtm
    # args.= make_dtm(args.
    # args.pc.loc[args.pc.n_z <= args.ground_height_threshold, 'label'] = args.terrain_class

    save_file(os.path.join(args.odir, args.basename + '.segmented.' + args.output_fmt), 
              args.pc, additional_fields=['label', 'pWood', 'pLeaf'] + args.headers)

    if not args.keep_npy: [os.unlink(f) for f in test_dataset.filenames]
    
    return args

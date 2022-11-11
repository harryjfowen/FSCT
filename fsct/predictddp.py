import sys
import os
import argparse
import random
import shutil

from src.parameters import predict_parameters
from src.preprocessingV2 import pcd2voxels
from src.predicter import SemanticSegmentation
from src.tools import load_file

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def get_fsct_path(location_in_fsct=""):

    current_wdir = os.getcwd()
    
    output_path = current_wdir[: current_wdir.index("fsct") + 4]
    
    if len(location_in_fsct) > 0:
        output_path = os.path.join(output_path, location_in_fsct)
    
    return output_path.replace("\\", "/")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default='', type=str, help='path to point cloud')
    parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    
    # Set these appropriately for your hardware.
    parser.add_argument('--batch_size', default=10, type=int, help="If you get CUDA errors, try lowering this.")
    parser.add_argument('--num_procs', default=10, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")

    parser.add_argument('--model', type=str, default='model.pth', help='path to candidate model')
    parser.add_argument('--is-wood', default=0.90, type=float, help='a probability above which points are classified as wood')

    parser.add_argument('--keep_npy', action='store_true', help="Keeps .npy files used for segmentation after inference is finished.")
                           
    parser.add_argument('--output_fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    args = parser.parse_args()

    if args.verbose:
        print('\n---- parameters used ----')
        for k, v in args.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    args.wdir = get_fsct_path()
    args.min_pts=1000
    args.max_pts=20000

    '''
    Establish GPU distribution parameters. 
    '''

    args.world_size = args.gpus * args.nodes    
    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = str(random.randint(49152,65535)) 

    '''
    Sanity check
    '''
    if args.point_cloud == '':
        raise Exception('no input specified, please specify --point-cloud')
    
    if not os.path.isfile(args.point_cloud):
            raise Exception(f'no point cloud at {args.point_cloud}')
    
    '''
    Denoise and split cloud into voxels. 
    '''

    pcd2voxels(args)

    '''
    Run semantic training. 
    '''
    
    args.pc, args.headers = load_file(filename='/home/harryowen/Desktop/small_train_set/raw.ply', additional_headers=True, verbose=True)
    args.odir = '/home/harryowen/Desktop/small_train_set/'
    args.basename = 'raw'
    
    mp.spawn(SemanticSegmentation, nprocs=args.gpus, args=(args,)) 

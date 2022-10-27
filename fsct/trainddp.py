from src.trainerddp import SemanticTraining
from src.tools import get_fsct_path
import argparse, glob, os
import torch
import numpy as np

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--augmentation', action='store_true', help="Perform data augmentation")
        parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")

        parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
        parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
        parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
        parser.add_argument('--num_epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--checkpoint_saves', default=1, type=int, metavar='N', help='number of times to save model')

        parser.add_argument('--model', type=str, default='model.pth', help='Name of global model [e.g. model.pth')
        parser.add_argument('--best_model', action='store_true', default='false', help="Save best model across epochs rather than final model")

        parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--batch_size', type=int, default=2, help='Batch size for cuda processing [Lower less memory usage]')

        parser.add_argument('--evaluate', action='store_true', default='false', help="Perform evaluation")
        parser.add_argument('--verbose', action='store_true', help="print stuff")

        args = parser.parse_args()
        
        args.wdir = get_fsct_path()
        args.min_pts=1000
        args.max_pts=20000
        
        #For now switch of evaluation 
        args.evaluate = False
        args.learning_rate = 0.000025
        args.checkpoints = np.arange(0, args.num_epochs+1, int(args.num_epochs / args.checkpoint_saves))
        print('Saving ', len(args.checkpoints)-1, ' checkpoints.')

        #Clear checkpoint directory
        old_checkpoints = glob.glob(os.path.join(args.wdir,'checkpoints/*.pth'))
        for f in old_checkpoints:
                os.remove(f)

        '''
        Establish GPU distribution parameters. 
        '''

        args.world_size = args.gpus * args.nodes    
        os.environ['MASTER_ADDR'] = 'localhost'              
        os.environ['MASTER_PORT'] = '12355'                      
        
        '''
        Sanity checks. 
        '''

        if (len(glob.glob(os.path.join(args.wdir + '/data/train/*/*.npy'))) == 0):
                raise Exception(f'no training data at {os.path.join(args.wdir + "/data/train/*/*.npy")}')

        if (len(glob.glob(os.path.join(args.wdir + '/data/validation/*/*.npy'))) == 0):
                raise Exception(f'no training data at {os.path.join(args.wdir + "/data/validation/*/*.npy")}')

        if len(args.checkpoints) == 0:
                args.checkpoints == np.asarray([args.num_epochs-1])

        '''
        Run semantic training. 
        '''

        mp.spawn(SemanticTraining, nprocs=args.gpus, args=(args,)) 


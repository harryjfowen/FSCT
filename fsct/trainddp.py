from src.trainerddp import SemanticTraining
from src.tools import dict2class
from src.tools import get_fsct_path
import argparse, glob, os
import torch

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--augmentation', action='store_true', help="Perform data augmentation")
        parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")

        parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
        parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
        parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
        parser.add_argument('--num_epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--model', type=str, default='model.pth', help='Name of global model [e.g. model.pth')
        parser.add_argument('--best_model', action='store_true', default='false', help="Save best model across epochs rather than final model")

        parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--batch_size', type=int, default=2, help='Batch size for cuda processing [Lower less memory usage]')

        parser.add_argument('--evaluate', action='store_true', default='false', help="Perform evaluation")
        parser.add_argument('--verbose', action='store_true', help="print stuff")

        args = parser.parse_args()
        
        #Number of GPU's per node
        args.gpus=2#list(range(torch.cuda.device_count()))
        args.min_pts=1000
        args.max_pts=20000
        
        #For now switch of evaluation 
        args.evaluate = False
        args.learning_rate = 0.000025

        args.world_size = args.gpus * args.nodes    

        os.environ['MASTER_ADDR'] = 'localhost'              
        os.environ['MASTER_PORT'] = '12355'                      

        args.mode = os.path.splitext(os.path.basename(__file__))[0]
        args.wdir = get_fsct_path()
        
        #if (args.preprocess_datasets == False) & (len(glob.glob(os.path.join(args.wdir + '/data/*/*/*.npy'))) == 0):
        if (len(glob.glob(os.path.join(args.wdir + '/data/*/*/*.npy'))) == 0):
                args.preprocess_datasets = True
                print("No preprocessed data found. Switching preprocessing to 'True'")
        else:
                print("Found training data")

        '''
        Run semantic training. 
        '''

        # Run semantic training across multiple gpus and nodes 
        mp.spawn(SemanticTraining, nprocs=args.gpus, args=(args,)) 

        #torch.multiprocessing.set_start_method('spawn', force=True)
        #SemanticTraining(params)


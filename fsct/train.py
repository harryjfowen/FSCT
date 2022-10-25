from src.parameters import train_parameters
from src.trainer import SemanticTraining
from src.tools import dict2class
from src.tools import get_fsct_path
import argparse, glob, os
import torch

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--validation', action='store_true', help="Perform validaiton alongside training")
        parser.add_argument('--augmentation', action='store_true', help="Perform data augmentation")
        parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")

        parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
        parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
        parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
        parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--model', type=str, default='model.pth', help='Name of candidate model [e.g. model.pth')
        
        parser.add_argument('--device', type=str, default='cpu', help='Insert either "cuda" or "cpu"')
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size for cuda processing [Lower less memory usage]')

        parser.add_argument('--verbose', action='store_true', help="print stuff")

        params = parser.parse_args()

        for k, v in train_parameters.items():
                setattr(params, k, v)

        params.mode = os.path.splitext(os.path.basename(__file__))[0]
        params.wdir = get_fsct_path()
        
        #if (params.preprocess_datasets == False) & (len(glob.glob(os.path.join(params.wdir + '/data/*/*/*.npy'))) == 0):
        if (len(glob.glob(os.path.join(params.wdir + '/data/*/*/*.npy'))) == 0):
                params.preprocess_datasets = True
                print("No preprocessed data found. Switching preprocessing to 'True'")

        '''
        Run semantic training. 
        '''
        # Run semantic training 
        torch.multiprocessing.set_start_method('spawn', force=True)
        SemanticTraining(params)


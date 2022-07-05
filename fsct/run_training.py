from training_parameters import other_parameters
from preprocessing import Preprocessing
from training_functions import SemanticTraining
import argparse, glob, os
import tools
from tools import dict2class
import torch

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--preprocess_datasets', action='store_true', help="Preprocess Data")
        parser.add_argument('--augmentation', action='store_true', help="Perform Data Augmentation")
        parser.add_argument('--informed_downsampling', action='store_false', help="Weight random downsampling by coarse leaf/wood classifier")
        parser.add_argument('--validation', action='store_true', help="Perform validaiton alongside training")

        parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")

        parser.add_argument('--model_name', type=str, default='model.pth', help='Name of NN model [.pth]')
        parser.add_argument('--num_epochs', type=int, default=100, help='How many epochs to train')

        parser.add_argument('--device', type=str, default='cpu', help='insert either "cuda" or "cpu"')
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size for cuda processing [Lower less memory usage]')

        parser.add_argument('--verbose', action='store_true', help="print stuff")

        params = parser.parse_args()

        for k, v in other_parameters.items():
                setattr(params, k, v)

        params.wdir = tools.get_fsct_path()
        params.point_cloud = glob.glob(os.path.join(params.wdir + '/data/*/*.ply'))

        # Sanity checks
        if len(params.point_cloud) == 0:
                print("No data found!")
        
        if (params.preprocess_datasets == False) & (len(glob.glob(os.path.join(params.wdir + '/data/*/*/*.npy'))) == 0):
                params.preprocess_datasets = True
                print("No preprocessed data found. Switching preprocessing to 'True'")
        # End of sanity checks

        # Preprocess data blocks
        if params.preprocess_datasets:
                params.clean_wdir = True
                Preprocessing(params)

        # Run semantic training 
        torch.multiprocessing.set_start_method('spawn', force=True)
        SemanticTraining(params)
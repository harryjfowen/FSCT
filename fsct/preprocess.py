from src.train_params import other_parameters
from src.preprocessing import Preprocessing
from src.tools import dict2class
from src.tools import get_fsct_path
import argparse, glob, os

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--verbose', action='store_true', help="print stuff")

        params = parser.parse_args()

        for k, v in other_parameters.items():
                setattr(params, k, v)

        params.wdir = get_fsct_path()
        params.point_cloud = glob.glob(os.path.join(params.wdir + '/data/*/*.ply'))

        '''
        Minor sanity checks. 
        '''
        
        if len(params.point_cloud) == 0:
                print("No data found!")
        
        if (params.preprocess_datasets == False) & (len(glob.glob(os.path.join(params.wdir + '/data/*/*/*.npy'))) == 0):
                params.preprocess_datasets = True
                print("No preprocessed data found. Switching preprocessing to 'True'")

        '''
        Preprocess clouds into voxels. 
        '''
        
        Preprocessing(params)

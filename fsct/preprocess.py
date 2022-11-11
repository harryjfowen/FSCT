from src.preprocessing import Preprocessing
from src.tools import dict2class
from src.tools import get_fsct_path
import argparse, glob, os
from src.parameters import train_parameters

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--point_spacing', type=float, default=0.01, help="Downsampling resolution [set to 0 if no downsampling")
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--verbose', action='store_true', help="print stuff")

        params = parser.parse_args()
        params.mode='train'
        params.clean_wdir = True

        for k, v in train_parameters.items():
                setattr(params, k, v)

        params.wdir = get_fsct_path()
        params.point_cloud = glob.glob(os.path.join(params.wdir + '/data/*/*.ply'))

        '''
        Preprocess clouds into voxels. 
        '''
        
        Preprocessing(params)

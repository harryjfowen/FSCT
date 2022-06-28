from tools import load_file, save_file, get_fsct_path
from model import Net
from fsct_exceptions import NoDataFound
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import glob
import random
import threading
import os
import shutil
from training_parameters import main_parameters

from training import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    params = parser.parse_args()
    
    for k, v in main_parameters.items():
            setattr(params, k, v)

    torch.multiprocessing.set_start_method('spawn', force=True)
    SemanticTraining(params)

from src.tools import load_file, save_file, get_fsct_path
from src.augmentation import augmentations
from src.model import Net

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset, DataLoader, Data
import glob
import random
import threading
import os
import shutil
from abc import ABC

from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_geometric.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler

class TrainingDataset(Dataset, ABC):
    
    '''
    Create training dataset function for torch data loader. 
    '''
    def __init__(self, root_dir, device, min_pts, max_pts, augmentation):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.max_pts = max_pts
        self.label_index = 3
        self.device = device
        self.min_pts = min_pts
        self.augmentation = augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])

        x = point_cloud[:, :3]
        y = point_cloud[:, self.label_index]#-1 #
        
        if self.augmentation:
            x, y = augmentations(x, y, self.min_pts)
        
        x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

        # Place sample at origin
        global_shift = torch.mean(x[:, :3], axis=0)
        x = x - global_shift

        data = Data(pos=x, x=None, y=y)
        return data


def load_model(file, rank, model, optimizer):

    torch.distributed.barrier()

    map_location = {'cuda:0': f'cuda:{rank}'}

    checkpoint = torch.load(file, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def save_model(path,epoch,model_state,optimizer_state):

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state
        },
        path)


#########################################################################################################
#                                       SEMANTIC TRAINING FUNCTION                                      #
#                                       ==========================                                      #

def SemanticTraining(gpu,args):

    '''
    Setup Multi GPU processing. 
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.verbose: print('Using:', gpus, "GPUs with", device)
    
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(backend='nccl',init_method='env://',world_size=args.world_size,rank=rank)  

    #####################################################################################################

    '''
    Setup model. 
    '''
    
    #Change to sync batch norm for across gpu enhanced performance 
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Net(num_classes=2)).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    #####################################################################################################
    
    '''
    Setup data loaders. 
    '''

    train_dataset = TrainingDataset(root_dir=os.path.join(args.wdir, "data", "train/sample_dir/"),
                                    device = rank, min_pts=args.min_pts,
                                    max_pts=args.max_pts, augmentation=args.augmentation)

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
                                    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=0,
                              pin_memory=False, sampler=train_sampler)


    #####################################################################################################

    '''
    Find existing model or create new model file. 
    '''

    args.model_filepath = os.path.join(args.wdir,'model',args.model)
    if os.path.isfile(args.model_filepath):
        print("")
    else:
        print("\nModel not found, creating new file...")
        torch.save(model.state_dict(),args.model_filepath)

    
    #####################################################################################################
    
    '''
    Train model. 
    '''

    it = 0
    for epoch in range(args.num_epochs):

        model.train()
        running_loss = 0.0
        running_acc = 0
        epoch_loss = []
        epoch_acc = []

        if rank == 0:
            print("=====================================================================")
            print("EPOCH ", epoch)

        if epoch > 1:
            model, optimizer = load_model(args.model_filepath, rank, model, optimizer)

        for i, data in enumerate(train_loader):
            data.pos = data.pos.to(rank)
            data.y = torch.unsqueeze(data.y, 0).to(rank)
            outputs = model(data)
            loss = criterion(outputs, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.detach().item()
            running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]

            if i % 20 == 0 and rank == 0:
                print("Train sample accuracy: ",
                      np.around(running_acc / (i + 1), 4),
                      ", Loss: ",
                      np.around(running_loss / (i + 1), 4))
        
        epoch_loss.append(running_loss / len(train_loader))
        epoch_acc.append(running_acc / len(train_loader))
        
        print("\nTrain EPOCH accuracy: ", np.around(epoch_acc[it], 4), ", Loss: ", np.around(epoch_loss[it], 4))

        #Save model
        if rank == 0:
            if args.best_model and epoch_loss[it] <= min(epoch_loss):
                print("Best model saved...\n")
                save_model(args.model_filepath, epoch, model.state_dict(), optimizer.state_dict())
            else:
                save_model(args.model_filepath, epoch, model.state_dict(), optimizer.state_dict())
        
        it+=1
        dist.barrier()
    
    #Cleanup processes
    dist.destroy_process_group()
        

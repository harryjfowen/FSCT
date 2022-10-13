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
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from abc import ABC
from sklearn.metrics import f1_score
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


class ValidationDataset(Dataset, ABC):

    '''
    Create validation dataset function for torch data loader. 
    '''
    def __init__(self, root_dir, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.label_index = 3
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with torch.no_grad():
            point_cloud = np.load(self.filenames[index])
            x = point_cloud[:, :3]
            y = point_cloud[:, self.label_index]# - 1
            x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
            y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

            # Place sample at origin
            global_shift = torch.mean(x[:, :3], axis=0)
            x = x - global_shift

            data = Data(pos=x, x=None, y=y)
            return data

def update_log(training_history):
        try:
            np.savetxt(os.path.join(get_fsct_path("model"), "training_history.csv"), training_history)
        except PermissionError:
            print("training_history not saved this epoch, please close training_history.csv to enable saving.")
            try:
                np.savetxt(
                    os.path.join(get_fsct_path("model"), "training_history_permission_error_backup.csv"),
                    training_history,
                )
            except PermissionError:
                pass
            
#                                       SEMANTIC TRAINING FUNCTION                                      #
#               ============================================================================            #

def SemanticTraining(params):
     
    '''
    Setup data loaders. 
    '''

    train_dataset = TrainingDataset(root_dir=os.path.join(params.wdir, "data", "train/sample_dir/"),
                                    device=params.device, min_pts=params.min_pts,
                                    max_pts=params.max_pts, augmentation=params.augmentation)
                                    
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                              shuffle=True, drop_last=True)

    if params.validation:
        validation_dataset = ValidationDataset(root_dir=os.path.join(get_fsct_path("data"),
                                               "validation/sample_dir/"),
                                               device=params.device)

        validation_loader = DataLoader(validation_dataset,
                                       batch_size=params.batch_size,
                                       shuffle=True, drop_last=True)

    '''
    Detect whether cuda and gpu's are available.
    '''
    
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.verbose: print('Using:', params.device)

    #model = Net(num_classes=2).to(params.device)
    model = nn.DataParallel(Net(num_classes=2)).to(params.device)
    
    params.model_filepath = os.path.join(params.wdir,'model',params.model)
    if os.path.isfile(params.model_filepath):
        print("\nLoading ", params.model_filepath)
        
        model.load_state_dict(torch.load(params.model_filepath,
                              map_location=params.device),
                              strict=False)
        try:
            training_history = np.loadtxt(os.path.join(params.wdir, 'model', "training_history.csv"))
            print("Loaded training history successfully.")
        except OSError:
            training_history = np.empty([0])
            pass
    else:
        print("\nModel not found, creating new file...")
        torch.save(model.state_dict(),params.model_filepath)
        training_history = np.empty([0])

    #   Initialise model on single or multiple GPU's
    ##  To specify a specific GPU "params.device = torch.device(‘cuda:0’)" for GPU number 0
    #model = model.to(params.device)
    model = nn.DataParallel(model).to(params.device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()
    val_epoch_loss = 0
    val_epoch_acc = 0
    
    '''
    Train model. 
    '''
    for epoch in range(params.num_epochs):
        print("=====================================================================")
        print("EPOCH ", epoch)

        model.train()
        running_loss = 0.0
        running_acc = 0
        running_f1 = 0.0
        i = 0
        for data in train_loader:
            data.pos = data.pos.to(params.device)
            data.y = torch.unsqueeze(data.y, 0).to(params.device)
            outputs = model(data)
            loss = criterion(outputs, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.detach().item()
            running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
            running_f1 += f1_score(data.y.cpu().data, preds.cpu(),average='micro')

            if i % 20 == 0:
                print("Train sample accuracy: ",
                      np.around(running_acc / (i + 1), 4),
                      ", Loss: ",
                      np.around(running_loss / (i + 1), 4),
                      ", F1: ",
                      np.around(running_f1 / (i + 1), 4)) 
            i += 1
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        epoch_f1 = running_f1 / len(train_loader)
                
        val_epoch_loss = 0
        val_epoch_acc = 0
        val_epoch_f1 = 0	
        
        epoch_results = np.array([[epoch, epoch_loss, epoch_acc, epoch_f1, val_epoch_loss, val_epoch_acc, val_epoch_f1]])

        if params.validation == False:
            if len(training_history) == 0:
                update_log(epoch_results)
                training_history = epoch_results
            else:
                training_history = np.vstack((training_history,epoch_results))
                update_log(training_history)
        
        print("\nTrain EPOCH accuracy: ", np.around(epoch_acc, 4), ", Loss: ", np.around(epoch_loss, 4), ", F1: ",np.around(epoch_f1, 4), "\n")

        if params.validation:
            model.eval()
            running_loss = 0.0
            running_acc = 0
            running_f1 = 0.0
            i = 0
            for data in validation_loader:
                data.pos = data.pos.to(params.device)
                data.y = torch.unsqueeze(data.y, 0).to(params.device)

                outputs = model(data)
                loss = criterion(outputs, data.y)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                running_f1 += f1_score(data.y.cpu().data, preds.cpu(),average='micro')
                
                if i % 50 == 0:
                    print("Validation sample accuracy: ",
                      np.around(running_acc / (i + 1), 4),
                      ", Loss: ",
                      np.around(running_loss / (i + 1), 4),
                      ", F1: ",
                      np.around(running_f1 / (i + 1), 4)) 
                i += 1
                
            val_epoch_loss = running_loss / len(validation_loader)
            val_epoch_acc = running_acc / len(validation_loader)
            val_epoch_f1 = running_f1 / len(validation_loader)
            
            epoch_results = np.array([[epoch, epoch_loss, epoch_acc, epoch_f1, val_epoch_loss, val_epoch_acc, val_epoch_f1]])

            if  len(training_history) == 0:
                update_log(epoch_results)
                training_history = epoch_results
            else:
                training_history = np.vstack((training_history,epoch_results))
                update_log(training_history)
            
            print("\nValidation EPOCH accuracy: ", np.around(val_epoch_acc, 4), ", Loss: ", np.around(val_epoch_loss, 4), ", F1: ",np.around(val_epoch_f1, 4), "\n")
            print("=====================================================================")

        torch.save(model.state_dict(),params.model_filepath)

from tools import load_file, save_file, get_fsct_path
from model import Net
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
from data_augmentation import augmentations
from jakteristics import compute_features
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from abc import ABC

class TrainingDataset(Dataset, ABC):
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
        y = point_cloud[:, self.label_index] - 1
        
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
            y = point_cloud[:, self.label_index] - 1
            x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
            y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

            # Place sample at origin
            global_shift = torch.mean(x[:, :3], axis=0)
            x = x - global_shift

            data = Data(pos=x, x=None, y=y)
            return data


#                                       SEMANTIC TRAINING FUNCTION                                      #
#               ============================================================================            #

def SemanticTraining(params):
    
    # check status of GPU
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.verbose: print('\nUsing:', params.device)

#                                        set up dataloaders                                              #
    
    training_history = np.zeros((0, 5))

    train_dataset = TrainingDataset(root_dir=os.path.join(params.wdir, "data", "train/sample_dir/"),
                                    device=params.device,
                                    min_pts=params.min_pts,
                                    max_pts=params.max_pts,
                                    augmentation=params.augmentation)
                                    
    train_loader = DataLoader(train_dataset,
                              batch_size=params.batch_size,
                              shuffle=True,
                              drop_last=True)

    if params.validation:
        validation_dataset = ValidationDataset(root_dir=os.path.join(get_fsct_path("data"), "validation/sample_dir/"),
                                               device=params.device)

        validation_loader = DataLoader(validation_dataset,
                                       batch_size=params.batch_size,
                                       shuffle=True,
                                       drop_last=True)


#                                        load training file                                              #

    model = Net(num_classes=2).to(params.device)
    
    model_filepath = os.path.join(params.wdir,'model',params.model_name)
    if os.path.isdir(model_filepath):
        print("Loading existing model...")
        
        model.load_state_dict(torch.load(model_filepath,
                              map_location=params.device),
                              strict=False)
        try:
            training_history = np.loadtxt(os.path.join(params.wdir, 'model', "training_history.csv"))
            print("Loaded training history successfully.")
        except OSError:
            pass
    else:
        print("File not found, creating new model...")
        torch.save(model.state_dict(),model_filepath)

    model = model.to(params.device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()
    val_epoch_loss = 0
    val_epoch_acc = 0
    
#                                       train semantic model                                             #
    
    for epoch in range(params.num_epochs):
        print("\n=====================================================================")
        print("EPOCH ", epoch)

        model.train()
        running_loss = 0.0
        running_acc = 0
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

            if i % 20 == 0:
                print("Train sample accuracy: ",
                      np.around(running_acc / (i + 1), 4),
                      ", Loss: ",
                      np.around(running_loss / (i + 1), 4)) 
            i += 1
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
        print("Train epoch accuracy: ", np.around(epoch_acc, 4), ", Loss: ", np.around(epoch_loss, 4), "\n")

        if params.validation:
            print("Validation")

            model.eval()
            running_loss = 0.0
            running_acc = 0
            i = 0
            for data in validation_loader:
                data.pos = data.pos.to(params.device)
                data.y = torch.unsqueeze(data.y, 0).to(params.device)

                outputs = model(data)
                loss = criterion(outputs, data.y)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                if i % 50 == 0:
                    print("Validation sample accuracy: ",
                          np.around(running_acc / (i + 1), 4),
                          ", Loss: ",np.around(running_loss / (i + 1), 4))
                i += 1
            val_epoch_loss = running_loss / len(validation_loader)
            val_epoch_acc = running_acc / len(validation_loader)
            update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
            print("Validation epoch accuracy: ", np.around(val_epoch_acc, 4), ", Loss: ", np.around(val_epoch_loss, 4))
            print("=====================================================================")

        torch.save(model.state_dict(),os.path.join(get_fsct_path("model"), params.model_filename))

from src.augmentation import augmentations
from src.model import Net
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset, DataLoader, Data
import glob
import os
from abc import ABC

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from time import sleep
from sklearn.metrics import f1_score as F1
from src.poly_focal_loss import Poly1FocalLoss

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


def load_model(path, epoch, rank, model, optimizer):

    file = path / f'epoch_{epoch}.pth'

    torch.distributed.barrier()

    map_location = {'cuda:0': f'cuda:{rank}'}

    checkpoint = torch.load(file, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def save_checkpoints(args, epoch, model_state, optimizer_state):

    checkpoint_folder = os.path.join(args.wdir,'checkpoints')
    
    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    
    file = checkpoint_folder + '/'f'epoch_{epoch}.pth'

    torch.save({'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state},file)
    
    return True


#########################################################################################################
#                                       SEMANTIC TRAINING FUNCTION                                      #
#                                       ==========================                                      #

def SemanticTraining(gpu,args):

    '''
    Setup Multi GPU processing. 
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.verbose: print('Using:', args.gpus, "GPUs with", device)
    
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(backend='nccl',init_method='env://',world_size=args.world_size,rank=rank)  

    #####################################################################################################

    '''
    Setup model. 
    '''
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Net(num_classes=2)).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    #criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()#
    W = torch.tensor([10., 1.]).reshape([1, 2, 1]).to(rank)
    criterion = Poly1FocalLoss(num_classes=2,reduction='mean',label_is_onehot=False,pos_weight=W,gamma=2)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=10, epochs=args.num_epochs,anneal_strategy='linear')

    #####################################################################################################
    
    '''
    Setup data loaders. 
    '''

    train_dataset = TrainingDataset(root_dir=args.trdir,
                                    device = rank, min_pts=args.min_pts,
                                    max_pts=args.max_pts, augmentation=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
                                    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=0,
                              pin_memory=False, sampler=train_sampler)

    '''
    And for validation...
    '''

    val_dataset = TrainingDataset(root_dir=args.vadir,
                                    device = rank, min_pts=args.min_pts,
                                    max_pts=args.max_pts, augmentation=False)

    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)
                                    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, drop_last=True, num_workers=0,
                            pin_memory=False, sampler=val_sampler)

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

    '''
    Create function to log training history 
    '''

    def log_history(args,history):
        try:
            training_history = np.savetxt(os.path.join(args.wdir, 'model', os.path.splitext(args.model)[0] + "_history.csv"), history)
            print("Saved training history successfully.")

        except OSError:
            training_history = np.savetxt(os.path.join(args.wdir, 'model', os.path.splitext(args.model)[0] + "_history_backup.csv"), history)
            pass

    #####################################################################################################
    
    '''
    Train and Validate model. 
    '''

    for epoch in range(1,args.num_epochs+1):

        '''
        Train model on all GPU's. 
        '''

        model.train()
        train_running_loss = 0.0
        train_running_acc = 0
        running_wood_acc = 0

        if rank == 0:
            sleep(0.1)
            print("\n=============================================================================================")
            print("EPOCH ", epoch)

        it_batch = args.nodes * args.gpus * args.batch_size
        total_length = len(train_loader)*it_batch    
        with tqdm(total=int(total_length), colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:

            for i, data in enumerate(train_loader):

                data.pos = data.pos.to(rank)
                data.y = torch.unsqueeze(data.y, 0).to(rank)
                outputs = model(data.to(rank))

                loss = criterion(outputs, data.y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)

                train_running_loss += loss.detach().item()
                train_running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                running_wood_acc += F1(data.y.flatten().cpu(), preds.flatten().cpu(), average='binary', pos_label=0, zero_division=0)

                if rank == 0:
                    #tepoch.set_description(f"Train Epoch {epoch}")
                    tepoch.set_description(f"Train")
                    tepoch.update(it_batch)
                    tepoch.set_postfix(acc=np.around(train_running_acc / (i + 1), 5),
                                       loss=np.around(train_running_loss / (i + 1), 5),
                                       wood_acc=np.around(running_wood_acc / (i + 1), 5))
                    sleep(0.1)

            tepoch.close()
            dist.barrier()

        '''
        Validate model only on single GPU for now. 
        '''

        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0
        val_running_wood_acc = 0
        sleep(0.1)

        total_length = len(val_loader)*it_batch	   
        with tqdm(total=int(total_length), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:

            for j, data in enumerate(val_loader):
                data.pos = data.pos.to(rank)
                data.y = torch.unsqueeze(data.y, 0).to(rank)
                outputs = model(data.to(rank))
                
                loss = criterion(outputs, data.y)
                
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.detach().item()
                val_running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                val_running_wood_acc += F1(data.y.flatten().cpu(), preds.flatten().cpu(), average='binary', pos_label=0, zero_division=0)
                
                if rank == 0:
                    #tepoch.set_description(f"Eval Epoch {epoch}")
                    tepoch.set_description(f"Eval ")
                    tepoch.update(it_batch)
                    tepoch.set_postfix(acc=np.around(val_running_acc / (j + 1), 5),
                                       loss=np.around(val_running_loss / (j + 1), 5),
                                       wood_acc=np.around(val_running_wood_acc / (j + 1), 5))
                    sleep(0.1)

        tepoch.close()
        dist.barrier()

        '''
        Save model checkpoints and final epoch model 
        '''

        if rank == 0:
            train_epoch_loss = train_running_loss / len(train_loader)
            train_epoch_acc = train_running_acc / len(train_loader)

            val_epoch_loss = val_running_loss / len(val_loader)
            val_epoch_acc = val_running_acc / len(val_loader)

            epoch_results = np.array([[epoch, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc]])

            print("\nTrain EPOCH Acc: ", np.around(train_epoch_acc, 4), ", Loss: ", np.around(train_epoch_loss, 4),
                    ", Val Acc: ", np.around(val_epoch_acc, 4),  ", Val Loss: ", np.around(val_epoch_loss, 4))
            
            if epoch == 1:
                history = epoch_results
            else:
                history = np.vstack((history,epoch_results))

            log_history(args,history)

            if epoch in args.checkpoints:
                save_checkpoints(args, epoch, model.state_dict(), optimizer.state_dict())
            
            if epoch == args.num_epochs:
                print("Saving final GLOBAL model")
                torch.save(model.state_dict(), args.model_filepath)
    
    #Cleanup processes
    dist.destroy_process_group()
        

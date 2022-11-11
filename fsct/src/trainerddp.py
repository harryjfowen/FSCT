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
import torch.nn.functional as F

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


def load_checkpoint(path, rank, model, optimizer):

    #file = path / f'epoch_{epoch}.pth'

    torch.distributed.barrier()

    map_location = {'cuda:0': f'cuda:{rank}'}

    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

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
    model = Net(num_classes=2).to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    W = torch.tensor([1., 10.]).to(rank)
    criterion = Poly1FocalLoss(num_classes=2,label_is_onehot=False,reduction='mean',gamma=5, pos_weight=W)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.001,step_size_up=(args.num_epochs/10),mode="triangular2",cycle_momentum=False)

    if os.path.isfile(os.path.join(args.wdir,'model',args.model)):
        print("Loading model")
        try:
            load_checkpoint(os.path.join(args.wdir,'model',args.model), rank, model, optimizer)
        except KeyError:
            print("Failed to load, creating new...")
            torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))
    else:
        print("\nModel not found, creating new file...")
        torch.save(os.path.join(args.wdir,'model',args.model))

    
    #####################################################################################################
    
    '''
    Setup data loaders. 
    '''

    train_dataset = TrainingDataset(root_dir=args.trdir,
                                    device = rank, min_pts=args.min_pts,
                                    max_pts=args.max_pts, augmentation=False)

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
        train_running_F1 = 0
        running_leaf_acc = 0
        running_wood_acc = 0

        if rank == 0:
            sleep(0.1)
            print("\n=============================================================================================")
            print("EPOCH ", epoch)

        it_batch = args.nodes * args.gpus * args.batch_size
        total_length = len(train_loader)*it_batch

        scaler = torch.cuda.amp.GradScaler()

        with tqdm(total=int(total_length), colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            
            for i, data in enumerate(train_loader):

                data.pos = data.pos.to(rank)
                data.y = data.y.to(rank)
                #data.y.hot = F.one_hot(data.y.to(rank), num_classes=2).float()

                with torch.cuda.amp.autocast(): 
                    outputs = model(data.to(rank))
                    loss = criterion(outputs, data.y)#hot if not using poly focal loss

                optimizer.zero_grad()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                
                # CALC ACCURACY METRICS
                outputs = torch.sigmoid(outputs)
                probs, preds = torch.max(outputs, 1) 

                train_running_loss += loss.detach().item()
                train_running_F1 += F1(data.y.cpu(), preds.cpu(), average='binary', zero_division=0)

                correct_preds = preds[torch.where(preds == data.y)]
                running_wood_acc += (correct_preds == 1.).cpu().sum().item() / (data.y.cpu() == 1.).sum().item()
                running_leaf_acc += (correct_preds == 0.).cpu().sum().item() / (data.y.cpu() == 0.).sum().item()

                if rank == 0:
                    tepoch.set_description(f"Train")
                    tepoch.update(it_batch)
                    tepoch.set_postfix(epochLR=optimizer.param_groups[0]["lr"],
                                       F1=np.around(train_running_F1 / (i + 1), 5),
                                       LOSS=np.around(train_running_loss / (i + 1), 5),
                                       accWOOD=np.around(running_wood_acc / (i + 1), 5),
                                       accLEAF=np.around(running_leaf_acc / (i + 1), 5))

                    sleep(0.1)

            scheduler.step() 
            tepoch.close()
            dist.barrier()

        '''
        Validate model only on single GPU for now. 
        '''
        if args.validate:

            model.eval()
            val_running_loss = 0.0
            val_running_F1 = 0.0
            val_running_leaf_acc = 0
            val_running_wood_acc = 0
            sleep(0.1)

            total_length = len(val_loader)*it_batch	   
            with tqdm(total=int(total_length), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
                with torch.no_grad():
                    for j, data in enumerate(val_loader):

                        data.pos = data.pos.to(rank)
                        data.y = data.y.to(rank)

                        with torch.cuda.amp.autocast():  
                            outputs = model(data.to(rank))
                            loss = criterion(outputs, data.y)#hot if not using poly focal loss

                        # CALC ACCURACY METRICS
                        outputs = torch.sigmoid(outputs)
                        probs, preds = torch.max(outputs, 1) 

                        val_running_loss += loss.detach().item()
                        val_running_F1 += F1(data.y.cpu(), preds.cpu(), average='binary', zero_division=0)

                        correct_preds = preds[torch.where(preds == data.y)]
                        val_running_wood_acc += (correct_preds == 1.).cpu().sum().item() / (data.y.cpu() == 1.).sum().item()
                        val_running_leaf_acc += (correct_preds == 0.).cpu().sum().item() / (data.y.cpu() == 0.).sum().item()

                        if rank == 0:
                            tepoch.set_description(f"Val  ")
                            tepoch.update(it_batch)
                            tepoch.set_postfix(epochLR=optimizer.param_groups[0]["lr"],
                                            F1=np.around(val_running_F1 / (j + 1), 5),
                                            LOSS=np.around(val_running_loss / (j + 1), 5),
                                            accWOOD=np.around(val_running_wood_acc / (j + 1), 5),
                                            accLEAF=np.around(val_running_leaf_acc / (j + 1), 5))

                            sleep(0.1)

            tepoch.close()
            dist.barrier()
        else:
            val_running_loss = 0
            val_running_F1 = 0

        '''
        Save model checkpoints and final epoch model 
        '''

        if rank == 0:
            train_epoch_loss = train_running_loss / len(train_loader)
            train_epoch_F1 = train_running_F1 / len(train_loader)

            val_epoch_loss = val_running_loss / len(val_loader)
            val_epoch_F1 = val_running_F1 / len(val_loader)

            epoch_results = np.array([[epoch, train_epoch_loss, train_epoch_F1, val_epoch_loss, val_epoch_F1]])

            print("\nTrain EPOCH Acc: ", np.around(train_epoch_F1, 4), ", Loss: ", np.around(train_epoch_loss, 4),
                    ", Val Acc: ", np.around(val_epoch_F1, 4),  ", Val Loss: ", np.around(val_epoch_loss, 4))
            
            if epoch == 1:
                history = epoch_results
            else:
                history = np.vstack((history,epoch_results))

            log_history(args,history)

            if epoch in args.checkpoints:
                save_checkpoints(args, epoch, model.state_dict(), optimizer.state_dict())
            
            #EARLY STOP IF VALIDATION LOSS PROGRESSIVELY INCREASES [MODEL STARTING TO OVERFIT]
            patience = 5
            if args.stop_early:
                if epoch > 0 and history[-1, 3] > history[-2, 3]:
                    consec_increases += 1
                else:
                    consec_increases = 0
            if consec_increases == patience:
                print(f"Stopped early at epoch {epoch + 1} - val loss increased for {consec_increases} consecutive epochs!")
                break

            #SAVE FINAL GLOBAL MODEL AT LAST EPOCH
            if epoch == args.num_epochs:
                print("Saving final GLOBAL model")
                torch.save(model.state_dict(), args.model_filepath)
    
    #Cleanup processes
    dist.destroy_process_group()
        

import torch
from torch_geometric.data import Dataset, DataLoader, Data
import numpy as np
import random
import glob
from sklearn.neighbors import NearestNeighbors
from data_augmentation import augmentations

class TrainingDataset(Dataset):
    def __init__(self, root_dir, device, min_sample_points, max_sample_points, augmentation):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.max_sample_points = max_sample_points
        self.label_index = 3
        self.device = device
        self.min_sample_points = min_sample_points
        self.augmentation = augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])

        x = point_cloud[:, :3]
        y = point_cloud[:, self.label_index] - 1

        if (self.augmentation):
            x, y = data_augmentation(x, y, self.min_sample_points)

        x = torch.from_numpy(x.copy()).type(torch.float).to(self.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

        # Place sample at origin
        global_shift = torch.mean(x[:, :3], axis=0)
        x = x - global_shift

        data = Data(pos=x, x=None, y=y)
        return data


class ValidationDataset(Dataset):
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


def subsample_point_cloud(x, y, min_spacing, min_sample_points):
    x = np.hstack((x, np.atleast_2d(y).T))
    neighbours = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(x[:, :3])
    distances, indices = neighbours.kneighbors(x[:, :3])
    x_keep = x[distances[:, 1] >= min_spacing]
    i1 = [distances[:, 1] < min_spacing][0]
    i2 = [x[indices[:, 0], 2] < x[indices[:, 1], 2]][0]
    x_check = x[np.logical_and(i1, i2)]

    while np.shape(x_check)[0] > 1:
        neighbours = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(x_check[:, :3])
        distances, indices = neighbours.kneighbors(x_check[:, :3])
        x_keep = np.vstack((x_keep, x_check[distances[:, 1] >= min_spacing, :]))
        i1 = [distances[:, 1] < min_spacing][0]
        i2 = [x_check[indices[:, 0], 2] < x_check[indices[:, 1], 2]][0]
        x_check = x_check[np.logical_and(i1, i2)]
    if x_keep.shape[0] >= min_sample_points:
        return x_keep[:, :3], x_keep[:, 3]
    else:
        return x[:, :3], x[:, 3]
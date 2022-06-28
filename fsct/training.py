from tools import load_file, save_file, get_fsct_path
from model import Net
from fsct_exceptions import NoDataFound
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

from jakteristics import compute_features
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors

def check_and_fix_data_directory_structure(self, data_sub_directory):
    """
    Creates the data directory and required subdirectories in
    the FSCT directory if they do not already exist.
    """
    fsct_dir = get_fsct_path()
    dir_list = [
        os.path.join(fsct_dir, "data"),
        os.path.join(fsct_dir, "data", data_sub_directory),
        os.path.join(fsct_dir, "data", data_sub_directory, "sample_dir"),
    ]

    for directory in dir_list:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print(directory, "directory created.")

        elif "sample_dir" in directory and params.clean_sample_directories==1:
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory)
            print(directory, "directory created.")

        else:
            print(directory, "directory found.")


def global_shift_to_origin(self, point_cloud):

    point_cloud_min, point_cloud_max = point_cloud[['x', 'y', 'z']].min(), point_cloud[['x', 'y', 'z']].max()

    point_cloud_centre = (point_cloud_min + ((point_cloud_max - point_cloud_min) / 2)).values

    point_cloud[['x', 'y', 'z']] = point_cloud[['x', 'y', 'z']] - point_cloud_centre

    return point_cloud, point_cloud_centre


def save_pts(params, I, bx, by, bz):

    pc = params.pc.loc[(params.pc.x.between(bx, bx + params.box_dims[0])) &
                      (params.pc.y.between(by, by + params.box_dims[0])) &
                      (params.pc.z.between(bz, bz + params.box_dims[0]))]

    if len(pc) > params.min_points_per_box:

        if len(pc) > params.max_points_per_box:
            pc = pc.sample(n=params.max_points_per_box)

        np.savetxt(os.path.join(params.working_dir, f'{I:07}.txt'), pc[['x', 'y', 'z']].values)


    #needs to take a point cloud and path as input 
def preprocess_point_cloud(params):

    print("Globally shifting point cloud to minimise precision errors during segmentation")
    params.pc, _ = self.global_shift_to_origin(params.pc)

    # classify ground returns using cloth simulation
    params.ground_idx = classify_ground(params)

    print("Classified and dropped ground points...")
    #veg_idx = params.pc[~np.in1d(np.arange(len(params.pc)), ground_idx].index.values

    # remove ground points 
    params.pc = params.pc.drop(params.ground_idx)

    if params.subsample: # subsample if specified
        if params.verbose: print('downsampling to: %s m' % params.subsampling_min_spacing)
        params.pc = downsample(params.pc, params.subsampling_min_spacing, 
                             accurate=False, keep_points=False)
    
    params.pc.reset_index(inplace=True)
    params.pc.loc[:, 'pid'] = params.pc.index

    # generate bounding boxes
    xmin, xmax = np.floor(params.pc.x.min()), np.ceil(params.pc.x.max())
    ymin, ymax = np.floor(params.pc.y.min()), np.ceil(params.pc.y.max())
    zmin, zmax = np.floor(params.pc.z.min()), np.ceil(params.pc.z.max())

    box_overlap = params.box_dims[0] * params.box_overlap[0]

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)

    # multithread segmenting points into boxes and save
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts, args=(params, i, bx, by, bz)))

    for x in tqdm(threads, 
                  desc='generating data blocks',
                  disable=False if params.verbose else True):
        x.start()

    for x in threads:
        x.join()

    if params.verbose: print("Preprocessing done in {} seconds\n".format(time.time() - start_time))
    
    return params

def preprocessing_setup(self, data_subdirectory):
    self.check_and_fix_data_directory_structure(data_subdirectory)
    point_cloud_list = glob.glob(get_fsct_path("data") + "/" + data_subdirectory + "/*.ply")
    if len(point_cloud_list) > 0:
        print("Preprocessing training point clouds...")
        for point_cloud_file in point_cloud_list:
            print(point_cloud_file)
            point_cloud, headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
            self.preprocess_point_cloud(params)


def update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc):
    training_history = np.vstack(
        (training_history, np.array([[epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc]]))
    )
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
            x, y = augmentations(x, y, self.min_sample_points)
        
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


def augmentations(x, y, min_sample_points):
    def rotate_3d(points, rotations):
        rotations[0] = np.radians(rotations[0])
        rotations[1] = np.radians(rotations[1])
        rotations[2] = np.radians(rotations[2])

        roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotations[0]), -np.sin(rotations[0])],
                [0, np.sin(rotations[0]), np.cos(rotations[0])],
            ]
        )

        pitch_mat = np.array(
            [
                [np.cos(rotations[1]), 0, np.sin(rotations[1])],
                [0, 1, 0],
                [-np.sin(rotations[1]), 0, np.cos(rotations[1])],
            ]
        )

        yaw_mat = np.array(
            [
                [np.cos(rotations[2]), -np.sin(rotations[2]), 0],
                [np.sin(rotations[2]), np.cos(rotations[2]), 0],
                [0, 0, 1],
            ]
        )

        points[:, :3] = np.matmul(np.matmul(np.matmul(points[:, :3], roll_mat), pitch_mat), yaw_mat)
        return points

    def random_scale_change(points, min_multiplier, max_multiplier):
        points = points * np.random.uniform(min_multiplier, max_multiplier)
        return points

    def random_point_removal(x, y, min_sample_points):
        indices = np.arange(np.shape(x)[0])
        np.random.shuffle(indices)
        num_points_to_keep = min_sample_points + int(np.random.uniform(0, 0.95) * (np.shape(x)[0] - min_sample_points))
        indices = indices[:num_points_to_keep]
        return x[indices], y[indices]

    def random_noise_addition(points):
        # 50% chance per sample of adding noise.
        random_noise_std_dev = np.random.uniform(0.01, 0.025)
        if np.random.uniform(0, 1) >= 0.5:
            points = points + np.random.normal(0, random_noise_std_dev, size=(np.shape(points)[0], 3))
        return points

    rotations = [np.random.uniform(-90, 90), np.random.uniform(-90, 90), np.random.uniform(-180, 180)]

    x = rotate_3d(x, rotations)
    x = random_scale_change(x, 0.8, 1.2)
    if np.random.uniform(0, 1) >= 0.5 and x.shape[0] > min_sample_points:
        x, y = subsample_point_cloud(x, y, np.random.uniform(0.01, 0.025), min_sample_points)

    return x, y


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

def SemanticTraining(params):

    training_history = np.zeros((0, 5))

    train_dataset = TrainingDataset(
        root_dir=os.path.join(get_fsct_path("data"), "train/sample_dir/"),
        device=params.device,
        min_sample_points=params.min_points_per_box,
        max_sample_points=params.max_points_per_box,
        augmentation=params.perform_data_augmentation,
    )
    if len(train_dataset) == 0:
        raise NoDataFound("No training samples found.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True,
    )

    if params.perform_validation_during_training==1:
        validation_dataset = ValidationDataset(
            root_dir=os.path.join(get_fsct_path("data"), "validation/sample_dir/"),
            device=params.device,
        )

        if len(validation_dataset) == 0:
            raise NoDataFound("No validation samples found.")

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            drop_last=True,
        )

    model = Net(num_classes=2).to(params.device)
    if params.load_existing_model:
        print("Loading existing model...")
        try:
            model.load_state_dict(
                torch.load(os.path.join(get_fsct_path("model"), params.model_filename), map_location=params.device),
                strict=False,
            )

        except FileNotFoundError:
            print("File not found, creating new model...")
            torch.save(
                model.state_dict(),
                os.path.join(get_fsct_path("model"), params.model_filename),
            )

        try:
            training_history = np.loadtxt(os.path.join(get_fsct_path("model"), "training_history.csv"))
            print("Loaded training history successfully.")
        except OSError:
            pass

    model = model.to(params.device)
    print(params.device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()
    val_epoch_loss = 0
    val_epoch_acc = 0
    
    for epoch in range(params.num_epochs):
        print("=====================================================================")
        print("EPOCH ", epoch)
        # TRAINING
        model.train()
        running_loss = 0.0
        running_acc = 0
        i = 0
        running_point_cloud_vis = np.zeros((0, 5))
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
            running_point_cloud_vis = np.vstack(
                (
                    running_point_cloud_vis,
                    np.hstack((data.pos.cpu() + np.array([i * 7, 0, 0]), data.y.cpu().T, preds.cpu().T)),
                )
            )
            if i % 20 == 0:
                print(
                    "Train sample accuracy: ",
                    np.around(running_acc / (i + 1), 4),
                    ", Loss: ",
                    np.around(running_loss / (i + 1), 4),
                )

                if params.generate_point_cloud_vis:
                    save_file(
                        os.path.join(get_fsct_path("data"), "latest_prediction.las"),
                        running_point_cloud_vis,
                        headers_of_interest=["x", "y", "z", "label", "prediction"],
                    )
            i += 1
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
        print("Train epoch accuracy: ", np.around(epoch_acc, 4), ", Loss: ", np.around(epoch_loss, 4), "\n")

        # VALIDATION
        print("Validation")

        if params.perform_validation_during_training==1:
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
                    print(
                        "Validation sample accuracy: ",
                        np.around(running_acc / (i + 1), 4),
                        ", Loss: ",
                        np.around(running_loss / (i + 1), 4),
                    )

                i += 1
            val_epoch_loss = running_loss / len(validation_loader)
            val_epoch_acc = running_acc / len(validation_loader)
            update_log(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)
            print(
                "Validation epoch accuracy: ", np.around(val_epoch_acc, 4), ", Loss: ", np.around(val_epoch_loss, 4)
            )
            print("=====================================================================")
        torch.save(
            model.state_dict(),
            os.path.join(get_fsct_path("model"), params.model_filename),
        )



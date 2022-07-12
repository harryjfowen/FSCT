import numpy as np
import random

def augmentations(x, y, min_sample_points):
    
    '''
    Augment data by rotating and rescaling. 
    '''
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
    #if np.random.uniform(0, 1) >= 0.5 and x.shape[0] > min_sample_points:
        #x, y = subsample_point_cloud(x, y, np.random.uniform(0.01, 0.025), min_sample_points)

    return x, y
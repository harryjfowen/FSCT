import os
import numpy as np

predict_parameters = dict(box_dims=np.array([1, 1, 1]),
                        box_overlap=np.array([0.5, 0.5, 0.5]),
                        min_pts=500,
                        max_pts=20000,
                        wood_class=0,
                        leaf_class=1,
                        terrain_class=2,
                        grid_resolution=0.5,
                        ground_height_threshold=.1,
                        subsample=True,
                        subsampling_min_spacing=0.01)

train_parameters = dict(learning_rate=0.000025,
                        box_dims=np.array([1, 1, 1]),
                        box_overlap=[0.50, 0.50, 0.50],
                        min_pts=1000,
                        max_pts=20000)

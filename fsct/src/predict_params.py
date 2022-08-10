import os
import numpy as np

other_parameters = dict(box_dims=np.array([2, 2, 2]),
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



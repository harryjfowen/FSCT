import os
import numpy as np

other_parameters = dict(box_dims=np.array([2, 2, 2]),
                        box_overlap=np.array([0.5, 0.5, 0.5]),
                        min_pts=1000,
                        max_pts=20000,
                        noise_class=0,
                        wood_class=1,
                        leaf_class=2,
                        terrain_class=3,
                        grid_resolution=0.5,
                        ground_height_threshold=.1,
                        num_neighbours=5,
                        slice_thickness=0.2,
                        slice_increment=0.05,
                        subsample=True,
                        subsampling_min_spacing=0.01,
                        min_tree_cyls=10, 
                        max_distance_between_tiles=np.inf)



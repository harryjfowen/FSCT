import os
import numpy as np

# Don't change these unless you really understand what you are doing with them/are learning the code base.
# These have been tuned to work on most high resolution forest point clouds without changing them.
other_parameters = dict(model_filename=os.path.join('model', 'modelV1.pth'),
                        box_dims=np.array([8, 8, 8]),
                        box_overlap=np.array([0.75, 0.75, 0.75]),
                        min_points_per_box=1000,
                        max_points_per_box=20000,
                        terrain_class=0,
                        vegetation_class=1,
                        cwd_class=2,
                        stem_class=3,
                        noise_class=4,
                        # terrain_class=0,
                        # wood_class=1,
                        # leaf_class=2,
                        # noise_class=3,
                        grid_resolution=0.5,
                        ground_height_threshold=.1,
                        num_neighbours=5,
                        slice_thickness=0.2,
                        slice_increment=0.05,
                        subsample=True,
                        subsampling_min_spacing=0.05,
                        min_tree_cyls=10, 
                        max_distance_between_tiles=np.inf)



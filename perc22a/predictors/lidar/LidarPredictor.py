""" LidarPredictor.py

This file contains the implementation of the perceptions predictions algorithm
that is solely dependent on raw LiDAR point clouds.
"""

# interface
from typing import List
from perc22a.predictors import Predictor

# data datatypes
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

# predict output datatype
from perc22a.predictors.utils.cones import Cones

# visualization and core lidar algorithm functions
import perc22a.predictors.utils.lidar.visualization as vis
import perc22a.predictors.utils.lidar.filter as filter
import perc22a.predictors.utils.lidar.cluster as cluster
import perc22a.predictors.utils.lidar.color as color

import numpy as np

class LidarPredictor(Predictor):

    def __init__(self):
        self.window = vis.init_visualizer_window()
        pass

    def required_data() -> List[DataType]:
        return [DataType.HESAI_POINTCLOUD]

    def _transform_points(self, points):
        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:,0] = -points[:,0]

        return points

    def predict(self, data: DataInstance) -> Cones:
        points = self._transform_points(data[DataType.HESAI_POINTCLOUD])
        self.points = points

        # remove all points with nan values
        points = points[~np.any(np.isnan(points), axis=-1)]

        # perform a box range on the data
        points_ground_plane = filter.box_range(
            points, xmin=-20, xmax=20, ymin=-20, ymax=20, zmin=-1, zmax=1)
        
        # vis.update_visualizer_window(None, points=points_ground_plane)

        # perform a plane fit and remove ground points
        xbound = 10
        points_filtered_ground, _, ground_planevals = filter.plane_fit(
            points, points_ground_plane, return_mask=True, boxdim=2, height_threshold=0.1)
        
        # perform another filtering algorithm to dissect boxed-region
        points_cluster, mask_cluster = filter.box_range(
            points_filtered_ground, xmin=-xbound, xmax=xbound, ymin=-10, ymax=50, zmin=-10, zmax=100, return_mask=True)

        # still over 200K points, so take subset to make DBSCAN faster
        points_cluster_subset = filter.random_subset(points_cluster, 0.03)

        # predict cones using a squashed point cloud and then unsquash
        cone_centers = cluster.predict_cones_z(
            points_cluster_subset, ground_planevals, hdbscan=False, dist_threshold=0.6, x_threshold_scale=0.15, height_threshold=0.3, scalar=1, x_bound=xbound, x_dist=3)

        # P, C = vis.color_matrix(fns=None, pcs=[points, points_filtered_ground, points_cluster])

        # color cones and return them
        cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)

        # correct the positions of the cones to the center of mass of the car
        cone_output = cluster.correct_clusters(cone_output)

        # visualize points
        vis.update_visualizer_window(self.window, points=points_cluster, pred_cones=cone_centers, colors_cones=cone_colors)

        # create a Cones object to return
        cones = Cones()
        for i in range(cone_output.shape[0]):
            x, y, c = cone_output[i, :]
            z = cone_centers[i, 2]
            if c == 1:
                cones.add_yellow_cone(x, y, z)
            elif c == 0:
                cones.add_blue_cone(x, y, z)

        return cones 

    def display(self):
        # vis.update_visualizer_window(self.window, self.points)
        
        return
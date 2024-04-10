""" LidarPredictor.py

This file contains the implementation of the perceptions predictions algorithm
that is solely dependent on raw LiDAR point clouds.
"""

import cProfile

import numpy as np
np.set_printoptions(threshold=np.inf)

# interface
from perc22a.predictors.interface.PredictorInterface import Predictor

# data datatypes
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

# predict output datatype
from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.transform.transform import PoseTransformations

# visualization and core lidar algorithm functions
import perc22a.predictors.utils.lidar.visualization as vis
import perc22a.predictors.utils.lidar.filter as filter
import perc22a.predictors.utils.lidar.cluster as cluster
import perc22a.predictors.utils.lidar.color as color

# timer utilities
from perc22a.utils.Timer import Timer

# constants
from perc22a.predictors.lidar.constants import *

# general imports
import numpy as np
from typing import List

class LidarPredictor(Predictor):
    def __init__(self):
        # self.window = vis.init_visualizer_window()
        self.sensor_name = "lidar"
        self.transformer = PoseTransformations()
        self.timer = Timer()
        self.filter_error_count = 0
        pass

    def profile_predict(self, data):
        profiler = cProfile.Profile()
        profiler.enable()
        cones = self.predict(data)
        profiler.disable()
        return cones, profiler
    def required_data(self):
        return [DataType.HESAI_POINTCLOUD]

    def _transform_points(self, points):
        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:, 0] = -points[:, 0]

        return points

    def predict(self, data) -> Cones:
        # coordinate frame points to perceptions coordinates system
        points = self._transform_points(data[DataType.HESAI_POINTCLOUD])
        points = points[~np.any(points == 0, axis=1)]
        points = points[~np.all(points == 0, axis=1)]
        points = points[~np.any(np.isnan(points), axis=-1)]
        points = points[:, :3]
        self.points = points

        # transfer to origin of car
        points = self.transformer.to_origin(self.sensor_name, points, inverse=False)

        points_ground_plane = filter.fov_range(
            points, 
            fov=180, 
            minradius=0, 
            maxradius=INIT_PC_MAX_RADIUS
        )

        # avoid crashing sometimes
        if points_ground_plane.shape[0] == 0:
            return Cones()

        points_filtered_ground = filter.GraceAndConrad(
            points_ground_plane, 
            points_ground_plane, 
            0.1, 
            10, 
            0.13
        )
       
        _, _, ground_planevals = filter.plane_fit(
            points,
            points_ground_plane,
            return_mask=True,
            boxdim=5,
            height_threshold=0.12,
        )

        points_cluster_subset = filter.voxel_downsample(
            points_filtered_ground, 
            DOWNSAMPLE_VOXEL_SIZE
        )

        # predict cone positions 
        cone_centers = cluster.predict_cones_z(
            points_cluster_subset,
            ground_planevals,
            height_threshold=0.5,
        )

        # color cones and return them
        cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)

        # correct the positions of the cones to the center of mass of the car
        cone_output = cluster.correct_clusters(cone_output)

        # create a Cones object to return
        cones = Cones()
        for i in range(cone_output.shape[0]):
            x, y, c = cone_output[i, :]
            z = cone_centers[i, 2]
            if c == 1:
                cones.add_yellow_cone(x, y, z)
            elif c == 0:
                cones.add_blue_cone(x, y, z)

        return self.transformer.transform_cones(self.sensor_name, cones)


    def display(self):
        return

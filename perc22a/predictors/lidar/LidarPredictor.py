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
from perc22a.predictors.utils.vis.Vis3D import Vis3D
import perc22a.predictors.utils.lidar.visualization as vis
import perc22a.predictors.utils.lidar.filter as filter
import perc22a.predictors.utils.lidar.cluster as cluster
import perc22a.predictors.utils.lidar.color as color
from perc22a.predictors.utils.lidar.ICPColorer import ICPColorer

# timer utilities
from perc22a.utils.Timer import Timer

# constants
from perc22a.predictors.lidar.constants import *

# general imports
import numpy as np
from typing import List

class LidarPredictor(Predictor):
    def __init__(self, debug=False):
        # self.window = vis.init_visualizer_window()
        self.sensor_name = "lidar"

        self.colorer = ICPColorer()
        self.transformer = PoseTransformations()
        self.timer = Timer()

        self.debug = debug
        if self.debug:
            self.use_old_vis = False 
            if self.use_old_vis:
                self.window = vis.init_visualizer_window()
            else:
                self.vis = Vis3D()

        return

    def required_data(self):
        return [DataType.HESAI_POINTCLOUD]

    def _transform_points(self, points):
        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:, 0] = -points[:, 0]

        return points

    def predict(self, data) -> Cones:
        if DEBUG_TIME: self.timer.start("predict")
        if DEBUG_TIME: self.timer.start("\tinit-process")

        # coordinate frame points to perceptions coordinates system
        points = data[DataType.HESAI_POINTCLOUD]
        points = points[~np.any(points == 0, axis=1)]
        points = points[~np.any(np.isnan(points), axis=-1)]
        points = points[:, :3]
        points = self._transform_points(points)
        self.points = points
        self.num_points = self.points.shape[0]

        # # transfer to origin of car
        points = self.transformer.to_origin(self.sensor_name, points, inverse=False)

        if DEBUG_TIME: self.timer.end("\tinit-process")
        if DEBUG_TIME: self.timer.start("\tfilter")
        if DEBUG_TIME: self.timer.start("\t\tfov-range")

        points_ground_plane = filter.fov_range(
            points, 
            fov=180, 
            minradius=0, 
            maxradius=INIT_PC_MAX_RADIUS
        )
        
        if DEBUG_TIME: self.timer.end("\t\tfov-range")
        if DEBUG_TIME: self.timer.start("\t\tground-removal")

        points_filtered_ground = filter.GraceAndConrad(
            points_ground_plane, 
            points_ground_plane, 
            SMART_GROUND_FILTER_SLICE_RADIANS, 
            SMART_GROUND_FILTER_SLICE_BINS, 
            SMART_GROUND_FILTER_HEIGHT_THRESHOLD
        )
       
        if DEBUG_TIME: self.timer.end("\t\tground-removal")
        if DEBUG_TIME: self.timer.start("\t\tnaive-plane-fit")

        _, _, ground_planevals = filter.plane_fit(
            points,
            points_ground_plane,
            return_mask=True,
            boxdim=NAIVE_PLANE_FIT_BOX_DIM,
            height_threshold=NAIVE_PLANE_FIT_HEIGHT_THRESHOLD,
        )

        if DEBUG_TIME: self.timer.end("\t\tnaive-plane-fit")
        if DEBUG_TIME: self.timer.start("\t\tvoxel-downsample")

        points_cluster_subset = filter.voxel_downsample(
            points_filtered_ground, 
            DOWNSAMPLE_VOXEL_SIZE
        )
        self.points_cluster_subset = points_cluster_subset

        if DEBUG_TIME: self.timer.end("\t\tvoxel-downsample")
        if DEBUG_TIME: self.timer.end("\tfilter")
        if VIS_PRE_CLUSTER_POINTS:
            vis.update_visualizer_window(None, points_cluster_subset)
        if DEBUG_TIME: self.timer.start("\tcluster")

        # predict cone position
        num_cluster_points = points_cluster_subset.shape[0] 
        cone_centers = cluster.predict_cones_z(
            points_cluster_subset,
            ground_planevals,
            height_threshold=MAX_CLUSTER_HEIGHT_THRESHOLD,
        )

        if DEBUG_TIME: self.timer.end("\tcluster", msg=f"({str(num_cluster_points)} points)")
        if DEBUG_TIME: self.timer.start("\tcoloring")

        # color cones and correct them
        cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)
        cone_output = cluster.correct_clusters(cone_output)
        self.cone_output_arr = cone_output
        self.cone_colors = cone_colors

        # create a Cones object to return
        cones = Cones()
        for i in range(cone_output.shape[0]):
            x, y, c = cone_output[i, :]
            z = cone_centers[i, 2]
            if c == 1:
                cones.add_yellow_cone(x, y, z)
            elif c == 0:
                cones.add_blue_cone(x, y, z)
        if DEBUG_TIME: self.timer.end("\tcoloring")

        self.cones = cones

        if DEBUG_TIME: self.timer.end("predict", msg=f"({self.num_points} points)")
        return cones


    def display(self):
        assert(self.debug)

        if self.use_old_vis:
            vis.update_visualizer_window(self.window, self.points_cluster_subset, self.cone_output_arr, self.cone_colors)
        else:
            self.vis.set_points(self.points_cluster_subset)
            self.vis.set_cones(self.cones)
            self.vis.update()

        return

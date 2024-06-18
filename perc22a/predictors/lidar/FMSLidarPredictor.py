""" LidarPredictor.py

This file contains the implementation of the perceptions predictions algorithm
that is solely dependent on raw LiDAR point clouds.

Functions:
    - init: initialize the predictor
    - profile_predict: profile the predict function
    - required_data: return the required data for the predictor
    - _transform_points: transform the points to the perceptions coordinate frame
    - predict: predict the cones from the LiDAR point cloud
    - display: display the transformed cones
"""

import cProfile

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

# timer utilities
from perc22a.utils.Timer import Timer

# constants
from perc22a.predictors.lidar.fms_constants import *

# general imports
import numpy as np
from typing import List

class FMSLidarPredictor(Predictor):
    def __init__(self, use_old_vis=True):
        # initialize the predictor with the sensor name, transformer, and timer and based on a flag for visualization

        self.sensor_name = "lidar"
        self.transformer = PoseTransformations()
        self.timer = Timer()
        self.use_old_vis = use_old_vis

        if self.use_old_vis:
            self.window = vis.init_visualizer_window()
        else:
            self.vis = Vis3D()

        return

    def profile_predict(self, data):
        # profile the predict function
            # Profiling is a method of measuring the performance of a program.

        profiler = cProfile.Profile()
        profiler.enable()
        cones = self.predict(data)
        profiler.disable()
        return cones, profiler

    def required_data(self):
        return [DataType.HESAI_POINTCLOUD]

    def _transform_points(self, points):
        # transform the points to the perceptions coordinate frame

        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:, 0] = -points[:, 0]

        return points

    def predict(self, data) -> Cones:
        '''
            The predict function takes in a data parameter and returns a Cones object.
            
            The predict function first initializes the points from the input data and transforms them to the perception coordinate system, 
            as well as filtering out invalid points. It then transfers the points to the origin of the car. After this, we apply a 
            field of view filter as well as box range filter to reduce the lookahead of the LiDAR to a more manageable range. We then perform 
            ground removal using the GraceAndConrad filter, which fits a plane to the points and extracts the ground plane values. We then 
            downsample the filtered points using voxel downsampling, making it easier to identify the cones and work with the clusters
            Then we predict the positions of the cones using an HDBSCAN clustering algorithm and color the cones based on their position.
        '''

        if DEBUG_TIME: self.timer.start("predict")
        if DEBUG_TIME: self.timer.start("\tinit-process")

        # coordinate frame points to perceptions coordinates system
        points = data[DataType.HESAI_POINTCLOUD]
        points = points[~np.any(points == 0, axis=1)]
        points = points[~np.any(np.isnan(points), axis=-1)]
        points = points[:, :3]
        points = self._transform_points(points)
        self.points = points

        # transfer to origin of car
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
        points_ground_plane = filter.box_range(
            points_ground_plane,
            xmin=-INIT_PC_BOX_RANGE,
            xmax=INIT_PC_BOX_RANGE
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

        # predict cone positions 
        cone_centers = cluster.predict_cones_z(
            points_cluster_subset,
            ground_planevals,
            height_threshold=MAX_CLUSTER_HEIGHT_THRESHOLD,
        )

        if DEBUG_TIME: self.timer.end("\tcluster", msg=str(len(points_cluster_subset)))
        if DEBUG_TIME: self.timer.start("\tcoloring")

        # color cones and correct them
        cone_output, cone_centers, cone_colors = color.color_cones(cone_centers)
        cone_output = cluster.correct_clusters(cone_output)
        self.cone_output_arr = cone_output

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
        if DEBUG_TIME: self.timer.start("\ttransform")

        cones = self.transformer.transform_cones(self.sensor_name, cones)
        self.cones = cones

        if DEBUG_TIME: self.timer.end("\ttransform")
        if DEBUG_TIME: self.timer.end("predict")
        return cones


    def display(self):

        if self.use_old_vis:
            vis.update_visualizer_window(self.window, self.points_cluster_subset, self.cone_output_arr)
        else:
            self.vis.set_points(self.points_cluster_subset)
            self.vis.set_cones(self.cones)
            self.vis.update()

        return

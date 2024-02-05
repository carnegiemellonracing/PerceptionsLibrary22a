'''Vis3D.py

Visualization class wrapping Open3D for displaying point clouds and cones
'''

from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.lidar.visualization import create_axis_vis, \
    update_visualizer_window, \
    update_visualizer_perspective, \
    create_cylinder_vis, \
    EXTRINSIC_BEHIND

import numpy as np
import open3d as o3d

class Vis3D:

    def __init__(self):

        # initialize display-able objects
        self.points = None
        self.cones = None

        # initialize geometry objects to visualize
        self.axis_vis = create_axis_vis()
        self.cones_vis = []

        # initialize a random point cloud so that display is correct
        init_points = np.random.rand(1000, 3) * 100  
        self.points_vis = o3d.geometry.PointCloud()
        self.points_vis.points = o3d.utility.Vector3dVector(init_points)

        # initialize window to visualize in, set perspectives, and add objects
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.vis.add_geometry(self.axis_vis)
        self.vis.add_geometry(self.points_vis)

        # initialize visualizer perspective
        update_visualizer_perspective(self.vis, EXTRINSIC_BEHIND)

        return


    def set_points(self, points: np.ndarray):
        '''sets the point cloud to visualize on next .display call'''
        self.points = points

    def set_cones(self, cones: Cones):
        '''sets the cones to visualize on next .display call'''
        self.cones = cones

    def _update_points(self):
        '''updates 3D visualization with latest points'''
        if self.points is None:
            return
        
        # remove any all zero points in the pointcloud
        self.points = self.points[np.any(self.points != 0, axis=1)][:,:3]

        # modify the pointcloud geometry        
        self.points_vis.points.clear()
        self.points_vis.points.extend(self.points)

        # update geometry in visualization
        self.vis.update_geometry(self.points_vis)

    def _update_cones(self):
        '''updates 3D visualization with latest points'''
        if self.cones is None:
            return
        
        # create geometry for each cone
        cones  = self.cones.to_numpy()
        blue_cones_arr, yellow_cones_arr, orange_cones_arr = cones

        blue_cylinders = create_cylinder_vis(blue_cones_arr, colors=[0, 0, 1])
        yellow_cylinders = create_cylinder_vis(yellow_cones_arr, colors=[1, 0, 0])
        orange_cylinders = create_cylinder_vis(orange_cones_arr, colors=[0, 1, 0])

        # remove old cone geometries
        for cone_vis in self.cones_vis:
            self.vis.remove_geometry(cone_vis, reset_bounding_box=False)
        self.cones_vis = []

        # add new cone geometries
        self.cones_vis = blue_cylinders + yellow_cylinders + orange_cylinders
        for cone_vis in self.cones_vis:
            self.vis.add_geometry(cone_vis, reset_bounding_box=False)

        # reset cones
        self.cones = None


    def update(self):
        '''updates 3D visualization with latest objects'''

        # update geometries
        self._update_points()
        self._update_cones()

        # poll events and update view
        self.vis.update_renderer()
        self.vis.poll_events()
        pass
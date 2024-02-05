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

GRID_RIGHT_BOUND = 10
GRID_FRONT_BOUND = 10
INTERVAL = 1

LEFT_POINTS = [[-GRID_RIGHT_BOUND, y, 0] for y in np.arange(0, GRID_FRONT_BOUND, step=INTERVAL)]
RIGHT_POINTS = [[GRID_RIGHT_BOUND, y, 0] for y in np.arange(0, GRID_FRONT_BOUND, step=INTERVAL)]
BACK_POINTS = [
    [x, 0, 0] for x in np.arange(0, GRID_RIGHT_BOUND, step=INTERVAL)
] + [
    [-x, 0, 0] for x in np.arange(0, GRID_RIGHT_BOUND, step=INTERVAL)
]
FRONT_POINTS = [
    [x, GRID_FRONT_BOUND, 0] for x in np.arange(0, GRID_RIGHT_BOUND, step=INTERVAL)
] + [
    [-x, GRID_FRONT_BOUND, 0] for x in np.arange(0, GRID_RIGHT_BOUND, step=INTERVAL)
]

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
        self._init_grid()

        # initialize visualizer perspective
        update_visualizer_perspective(self.vis, EXTRINSIC_BEHIND)

        # clear original point cloud to avoid random stuff
        self.points_vis.points.clear()
        self.vis.update_geometry(self.points_vis)

        return
    
    def _init_grid(self):
        assert (len(BACK_POINTS) == len(FRONT_POINTS))
        assert (len(LEFT_POINTS) == len(RIGHT_POINTS))

        # construct vertical grid lines
        vertical_points = BACK_POINTS + FRONT_POINTS
        vertical_lines = [[i, i + len(BACK_POINTS)] for i in range(len(BACK_POINTS))]
        vertical_colors = [[0.25, 0.25, 0.25] for i in range(len(BACK_POINTS))]
        self.vertical_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertical_points),
            lines=o3d.utility.Vector2iVector(vertical_lines),
        )
        self.vertical_lineset.colors = o3d.utility.Vector3dVector(vertical_colors)

        # construct horizontal grid lines
        horizontal_points = LEFT_POINTS + RIGHT_POINTS
        horizontal_lines = [[i, i + len(RIGHT_POINTS)] for i in range(len(RIGHT_POINTS))]
        horizontal_colors = [[0.25, 0.25, 0.25] for i in range(len(RIGHT_POINTS))]
        self.horizontal_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(horizontal_points),
            lines=o3d.utility.Vector2iVector(horizontal_lines),
        )
        self.horizontal_lineset.colors = o3d.utility.Vector3dVector(horizontal_colors)

        # visualize the lineset
        self.vis.add_geometry(self.vertical_lineset)
        self.vis.add_geometry(self.horizontal_lineset)


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
""" LidarPredictor.py

This file contains the implementation of the perceptions predictions algorithm
that is solely dependent on raw LiDAR point clouds.
"""

# interface
from perc22a.predictors import Predictor

# visualization functions
import perc22a.predictors.utils.lidar.visualization  as vis


class LidarPredictor(Predictor):

    def __init__(self):
        self.window = vis.init_visualizer_window()
        pass

    def _transform_points(self, points):
        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:,0] = -points[:,0]

        return points

    def predict(self, data):
        points = self._transform_points(data["points"])
        self.points = points

        pass

    def display(self):
        vis.update_visualizer_window(self.window, self.points)
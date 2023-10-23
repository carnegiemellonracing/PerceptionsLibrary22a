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

    def predict(self, data):
        self.points = data["points"]
        pass

    def display(self):
        points = self.points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:,0] = -points[:,0]
        vis.update_visualizer_window(self.window, self.points[:,:3])
        pass
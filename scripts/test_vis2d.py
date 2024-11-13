from perc22a.predictors.utils.vis.Vis2D import Vis2D

from perc22a.data.utils.dataloader import DataLoader
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

import matplotlib.pyplot as plt
import numpy as np
import time
import threading

def main():

    dl = DataLoader("perc22a/data/raw/hybrid-2-3")
    vis = Vis2D()
    lp = LidarPredictor()

    
    for i in range(50, len(dl)):
        # data = dl[i]
        # points = data[DataType.HESAI_POINTCLOUD]

        # points = points[:, :3]
        # points = points[:, [1, 0, 2]]
        # points[:, 0] = -points[:, 0]

        cones = lp.predict(dl[i])

        # vis.set_points(points)
        vis.set_cones(cones)
        vis.update()

if __name__ == "__main__":
    main()
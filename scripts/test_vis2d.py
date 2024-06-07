from perc22a.predictors.utils.vis.Vis2D import Vis2D

from perc22a.data.utils.dataloader import DataLoader
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

import matplotlib.pyplot as plt
import numpy as np
import time
import threading

def main():

    dl = DataLoader("perc22a/data/raw/tt-09-29")
    vis = Vis2D()
    yp = YOLOv5Predictor(camera="zed")

    
    for i in range(50, len(dl)):
        data = dl[i]
        points = data[DataType.HESAI_POINTCLOUD]

        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:, 0] = -points[:, 0]

        cones = yp.predict(data)
        print(cones)

        vis.set_points(points)
        vis.set_cones(cones)
        vis.update()

if __name__ == "__main__":
    main()
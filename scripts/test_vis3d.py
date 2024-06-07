from perc22a.predictors.utils.vis.Vis3D import Vis3D

from perc22a.data.utils.dataloader import DataLoader
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType


from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.utils.Timer import Timer

from perc22a.predictors.utils.ConeState import ConeState

from perc22a.svm.SVM import SVM


import open3d as o3d
import numpy as np
import time
import threading

def main():
    # yp = YOLOv5Predictor(camera="zed")
    lp = LidarPredictor()
    vis = Vis3D()
    merger = BaseMerger(required_pipelines=[], debug=True, zed_dist_limit=20, lidar_dist_limit=20)

    state = ConeState()

    dl = DataLoader("perc22a/data/raw/hybrid-2-3")
    svm = SVM()

    while True:
        for i in range(550, len(dl)):
            data = dl[i]
            # import pdb; pdb.set_trace()
            points = data[DataType.HESAI_POINTCLOUD]

            points = points[:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] = -points[:, 0]

            cones = lp.predict(data)
            merger.add(cones, PipelineType.LIDAR)
            cones = merger.merge() 
            merger.reset()

            cones = svm.recolor(cones)

            midline_points = svm.cones_to_midline(cones)

            midline_points = np.hstack((midline_points, np.zeros((midline_points.shape[0], 1))))

            vis.set_points(points)
            vis.set_cones(cones)
            vis.set_midline(midline_points)

            vis.update()


if __name__ == "__main__":
    main()
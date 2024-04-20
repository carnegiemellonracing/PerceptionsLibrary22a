from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType

from perc22a.data.utils.dataloader import DataLoader

from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.utils.Timer import Timer

from perc22a.predictors.utils.ConeState import ConeState

from perc22a.svm.SVM import SVM

import numpy as np

def main():
    lp = LidarPredictor()
    t = Timer()
    vis = Vis2D()

    # create merger
    merger = BaseMerger(required_pipelines=[], debug=True, zed_dist_limit=20, lidar_dist_limit=20)

    state = ConeState()

    dl = DataLoader("perc22a/data/raw/tt-4-6-lidar")
    svm = SVM()

    for i in range(40, len(dl)):
        # perform prediction
        cones = lp.predict(dl[i])

        # merge the cones together from other pipelines
        # TODO: is this necessary now?
        merger.add(cones, PipelineType.LIDAR)
        cones = merger.merge() 
        merger.reset()

        # color the cones
        cones = svm.recolor(cones)

        cones = state.update(cones)

        # convert cones to SVM midline points
        midline_points = svm.cones_to_midline(cones)

        vis.set_cones(cones)
        if len(midline_points) > 0:
            vis.set_points(midline_points)
            vis.update()        



if __name__ == "__main__":
    main()

from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType

from perc22a.data.utils.dataloader import DataLoader

from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.utils.Timer import Timer

from perc22a.svm.svm_utils import cones_to_midline, augment_dataset, augment_cones

def main():
    sp1 = YOLOv5Predictor(camera="zed")
    sp2 = YOLOv5Predictor(camera="zed2")
    lp = LidarPredictor()
    t = Timer()
    vis = Vis2D()

    # create merger
    merger = BaseMerger(required_pipelines=[], debug=True, zed_dist_limit=10, lidar_dist_limit=10)

    dl = DataLoader("perc22a/data/raw/three-laps-large")

    for i in range(20, len(dl)):
        cones_zed = sp1.predict(dl[i])
        cones_zed2 = sp2.predict(dl[i])
        cones_lidar = lp.predict(dl[i])

        merger.add(cones_zed, PipelineType.ZED_PIPELINE)
        merger.add(cones_zed2, PipelineType.ZED2_PIPELINE)
        merger.add(cones_lidar, PipelineType.LIDAR)
        merged_cones = merger.merge()
        
        t.start("spline")
        midline_points = cones_to_midline(merged_cones)
        t.end("spline")

        vis.set_cones(merged_cones)

        if len(midline_points) > 0:
            vis.set_points(midline_points)
            vis.update()        

        merger.reset()


if __name__ == "__main__":
    main()

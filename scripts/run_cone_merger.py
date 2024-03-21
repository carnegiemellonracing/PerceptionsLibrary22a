from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType

from perc22a.data.utils.dataloader import DataLoader

from perc22a.utils.Timer import Timer


def main():
    sp1 = YOLOv5Predictor(camera="zed")
    sp2 = YOLOv5Predictor(camera="zed2")
    lp = LidarPredictor()
    t = Timer()

    # create merger
    merger = BaseMerger(required_pipelines=[], debug=True)

    dl = DataLoader("perc22a/data/raw/three-laps-large")

    for i in range(40, len(dl)):
        cones_zed = sp1.predict(dl[i])
        cones_zed2 = sp2.predict(dl[i])
        cones_lidar = lp.predict(dl[i])

        merger.add(cones_zed, PipelineType.ZED_PIPELINE)
        merger.add(cones_zed2, PipelineType.ZED2_PIPELINE)
        merger.add(cones_lidar, PipelineType.LIDAR)

        t.start("merge")
        merged_cones = merger.merge()
        t.end("merge")

        merger.display()
        merger.reset()


if __name__ == "__main__":
    main()

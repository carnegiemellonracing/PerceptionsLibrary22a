from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType

from perc22a.data.utils.dataloader import DataLoader

from perc22a.utils.Timer import Timer

import matplotlib.pyplot as plt


def main():
    sp1 = YOLOv5Predictor(camera="zed")
    sp2 = YOLOv5Predictor(camera="zed2")
    lp = LidarPredictor()
    t = Timer()

    # create merger
    merger = BaseMerger(required_pipelines=[], debug=True, zed_dist_limit=10, lidar_dist_limit=10)
    naive_merger = BaseMerger(required_pipelines=[], zed_dist_limit=10, lidar_dist_limit=10)

    dl = DataLoader("perc22a/data/raw/three-laps-large")

    for i in range(40, len(dl)):
        cones_zed = sp1.predict(dl[i])
        cones_zed2 = sp2.predict(dl[i])
        cones_lidar = lp.predict(dl[i])

        merger.add(cones_zed, PipelineType.ZED_PIPELINE)
        merger.add(cones_zed2, PipelineType.ZED2_PIPELINE)
        merger.add(cones_lidar, PipelineType.LIDAR)
        naive_merger.add(cones_zed, PipelineType.ZED_PIPELINE)
        naive_merger.add(cones_zed2, PipelineType.ZED2_PIPELINE)
        naive_merger.add(cones_lidar, PipelineType.LIDAR)

        merged_cones = merger.merge()
        naive_merged_cones = naive_merger._naive_merge()
        
        merger.reset()
        naive_merger.reset()

        # print(merged_cones)

        fig, axes = plt.subplots(1, 2)
        merged_cones.plot2d(axes[0], show=False, title="merged")
        # naive_merged_cones.plot2d(axes[1], title="naive")
        cones_zed.plot2d(axes[1], show=False, label="Z")
        cones_zed2.plot2d(axes[1], show=False, label="Y")
        cones_lidar.plot2d(axes[1], show=True, label="L")


if __name__ == "__main__":
    main()

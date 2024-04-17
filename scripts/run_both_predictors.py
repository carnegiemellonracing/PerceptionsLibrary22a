import sys
import os
import numpy as np

from perc22a.predictors.stereo.StereoPredictor import StereoPredictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.data.utils.dataloader import DataLoader

def main():
    sp = StereoPredictor('ultralytics/yolov5', 'perc22a/predictors/stereo/model_params.pt')
    lp = LidarPredictor()
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    for i in range(len(dl)):
        stereo_cones = sp.predict(dl[i])
        lidar_cones = lp.predict(dl[i])
        print(stereo_cones)
        sp.display()
        lp.display()


if __name__ == "__main__":
    main()

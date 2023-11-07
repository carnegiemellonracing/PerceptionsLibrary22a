import sys
import os
import numpy as np

from perc22a.predictors.stereo.StereoPredictor import StereoPredictor
from perc22a.data.utils.dataloader import DataLoader

def main():
    sp = StereoPredictor('ultralytics/yolov5', 'perc22a/predictors/stereo/model_params.pt')
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    for i in range(len(dl)):
        cones = sp.predict(dl[i])
        print(cones)
        sp.display()


if __name__ == "__main__":
    main()

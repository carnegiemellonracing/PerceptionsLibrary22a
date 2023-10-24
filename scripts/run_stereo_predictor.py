import sys
import os
import numpy as np

from perc22a.predictors.StereoPredictor.StereoPredictor import StereoPredictor
from perc22a.data.utils.dataloader import DataLoader


import time
import cv2

def main():
    predictor = StereoPredictor('ultralytics/yolov5', 'perc22a/predictors/StereoPredictor/model_params.pt')
    dl = DataLoader("data/raw/track-testing-09-29")
    # print(dl[0]['xyz_image'])

    for i in range(len(dl)):
        #print(dl[i]['points'])

        predictor.predict(dl[i])


if __name__ == "__main__":
    main()

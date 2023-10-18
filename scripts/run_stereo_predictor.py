import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from predictors.StereoPredictor.StereoPredictor import StereoPredictor
from data.utils.dataloader import DataLoader


import time
import cv2

def main():
    predictor = StereoPredictor('ultralytics/yolov5', 'predictors/StereoPredictor/model_params.pt')
    dl = DataLoader("data/raw/track-testing-09-29")

    for i in range(len(dl)):
        print(dl[i])

        predictor.predict(dl[i])


if __name__ == "__main__":
    main()

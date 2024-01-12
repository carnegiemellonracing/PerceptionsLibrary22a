import sys
import os
import numpy as np
import cv2

from perc22a.predictors.stereo.StereoPredictor import StereoPredictor
from perc22a.data.utils.dataloader import DataLoader

def main():
    sp = StereoPredictor('ultralytics/yolov5', 'perc22a/predictors/stereo/model_params.pt')
    dl = DataLoader("perc22a/data/raw/12-02-ecg-track-test")
    i = 700
    # display the image
    print(i)
    img = dl[i]["left_color"]
    cv2.imshow(f"left image", img)
    cv2.waitKey(5000)
    # cones = sp.predict(dl[i])
    # print(cones)
    # sp.display()


if __name__ == "__main__":
    main()
